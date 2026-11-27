"""Phase 2: records -> images (§3.2), written straight into training-ready arrays.

One instruction = one pixel, in program order, on the smallest square that holds all
pixels, zero padded. Each image is resized once here for every requested input size and
stored in a memory-mapped array <set>/<features>_<size>.npy of shape (N, S, S, 3), so
training never decodes multi-megapixel PNGs again. Full-size PNGs are optional.

features:
  rgb  R = opcode, G = op1 registers, B = op2 method tree (the paper)
  tri  the three 3-gram channels, aligned pixel for pixel with rgb
Training can use rgb, tri, or both stacked as a 6-channel input ("hybrid").
"""
import csv
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from .records import Record
from .smali import method_path
from .splits import SPLITS, apply_splits, load_splits, read_manifest
from .utils import Resources, append_csv, log, parallel_map, summarize, write_json
from .vocab import Vocab, record_trigram_hashes, trigram_keys

_VOCAB = None
_ARRAYS = {}


def _vocab(path, scale):
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = Vocab(path, scale)
    return _VOCAB


def encode_rgb(rec, vocab):
    op_t, reg_t, arg_t = rec.tables
    op_v = vocab.scaler("op")([vocab.op.get(s, 0) for s in op_t])
    reg_v = vocab.scaler("reg")([np.mean([vocab.reg.get(r, 0) for r in s.split("|")]) for s in reg_t])
    arg_freq = []
    for s in arg_t:
        nodes = method_path(s, vocab.path_depth)
        arg_freq.append(np.mean([vocab.path.get("/".join(nodes[:d]), 0) for d in range(1, len(nodes) + 1)]))
    arg_v = vocab.scaler("path")(arg_freq)
    return np.stack([op_v[rec.idx[0]], reg_v[rec.idx[1]], arg_v[rec.idx[2]]], axis=1)


def encode_trigrams(rec, vocab):
    if vocab.tri is None:
        raise SystemExit("This vocabulary was fitted without --trigrams")
    out = np.zeros((len(rec), 3), dtype=np.uint8)
    for k, h in enumerate(record_trigram_hashes(rec, vocab.coarse_depth)):
        keys = trigram_keys(h)
        vkeys, vcounts = vocab.tri[k]
        pos = np.searchsorted(vkeys, keys)
        pos = np.minimum(pos, max(len(vkeys) - 1, 0))
        counts = np.where(vkeys[pos] == keys, vcounts[pos], 0) if len(vkeys) else np.zeros(keys.size)
        out[2:, k] = vocab.scaler(f"tri{k}")(counts)
    return out


def to_square(pixels):
    n = pixels.shape[0]
    side = math.ceil(math.sqrt(n))
    img = np.zeros((side * side, pixels.shape[1]), dtype=np.uint8)
    img[:n] = pixels
    return img.reshape(side, side, pixels.shape[1])


def resize(img, size):
    return np.asarray(Image.fromarray(img, "RGB").resize((size, size), Image.BILINEAR))


def _array(path):
    arr = _ARRAYS.get(path)
    if arr is None:
        arr = _ARRAYS[path] = np.load(path, mmap_mode="r+")
    return arr


def _encode_task(task, out_dir, vocab_dir, scale, features, sizes, full_png):
    i, row = task
    start = time.perf_counter()
    vocab = _vocab(vocab_dir, scale)
    rec = Record(row["record"])
    side = 0
    for feat in features:
        pixels = encode_rgb(rec, vocab) if feat == "rgb" else encode_trigrams(rec, vocab)
        img = to_square(pixels)
        side = img.shape[0]
        if full_png:
            dest = Path(out_dir) / "full" / feat / f"{row['sha256']}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img, "RGB").save(dest)
        for s in sizes:
            _array(str(Path(out_dir) / f"{feat}_{s}.npy"))[i] = resize(img, s)
    return i, side, time.perf_counter() - start


def build_set(rows, out_dir, args):
    res = Resources()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for feat in args.features:
        for s in args.sizes:
            np.lib.format.open_memmap(out_dir / f"{feat}_{s}.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(rows), s, s, 3))
    task = partial(_encode_task, out_dir=str(out_dir), vocab_dir=args.vocab, scale=args.scale,
                   features=args.features, sizes=args.sizes, full_png=args.full_png)
    sides = [0] * len(rows)
    times = []
    for i, side, sec in parallel_map(task, list(enumerate(rows)), args.jobs, desc=f"images {out_dir.name}"):
        sides[i] = side
        times.append(sec)
    np.save(out_dir / "labels.npy", np.array([r["label"] for r in rows], dtype=np.int64))
    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sha256", "label", "year", "instructions", "side"])
        for r, side in zip(rows, sides):
            w.writerow([r["sha256"], r["label"], r.get("year", ""), r["instructions"], side])
    write_json({"apks": len(rows), "features": args.features, "sizes": args.sizes,
                "vocab": str(Path(args.vocab).resolve()), "scale": args.scale or Vocab(args.vocab).scale,
                "seconds_per_apk": summarize(times)}, out_dir / "images.json")
    append_csv(out_dir.parent / "resources.csv", res.snapshot(f"images {out_dir.name}", apks=len(rows),
                                                              jobs=args.jobs))


def run(args):
    rows = read_manifest(args.manifest)
    if args.splits:
        parts = apply_splits(rows, load_splits(args.splits))
        for name in SPLITS:
            log.info("%s: %d APKs", name, len(parts[name]))
            build_set(parts[name], Path(args.output) / name, args)
    else:
        if args.exclude_splits:  # external test set must not overlap DroidTrain
            known = {s for part in load_splits(args.exclude_splits).values() for s in part}
            overlap = [r["sha256"] for r in rows if r["sha256"] in known]
            if overlap:
                raise SystemExit(f"{len(overlap)} APKs of this set are also in the training data")
        build_set(rows, Path(args.output) / args.name, args)
    if args.delete_records:
        _delete_records(parts.values() if args.splits else [rows])


def _delete_records(groups):
    """Drop the .npz records once every image is written: only the images stay on disk."""
    freed = removed = 0
    for rows in groups:
        for r in rows:
            p = Path(r["record"])
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
                removed += 1
    log.info("deleted %d records (%.1f GB freed)", removed, freed / 1e9)
