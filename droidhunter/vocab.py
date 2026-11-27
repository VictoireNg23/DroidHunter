"""Frequency lookup tables (Op_dict / Arg_dict of §3.2), learnt on the training split only.

Channels:
  R  opcode frequency
  G  op1: mean frequency of the registers of the instruction
  B  op2: method tree, mean frequency of the path prefixes (package -> class -> member -> signature)
Optional 3-gram channels (one value per instruction, from the 3-gram ending at it):
  T0 opcode 3-gram, T1 op1 3-gram, T2 coarse op2 3-gram
Frequencies become 0-255 intensities with min-max scaling over the training vocabulary.
"""
import hashlib
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

from .records import Record
from .smali import coarse_operand, method_path
from .splits import apply_splits, load_splits, read_manifest
from .utils import Resources, append_csv, chunked, log, parallel_map, read_json, write_json

MIX = np.uint64(0x9E3779B97F4A7C15)
FORMAT_VERSION = 1


def stable_hash(strings):
    """64-bit hash that is identical across processes and runs (unlike hash())."""
    return np.fromiter((int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
                        for s in strings), dtype=np.uint64, count=len(strings))


def trigram_keys(token_hashes):
    h = token_hashes
    if h.size < 3:
        return np.empty(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return ((h[:-2] * MIX) ^ h[1:-1]) * MIX ^ h[2:]


def record_trigram_hashes(rec, coarse_depth):
    """Per-instruction token hashes for the three 3-gram channels."""
    tables = [rec.tables[0], rec.tables[1], [coarse_operand(s, coarse_depth) for s in rec.tables[2]]]
    return [stable_hash(list(t))[i] for t, i in zip(tables, rec.idx)]


def _count_chunk(records, path_depth, trigrams, coarse_depth):
    op, reg, path = Counter(), Counter(), Counter()
    tri = [[], [], []]
    for rec_path in records:
        rec = Record(rec_path)
        for k, counter in ((0, op), (1, reg)):
            for s, c in zip(rec.tables[k], rec.counts(k)):
                if k == 1:
                    for r in s.split("|"):
                        counter[r] += int(c)
                else:
                    counter[s] += int(c)
        for s, c in zip(rec.tables[2], rec.counts(2)):
            nodes = method_path(s, path_depth)
            for d in range(1, len(nodes) + 1):
                path["/".join(nodes[:d])] += int(c)
        if trigrams:
            for k, h in enumerate(record_trigram_hashes(rec, coarse_depth)):
                tri[k].append(trigram_keys(h))
    tri_out = []
    if trigrams:
        for k in range(3):
            keys = np.concatenate(tri[k]) if tri[k] else np.empty(0, np.uint64)
            tri_out.append(np.unique(keys, return_counts=True))
    return op, reg, path, tri_out


def _reduce_pairs(pairs):
    keys = np.concatenate([k for k, _ in pairs])
    counts = np.concatenate([c for _, c in pairs])
    uniq, inv = np.unique(keys, return_inverse=True)
    return uniq, np.bincount(inv, weights=counts).astype(np.int64)


class Vocab:
    def __init__(self, path, scale=None):
        path = Path(path)
        self.meta = read_json(path / "vocab.json")
        if scale:
            self.meta["scale"] = scale
        with np.load(path / "vocab.npz", allow_pickle=False) as z:
            self.op = dict(zip(z["op_keys"].tolist(), z["op_counts"].tolist()))
            self.reg = dict(zip(z["reg_keys"].tolist(), z["reg_counts"].tolist()))
            self.path = dict(zip(z["path_keys"].tolist(), z["path_counts"].tolist()))
            self.tri = [(z[f"tri{k}_keys"], z[f"tri{k}_counts"]) for k in range(3)] if "tri0_keys" in z else None
        self.scale = self.meta["scale"]
        self.path_depth = self.meta["path_depth"]
        self.coarse_depth = self.meta["coarse_depth"]
        self._scalers = {
            "op": Scaler(np.fromiter(self.op.values(), float), self.scale),
            "reg": Scaler(np.fromiter(self.reg.values(), float), self.scale),
            "path": Scaler(np.fromiter(self.path.values(), float), self.scale),
        }
        if self.tri:
            for k, (_, c) in enumerate(self.tri):
                self._scalers[f"tri{k}"] = Scaler(c.astype(float), self.scale)

    def scaler(self, name):
        return self._scalers[name]


class Scaler:
    """Frequency -> uint8 intensity. A frequency of 0 (unseen in training) maps to 0."""

    def __init__(self, counts, mode):
        if counts.size == 0:
            raise ValueError("empty vocabulary")
        self.mode = mode
        self.sorted = np.sort(counts)
        self.lo, self.hi = float(self.sorted[0]), float(self.sorted[-1])

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if self.mode == "rank":
            v = np.searchsorted(self.sorted, x, side="right") / self.sorted.size
        elif self.mode == "log":
            lo, hi = np.log1p(self.lo), np.log1p(self.hi)
            v = (np.log1p(x) - lo) / (hi - lo) if hi > lo else np.ones_like(x)
        else:  # linear, as written in the paper
            v = (x - self.lo) / (self.hi - self.lo) if self.hi > self.lo else np.ones_like(x)
        out = np.clip(np.rint(v * 255), 0, 255).astype(np.uint8)
        out[x <= 0] = 0
        return out


def run(args):
    res = Resources()
    rows = read_manifest(args.manifest)
    train = apply_splits(rows, load_splits(args.splits))["train"]
    records = [r["record"] for r in train]
    log.info("fitting vocabulary on %d training APKs", len(records))
    task = partial(_count_chunk, path_depth=args.path_depth, trigrams=args.trigrams, coarse_depth=args.coarse_depth)
    op, reg, path = Counter(), Counter(), Counter()
    tri_parts = [[], [], []]
    for c_op, c_reg, c_path, c_tri in parallel_map(task, chunked(records, args.chunk), args.jobs, desc="vocab"):
        op.update(c_op)
        reg.update(c_reg)
        path.update(c_path)
        for k, pair in enumerate(c_tri):
            tri_parts[k].append(pair)
            if len(tri_parts[k]) >= 16:  # keep memory bounded
                tri_parts[k] = [_reduce_pairs(tri_parts[k])]

    if args.min_count > 1:
        path = Counter({k: v for k, v in path.items() if v >= args.min_count})
    arrays = {
        "op_keys": np.array(list(op), dtype=str), "op_counts": np.array(list(op.values()), dtype=np.int64),
        "reg_keys": np.array(list(reg), dtype=str), "reg_counts": np.array(list(reg.values()), dtype=np.int64),
        "path_keys": np.array(list(path), dtype=str), "path_counts": np.array(list(path.values()), dtype=np.int64),
    }
    sizes = {"op": len(op), "reg": len(reg), "path": len(path)}
    if args.trigrams:
        for k in range(3):
            keys, counts = _reduce_pairs(tri_parts[k])
            keep = counts >= args.min_count
            arrays[f"tri{k}_keys"], arrays[f"tri{k}_counts"] = keys[keep], counts[keep]
            sizes[f"tri{k}"] = int(keep.sum())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "vocab.npz", **arrays)
    write_json({"version": FORMAT_VERSION, "scale": args.scale, "path_depth": args.path_depth,
                "coarse_depth": args.coarse_depth, "min_count": args.min_count, "trigrams": args.trigrams,
                "training_apks": len(records), "sizes": sizes, "splits": str(Path(args.splits).resolve())},
               out / "vocab.json")
    append_csv(out / "resources.csv", res.snapshot("vocab", apks=len(records), jobs=args.jobs))
    log.info("vocabulary sizes: %s", sizes)
