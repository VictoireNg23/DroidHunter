"""Manifests and the stratified 80/10/10 partition (§5.1 of the paper)."""
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

from .utils import log, read_json, write_json

SPLITS = ("train", "val", "test")


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"Empty manifest: {path}")
    for r in rows:
        r["label"] = int(r["label"])
        r["instructions"] = int(r["instructions"])
    return rows


def make_splits(rows, seed, val=0.1, test=0.1):
    ids = list(range(len(rows)))
    y = [r["label"] for r in rows]
    train, hold = train_test_split(ids, test_size=val + test, stratify=y, random_state=seed)
    val_ids, test_ids = train_test_split(hold, test_size=test / (val + test),
                                         stratify=[y[i] for i in hold], random_state=seed)
    return {"train": [rows[i]["sha256"] for i in train],
            "val": [rows[i]["sha256"] for i in val_ids],
            "test": [rows[i]["sha256"] for i in test_ids]}


def apply_splits(rows, splits):
    by_sha = {r["sha256"]: r for r in rows}
    missing = [s for part in splits.values() for s in part if s not in by_sha]
    if missing:
        raise SystemExit(f"{len(missing)} APKs of splits.json are absent from the manifest")
    return {k: [by_sha[s] for s in splits[k]] for k in SPLITS}


def run(args):
    rows = read_manifest(args.manifest)
    if args.max_instructions:
        before = len(rows)
        rows = [r for r in rows if r["instructions"] <= args.max_instructions]
        log.info("dropped %d APKs above %d instructions", before - len(rows), args.max_instructions)
    splits = make_splits(rows, args.seed, args.val, args.test)
    counts = {}
    by_sha = {r["sha256"]: r["label"] for r in rows}
    for k, part in splits.items():
        counts[k] = {"total": len(part), "malware": sum(by_sha[s] for s in part)}
    write_json({"seed": args.seed, "manifest": str(Path(args.manifest).resolve()), "counts": counts, **splits},
               args.output)
    log.info("splits: %s", counts)


def load_splits(path):
    data = read_json(path)
    return {k: data[k] for k in SPLITS}
