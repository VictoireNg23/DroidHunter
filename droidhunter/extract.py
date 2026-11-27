"""Phase 1: disassemble APKs with apktool and store instruction records.

Parallel inside a node (one apktool per worker) and across nodes (--shard i --num-shards N).
Resumable: APKs whose record already exists are skipped, so a killed OAR job can simply
be resubmitted. APKs are never executed.
"""
import csv
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from functools import partial
from pathlib import Path

from .records import RecordBuilder
from .smali import iter_smali_instructions, load_opcodes
from .utils import Resources, append_csv, log, parallel_map, summarize, write_json

MANIFEST_FIELDS = ["sha256", "label", "year", "apk", "record", "instructions", "seconds", "status", "error"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def list_apks(sources, year_regex=None):
    """sources: [(directory, label)] -> sorted [(path, label, year)]."""
    pattern = re.compile(year_regex) if year_regex else None
    found = []
    for directory, label in sources:
        directory = Path(directory)
        if not directory.is_dir():
            raise SystemExit(f"Directory not found: {directory}")
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".apk":
                year = ""
                if pattern:
                    m = pattern.search(str(p.relative_to(directory)))
                    year = m.group(1) if m else ""
                found.append((str(p), label, year))
    found.sort()
    return found


def extract_one(item, out_dir, opcodes, apktool, apktool_args, tmp_root, timeout):
    path, label, year = item
    row = {"label": label, "year": year, "apk": path, "record": "", "instructions": 0,
           "seconds": 0.0, "status": "ok", "error": ""}
    start = time.perf_counter()
    try:
        sha = sha256_of(path)
        row["sha256"] = sha
        record = Path(out_dir) / "records" / sha[:2] / f"{sha}.npz"
        row["record"] = str(record)
        if record.exists():
            row["status"] = "cached"
            return row
        with tempfile.TemporaryDirectory(prefix="dh_", dir=tmp_root) as tmp:
            decoded = Path(tmp) / "out"
            cmd = [apktool, "d", *apktool_args, "-f", "-o", str(decoded), path]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                 text=True, timeout=timeout)
            if res.returncode:
                raise RuntimeError(f"apktool exit {res.returncode}: {res.stderr[-1000:]}")
            builder = RecordBuilder()
            for triple in iter_smali_instructions(decoded, opcodes):
                builder.add(triple)
        if not len(builder):
            raise ValueError("no Dalvik instruction found")
        builder.save(record)
        row["instructions"] = len(builder)
    except subprocess.TimeoutExpired:
        row.update(status="error", error=f"apktool timeout after {timeout}s")
    except Exception as exc:  # one broken APK must not stop the run
        row.update(status="error", error=repr(exc)[:1000])
    row.setdefault("sha256", "")
    row["seconds"] = time.perf_counter() - start
    return row


def run(args):
    if not shutil.which(args.apktool):
        raise SystemExit(f"apktool not found: {args.apktool}")
    sources = []
    if args.benign:
        sources.append((args.benign, 0))
    if args.malware:
        sources.append((args.malware, 1))
    if not sources and args.input:
        sources = [(Path(args.input) / "Benign", 0), (Path(args.input) / "Malware", 1)]
    if not sources:
        raise SystemExit("Give --input, or --benign and/or --malware")

    apks = list_apks(sources, args.year_regex)
    mine = apks[args.shard::args.num_shards]
    log.info("%d APKs found, shard %d/%d handles %d", len(apks), args.shard, args.num_shards, len(mine))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tmp_root = args.tmp or os.environ.get("TMPDIR") or tempfile.gettempdir()
    worker = partial(extract_one, out_dir=out, opcodes=load_opcodes(args.ops), apktool=args.apktool,
                     apktool_args=args.apktool_args.split(), tmp_root=tmp_root, timeout=args.timeout)

    res = Resources()
    manifest = out / f"manifest.shard{args.shard:03d}.csv"
    times, failures = [], 0
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in parallel_map(worker, mine, args.jobs, desc="extract"):
            writer.writerow(row)
            fh.flush()
            if row["status"] == "ok":
                times.append(row["seconds"])
            elif row["status"] == "error":
                failures += 1
                log.warning("%s: %s", row["apk"], row["error"][:200])
    write_json({"shard": args.shard, "num_shards": args.num_shards, "apks": len(mine), "failures": failures,
                "seconds_per_apk": summarize(times), "ops": str(args.ops),
                "ops_sha256": sha256_of(args.ops), "apktool_args": args.apktool_args},
               out / f"extraction_summary.shard{args.shard:03d}.json")
    append_csv(out / "resources.csv", res.snapshot(f"extract shard {args.shard}", host=os.uname().nodename,
                                                   apks=len(mine), failures=failures, jobs=args.jobs))
    log.info("shard done: %d failures, manifest %s", failures, manifest)


def merge(args):
    """Concatenate shard manifests, drop failures and duplicates, check label conflicts."""
    out = Path(args.output)
    rows, seen = [], {}
    shards = sorted(out.glob("manifest.shard*.csv"))
    if not shards:
        raise SystemExit(f"No manifest.shard*.csv in {out}")
    dup = errors = 0
    for shard in shards:
        with shard.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["status"] == "error":
                    errors += 1
                    continue
                sha = row["sha256"]
                if sha in seen:
                    if seen[sha] != row["label"]:
                        raise SystemExit(f"Same APK {sha} labelled both benign and malware")
                    dup += 1
                    continue
                seen[sha] = row["label"]
                if row["status"] == "cached" or not row["instructions"] or row["instructions"] == "0":
                    from .records import Record
                    row["instructions"] = len(Record(row["record"]))
                rows.append(row)
    fields = ["sha256", "label", "year", "apk", "record", "instructions"]
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    by_label = {lab: sum(r["label"] == lab for r in rows) for lab in ("0", "1")}
    write_json({"apks": len(rows), "benign": by_label["0"], "malware": by_label["1"],
                "duplicates_dropped": dup, "failures": errors}, out / "manifest_summary.json")
    log.info("%d APKs (%d benign, %d malware), %d duplicates, %d failures",
             len(rows), by_label["0"], by_label["1"], dup, errors)
