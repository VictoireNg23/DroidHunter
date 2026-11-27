"""Shared helpers: logging, JSON, seeding, timing, process pools."""
import csv
import json
import logging
import resource
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np

log = logging.getLogger("droidhunter")


def setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def default_jobs():
    """Cores granted by OAR/cgroups, not the whole machine."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"Not JSON serialisable: {type(x)}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextmanager
def timed(label, sink=None):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    log.info("%s: %.1f s", label, elapsed)
    if sink is not None:
        sink[label] = elapsed


def summarize(values):
    """Distribution summary used for every timing report."""
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)), "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)), "min": float(a.min()), "max": float(a.max()),
    }


def parallel_map(fn, items, jobs, desc="tasks", chunks=None):
    """Run fn over items in a process pool, yielding results as they finish.

    With jobs <= 1 runs inline, which keeps tracebacks readable when debugging.
    """
    items = list(items)
    _STARTS[desc] = time.perf_counter()
    if jobs <= 1:
        for i, item in enumerate(items, 1):
            yield fn(item)
            _progress(desc, i, len(items))
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for i, fut in enumerate(as_completed(futures), 1):
            yield fut.result()
            _progress(desc, i, len(items))


_STARTS = {}


def _progress(desc, done, total):
    """Progress line every 2 %, with throughput and estimated time left."""
    start = _STARTS.get(desc, time.perf_counter())
    step = max(1, total // 50)
    if done == total or done % step == 0:
        elapsed = time.perf_counter() - start
        left = elapsed / done * (total - done) if done else 0
        log.info("%s: %d/%d (%.0f%%)  %.1f/s  elapsed %s  left ~%s", desc, done, total, 100 * done / total,
                 done / elapsed if elapsed else 0, _hms(elapsed), _hms(left))


def _hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def append_csv(path, row):
    """Append one row, writing the header the first time; the file is readable at any moment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row)
    if path.exists():  # keep the existing columns; missing values stay empty
        with path.open(newline="", encoding="utf-8") as fh:
            fields = next(csv.reader(fh), fields)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


class Resources:
    """Wall time, CPU time and peak memory of this process and its finished workers."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.r0 = (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN))

    def snapshot(self, step, **extra):
        s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        wall = time.perf_counter() - self.t0
        cpu = (s.ru_utime - self.r0[0].ru_utime + s.ru_stime - self.r0[0].ru_stime
               + c.ru_utime - self.r0[1].ru_utime + c.ru_stime - self.r0[1].ru_stime)
        row = {"step": step, "wall_seconds": round(wall, 2), "cpu_seconds": round(cpu, 2),
               "cores_used_avg": round(cpu / wall, 2) if wall else 0,
               "peak_ram_main_gb": round(s.ru_maxrss / 1e6, 3), "peak_ram_worker_gb": round(c.ru_maxrss / 1e6, 3)}
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                row["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
                row["gpu"] = torch.cuda.get_device_name()
        except ImportError:
            pass
        row.update(extra)
        return row


def chunked(seq, size):
    seq = list(seq)
    return [seq[i:i + size] for i in range(0, len(seq), size)]
