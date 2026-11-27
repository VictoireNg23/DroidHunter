"""Channel ablation of §5.6: retrain with one channel set to 0, same splits, same settings.

Images are not regenerated: channels are zeroed when loading. Variants run in parallel,
one per GPU listed in --gpus, then each model is evaluated on the --eval sets.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .utils import log, read_json, write_json

# §5.6: each of R (opcode), G (op1), B (op2) disabled in turn, against the full image.
# With --input hybrid only the R/G/B channel is disabled; the 3-gram channels stay.
VARIANTS = {
    "rgb": {"full": "111", "no_R": "011", "no_G": "101", "no_B": "110"},
    "hybrid": {"full": "111111", "no_R": "011111", "no_G": "101111", "no_B": "110111"},
}
VARIANTS["trigram"] = VARIANTS["rgb"]


def _launch(cmd, gpu, log_path):
    root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, env=env, stdout=log_path.open("w"), stderr=subprocess.STDOUT)


def _run_queue(jobs, gpus):
    """jobs: [(name, cmd, log_path)] -> run with at most one job per GPU."""
    pending, running, failed = list(jobs), {}, []
    while pending or running:
        for gpu in gpus:
            if gpu not in running and pending:
                name, cmd, lp = pending.pop(0)
                log.info("GPU %s: %s", gpu, name)
                running[gpu] = (name, _launch(cmd, gpu, lp))
        for gpu, (name, proc) in list(running.items()):
            if proc.poll() is not None:
                if proc.returncode:
                    failed.append(name)
                    log.error("%s failed, see its log", name)
                del running[gpu]
        time.sleep(5)
    return failed


def run(args):
    out = Path(args.output)
    variants = VARIANTS[args.input]
    names = args.variants or list(variants)
    gpus = args.gpus.split(",")
    py = [sys.executable, "-m", "droidhunter"]

    train_jobs = []
    for name in names:
        cmd = py + ["train", "--data", args.data, "--model", args.model, "--input", args.input,
                    "--mask", variants[name], "--output", str(out / name)] + args.train_args
        train_jobs.append((f"train {name}", cmd, out / name / "train.log"))
    failed = _run_queue(train_jobs, gpus)

    eval_jobs = []
    for name in names:
        for set_dir in args.eval or []:
            set_name = Path(set_dir).name
            cmd = py + ["evaluate", "--model-dir", str(out / name), "--data", set_dir, "--latency", "-1",
                        "--output", str(out / name / "eval" / set_name)]
            eval_jobs.append((f"eval {name} {set_name}", cmd, out / name / "eval" / set_name / "eval.log"))
    failed += _run_queue(eval_jobs, gpus)

    table = {}
    for name in names:
        mfile = out / name / "metrics.json"
        if not mfile.exists():
            continue
        row = {"mask": variants[name], "test": read_json(mfile)["test"]}
        for set_dir in args.eval or []:
            f = out / name / "eval" / Path(set_dir).name / "metrics.json"
            if f.exists():
                row[Path(set_dir).name] = read_json(f)["metrics"]
        table[name] = row
    if "full" in table:  # drop in accuracy w.r.t. all channels, in points, as in the paper
        for name, row in table.items():
            row["drop_test_points"] = 100 * (table["full"]["test"]["accuracy"] - row["test"]["accuracy"])
    write_json({"variants": table, "failed": failed}, out / "comparison.json")
    print(json.dumps({k: {"acc": round(v["test"]["accuracy"], 4), "drop": round(v.get("drop_test_points", 0), 2)}
                      for k, v in table.items()}, indent=2))
