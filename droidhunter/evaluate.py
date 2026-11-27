"""Phase 4: evaluate a trained model on any image set (internal test, DroidObPack, DroidDrift).

Reports the metrics of Tables 3-4, per-year metrics and AUT (Table 5) when the set has
years, and per-APK inference latency (§5.5) with batch 1 and CUDA synchronisation.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from .data import ImageArrays, parse_mask
from .metrics import binary_metrics, per_year
from .models import build_model, logits
from .train import amp_context, cnn_features, loader, predict
from .utils import Resources, append_csv, log, read_json, summarize, write_json


def load_trained(model_dir, device):
    model_dir = Path(model_dir)
    cfg = read_json(model_dir / "config.json")
    model = build_model(cfg["model"], cfg["channels"], cfg["size"], cnn_arch=cfg.get("cnn_arch", "droidhunter"),
                        vit_config=cfg.get("vit_config"))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    svm = joblib.load(model_dir / "svm.joblib") if cfg["model"] == "cnn-svm" else None
    return model.to(device).eval(), svm, cfg


def load_torch_model(path, device):
    """A ViT saved whole with torch.save, expecting RGB 224x224 input with ImageNet normalisation."""
    model = torch.load(path, map_location=device, weights_only=False)
    cfg = {"model": "vit", "input": "rgb", "size": 224, "norm": "imagenet", "mask": None, "channels": 3}
    return model.to(device).eval(), None, cfg


@torch.no_grad()
def latency(model, svm, ds, device, warmup, limit):
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    times = []
    for i in range(-warmup, n):
        x = ds[max(i, 0)][0].unsqueeze(0).to(device)
        sync()
        start = time.perf_counter()
        if svm is None:
            logits(model, x).argmax(1).item()
        else:
            svm.predict(model.features(x).float().cpu().numpy())
        sync()
        if i >= 0:
            times.append((time.perf_counter() - start) * 1000)
    return {"unit": "ms", "batch_size": 1, "apks": n, **summarize(times)}


def run(args):
    res = Resources()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.torch_model:
        model, svm, cfg = load_torch_model(args.torch_model, device)
    else:
        model, svm, cfg = load_trained(args.model_dir, device)
    set_dir = Path(args.data)
    ds = ImageArrays(set_dir, cfg["input"], cfg["size"], cfg["norm"], parse_mask(cfg.get("mask"), cfg["input"]))
    if not args.torch_model:
        meta = read_json(set_dir / "images.json")
        if Path(meta["vocab"]).resolve() != Path(cfg["vocab"]).resolve():
            raise SystemExit(f"Images were made with {meta['vocab']}, the model was trained with {cfg['vocab']}")
    autocast, _ = amp_context(args.amp, device)
    dl = loader(ds, args.batch_size, args.workers)
    if svm is None:
        y, pred, score = predict(model, dl, device, autocast)
    else:
        x, y = cnn_features(model, dl, device, autocast)
        pred, score = svm.predict(x), svm.decision_function(x)
    result = {"set": str(set_dir.resolve()), "model": str(args.torch_model or Path(args.model_dir).resolve()),
              "metrics": binary_metrics(y, pred, score if len(set(y.tolist())) == 2 else None)}
    yearly = per_year(y.tolist(), pred.tolist(), score.tolist(), [r["year"] for r in ds.index])
    if yearly:
        result["per_year"] = yearly
    if args.latency >= 0:
        result["latency"] = latency(model, svm, ds, device, args.warmup, args.latency)
    out = Path(args.output)
    write_json(result, out / "metrics.json")
    with (out / "predictions.csv").open("w", encoding="utf-8") as fh:
        fh.write("sha256,label,pred,score\n")
        for r, yy, pp, ss in zip(ds.index, y, pred, score):
            fh.write(f"{r['sha256']},{int(yy)},{int(pp)},{float(ss):.6f}\n")
    set_name = set_dir.name
    append_csv(out / "metrics.csv", {"set": set_name, **result["metrics"]})
    if yearly:
        for year, ym in yearly["years"].items():
            append_csv(out / "per_year.csv", {"set": set_name, "year": year, **ym})
        append_csv(out / "per_year.csv", {"set": set_name, "year": "AUT", "accuracy": yearly["aut_accuracy"],
                                          "f1": yearly["aut_f1"]})
    if "latency" in result:
        append_csv(out / "latency.csv", {"set": set_name, **result["latency"]})
    append_csv(out / "resources.csv", res.snapshot(f"evaluate {set_name}", apks=len(ds)))
    m = result["metrics"]
    log.info("accuracy %.4f  precision %.4f  recall %.4f  f1 %.4f", m["accuracy"], m["precision"], m["recall"], m["f1"])
    if yearly:
        log.info("AUT(accuracy) %.3f", yearly["aut_accuracy"])
