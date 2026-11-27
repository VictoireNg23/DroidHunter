"""Phase 3: train DroidHunter-ViT or DroidHunter-CNN+SVM.

Single GPU:  python -m droidhunter train ...
All GPUs of a node:  torchrun --nproc_per_node=gpu -m droidhunter train ...
The checkpoint is chosen on the validation split; the test split is only read once,
at the end, by the best checkpoint.
"""
import os
import time
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import ImageArrays, n_channels, parse_mask
from .metrics import binary_metrics
from .models import build_model, logits
from .utils import Resources, append_csv, log, read_json, seed_everything, write_json


def _ddp():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return dist.is_initialized()


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def amp_context(mode, device):
    if mode == "off" or device.type != "cuda":
        return nullcontext, None
    dtype = torch.bfloat16 if (mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported())) else torch.float16
    scaler = torch.cuda.amp.GradScaler() if dtype == torch.float16 else None
    return (lambda: torch.autocast("cuda", dtype=dtype)), scaler


def loader(ds, batch, workers, shuffle=False, sampler=None):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      prefetch_factor=4 if workers > 0 else None)


@torch.no_grad()
def predict(model, dl, device, autocast=nullcontext):
    model.eval()
    ys, preds, scores = [], [], []
    for x, y in dl:
        with autocast():
            out = logits(model, x.to(device, non_blocking=True)).float()
        prob = out.softmax(1)[:, 1]
        ys.append(y.numpy())
        preds.append(out.argmax(1).cpu().numpy())
        scores.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(scores)


@torch.no_grad()
def cnn_features(model, dl, device, autocast=nullcontext):
    model.eval()
    feats, ys = [], []
    for x, y in dl:
        with autocast():
            feats.append(model.features(x.to(device, non_blocking=True)).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def make_svm(kind, scale):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    if kind == "svc":  # exact setting of the paper, slow beyond ~50k samples
        svm = SVC(kernel="linear", C=0.2)
    else:  # same linear model, liblinear solver: minutes instead of hours
        svm = LinearSVC(C=0.2, loss="hinge", dual=True, max_iter=20000)
    return make_pipeline(StandardScaler(), svm) if scale else svm


def run(args):
    ddp = _ddp()
    rank = _rank()
    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    torch.backends.cudnn.benchmark = True
    out = Path(args.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    size = args.size or (224 if args.model == "vit" else 400)
    norm = args.norm or ("imagenet" if args.model == "vit" else "unit")
    mask = parse_mask(args.mask, args.input)
    channels = n_channels(args.input)
    data = Path(args.data)
    sets = {k: ImageArrays(data / k, args.input, size, norm, mask) for k in ("train", "val", "test")}

    model = build_model(args.model, channels, size, args.vit_source, args.cnn_arch).to(device)
    model = model.to(memory_format=torch.channels_last)
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    core = model.module if ddp else model

    sampler = DistributedSampler(sets["train"], shuffle=True, seed=args.seed) if ddp else None
    val_sampler = DistributedSampler(sets["val"], shuffle=False) if ddp else None
    train_dl = loader(sets["train"], args.batch_size, args.workers, shuffle=True, sampler=sampler)
    val_dl = loader(sets["val"], args.batch_size, args.workers, sampler=val_sampler)

    opt_cls = torch.optim.AdamW if args.model == "vit" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = None
    if args.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr * 10, total_steps=args.epochs * len(train_dl))
    elif args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    autocast, scaler = amp_context(args.amp, device)
    loss_fn = nn.CrossEntropyLoss()

    world = dist.get_world_size() if ddp else 1
    res = Resources()
    best, history, bad_epochs, first_epoch = -1.0, [], 0, 1
    resume_path = out / "last.pt"
    if resume_path.exists():  # job killed (besteffort, walltime): continue after the last finished epoch
        state = torch.load(resume_path, map_location=device, weights_only=False)
        core.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if sched and state["sched"]:
            sched.load_state_dict(state["sched"])
        if scaler and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        best, history, bad_epochs = state["best"], state["history"], state["bad_epochs"]
        first_epoch = state["epoch"] + 1
        log.info("resuming after epoch %d (best val_acc %.4f)", state["epoch"], best)

    for epoch in range(first_epoch, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        start, total, seen = time.perf_counter(), 0.0, 0
        n_batches = len(train_dl)
        for step, (x, y) in enumerate(train_dl, 1):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                loss = loss_fn(logits(model, x), y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            if sched:
                sched.step()  # per batch, so the schedule spans exactly the run
            total += loss.item() * len(y)
            seen += len(y)
            if rank == 0 and (step % max(1, n_batches // 10) == 0 or step == n_batches):
                el = time.perf_counter() - start
                log.info("epoch %d/%d  batch %d/%d (%.0f%%)  loss %.4f  %.0f img/s  epoch left ~%.0f min", epoch,
                         args.epochs, step, n_batches, 100 * step / n_batches, total / seen,
                         seen * world / el, el / step * (n_batches - step) / 60)

        train_seconds = time.perf_counter() - start
        y_val, p_val, _ = predict(model, val_dl, device, autocast)
        stats = torch.tensor([((p_val == 1) & (y_val == 1)).sum(), ((p_val == 1) & (y_val == 0)).sum(),
                              ((p_val == 0) & (y_val == 0)).sum(), ((p_val == 0) & (y_val == 1)).sum(),
                              total, seen], dtype=torch.float64, device=device)
        if ddp:
            dist.all_reduce(stats)
        tp, fp, tn, fn, loss_sum, n_seen = stats.tolist()
        val_acc = (tp + tn) / max(tp + fp + tn + fn, 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        row = {"epoch": epoch, "train_loss": round(loss_sum / n_seen, 6), "val_accuracy": round(val_acc, 6),
               "val_precision": round(prec, 6), "val_recall": round(rec, 6),
               "val_f1": round(2 * prec * rec / (prec + rec), 6) if prec + rec else 0.0,
               "val_tp": int(tp), "val_fp": int(fp), "val_tn": int(tn), "val_fn": int(fn),
               "lr": opt.param_groups[0]["lr"], "train_seconds": round(train_seconds, 1),
               "epoch_seconds": round(time.perf_counter() - start, 1),
               "train_img_per_s": round(n_seen / train_seconds, 1), "best_so_far": val_acc > best}
        history.append(row)
        if rank == 0:
            log.info("epoch %d/%d done  loss %.4f  val acc %.4f  P %.4f  R %.4f  F1 %.4f  (%.0f s)", epoch,
                     args.epochs, row["train_loss"], val_acc, prec, rec, row["val_f1"], row["epoch_seconds"])
            append_csv(out / "metrics_train.csv", row)
            append_csv(out / "resources.csv", res.snapshot(f"epoch {epoch}", gpus=world,
                                                           epoch_seconds=row["epoch_seconds"]))
            if val_acc > best:
                torch.save(core.state_dict(), out / "model.pt")
        if val_acc > best:
            best, bad_epochs = val_acc, 0
        else:
            bad_epochs += 1
        if rank == 0:
            torch.save({"model": core.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict() if sched else None,
                        "scaler": scaler.state_dict() if scaler else None, "epoch": epoch, "best": best,
                        "history": history, "bad_epochs": bad_epochs}, out / "last.pt.tmp")
            (out / "last.pt.tmp").replace(resume_path)
        if args.patience and bad_epochs >= args.patience:
            log.info("early stop after %d epochs without improvement", bad_epochs)
            break

    if ddp:
        dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return

    core.load_state_dict(torch.load(out / "model.pt", map_location=device, weights_only=True))
    images_meta = read_json(data / "train" / "images.json")
    config = {"model": args.model, "cnn_arch": args.cnn_arch, "input": args.input, "size": size, "norm": norm,
              "mask": args.mask, "channels": channels, "best_val_accuracy": best, "seed": args.seed,
              "data": str(data.resolve()), "vocab": images_meta["vocab"], "scale": images_meta["scale"],
              "args": {k: v for k, v in vars(args).items() if k != "func"}}
    if args.model == "vit":
        config["vit_config"] = core.config.to_dict()
    write_json(config, out / "config.json")
    write_json(history, out / "history.json")

    test_dl = loader(sets["test"], args.batch_size, args.workers)
    results = {}
    y, p, s = predict(core, test_dl, device, autocast)
    results["test"] = binary_metrics(y, p, s)
    if args.model == "cnn-svm":
        results["test_cnn_head"] = results.pop("test")
        t0 = time.perf_counter()
        x_tr, y_tr = cnn_features(core, loader(sets["train"], args.batch_size, args.workers), device, autocast)
        svm = make_svm(args.svm, not args.svm_no_scale)
        svm.fit(x_tr, y_tr)
        joblib.dump(svm, out / "svm.joblib")
        results["svm_seconds"] = time.perf_counter() - t0
        for name in ("val", "test"):
            x, y = cnn_features(core, loader(sets[name], args.batch_size, args.workers), device, autocast)
            results[name] = binary_metrics(y, svm.predict(x), svm.decision_function(x))
    write_json(results, out / "metrics.json")
    for name, m in results.items():
        if isinstance(m, dict):
            append_csv(out / "metrics_test.csv", {"split": name, "model": args.model, "input": args.input,
                                                  "mask": args.mask or "", **m})
    append_csv(out / "resources.csv", res.snapshot("test + svm" if args.model == "cnn-svm" else "test",
                                                   gpus=world, epoch_seconds=""))
    log.info("test: %s", {k: round(v, 4) for k, v in results["test"].items() if isinstance(v, float)})
    if ddp:
        dist.destroy_process_group()
