"""Command line: python -m droidhunter <command> --help"""
import argparse
from pathlib import Path

from .utils import default_jobs, setup_logging

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPS = ROOT / "resources" / "ops.txt"


def _extract(sub):
    p = sub.add_parser("extract", help="APK -> instruction records (apktool), parallel and shardable")
    p.add_argument("--input", help="Folder with Benign/ and Malware/ subfolders")
    p.add_argument("--benign", help="Benign folder (overrides --input)")
    p.add_argument("--malware", help="Malware folder (overrides --input); alone for malware-only sets")
    p.add_argument("--output", required=True)
    p.add_argument("--ops", default=str(DEFAULT_OPS), help="Valid Dalvik mnemonics")
    p.add_argument("--jobs", type=int, default=default_jobs(), help="Parallel apktool processes (default: all cores)")
    p.add_argument("--shard", type=int, default=0, help="Index of this node, 0..num-shards-1")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--tmp", help="Scratch folder for apktool output (default $TMPDIR); use local disk")
    p.add_argument("--timeout", type=int, default=900, help="Seconds per APK")
    p.add_argument("--apktool", default="apktool")
    p.add_argument("--apktool-args", default="-r", help="Extra apktool flags; -r skips resources")
    p.add_argument("--year-regex", help=r"Year taken from the APK path, e.g. '(20[12]\d)'")
    from . import extract
    p.set_defaults(func=extract.run)

    p = sub.add_parser("merge", help="Merge shard manifests into manifest.csv")
    p.add_argument("--output", required=True, help="Same folder as extract --output")
    p.set_defaults(func=extract.merge)


def _split(sub):
    from . import splits
    p = sub.add_parser("split", help="Stratified 80/10/10 partition -> splits.json")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True, help="splits.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--max-instructions", type=int, default=0, help="Drop larger APKs (0 = keep all)")
    p.set_defaults(func=splits.run)


def _vocab(sub):
    from . import vocab
    p = sub.add_parser("vocab", help="Frequency tables on the training split")
    p.add_argument("--manifest", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scale", choices=["linear", "log", "rank"], default="linear",
                   help="Frequency -> intensity; linear is the paper")
    p.add_argument("--path-depth", type=int, default=6, help="Depth of the op2 method tree")
    p.add_argument("--coarse-depth", type=int, default=2, help="op2 depth used inside 3-grams")
    p.add_argument("--trigrams", action="store_true", help="Also count 3-grams (needed for tri/hybrid)")
    p.add_argument("--min-count", type=int, default=2, help="Prune rare method paths and 3-grams")
    p.add_argument("--chunk", type=int, default=64, help="APKs per task")
    p.add_argument("--jobs", type=int, default=default_jobs())
    p.set_defaults(func=vocab.run)


def _images(sub):
    from . import images
    p = sub.add_parser("images", help="Records -> resized image arrays (train/val/test or an external set)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--vocab", required=True, help="Folder written by `vocab` (training split only)")
    p.add_argument("--output", required=True)
    p.add_argument("--splits", help="splits.json: builds train/, val/, test/")
    p.add_argument("--name", default="external", help="Folder name for a set without --splits")
    p.add_argument("--exclude-splits", help="splits.json of DroidTrain: refuse overlapping external APKs")
    p.add_argument("--features", nargs="+", choices=["rgb", "tri"], default=["rgb"])
    p.add_argument("--sizes", nargs="+", type=int, default=[224, 400], help="224 for ViT, 400 for the CNN")
    p.add_argument("--scale", choices=["linear", "log", "rank"], help="Override the vocabulary scale")
    p.add_argument("--no-png", dest="full_png", action="store_false",
                   help="Skip the full-size PNG per APK (written by default to <set>/full/<features>/)")
    p.add_argument("--keep-records", dest="delete_records", action="store_false",
                   help="Keep the extraction records; by default they are deleted once the images are written")
    p.add_argument("--jobs", type=int, default=default_jobs())
    p.set_defaults(func=images.run)


def _train(sub):
    from . import train
    p = sub.add_parser("train", help="Train ViT or CNN+SVM (use torchrun for several GPUs)")
    p.add_argument("--data", required=True, help="Folder with train/ val/ test/ from `images --splits`")
    p.add_argument("--output", required=True)
    p.add_argument("--model", choices=["vit", "cnn-svm"], required=True)
    p.add_argument("--input", choices=["rgb", "trigram", "hybrid"], default="rgb",
                   help="hybrid = RGB + 3-gram channels (6 channels)")
    p.add_argument("--mask", help="Channels to keep, e.g. 011 (ablation)")
    p.add_argument("--size", type=int, help="Default 224 (ViT) / 400 (CNN)")
    p.add_argument("--norm", choices=["imagenet", "half", "unit"], help="Default imagenet (ViT) / unit (CNN)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64, help="Per GPU")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", choices=["none", "onecycle", "cosine"], default="none")
    p.add_argument("--patience", type=int, default=0, help="Early stopping on val accuracy (0 = off)")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    p.add_argument("--vit-source", default="google/vit-base-patch16-224-in21k",
                   help="Hugging Face id or local copy (nodes without internet)")
    p.add_argument("--cnn-arch", choices=["droidhunter", "compact"], default="droidhunter")
    p.add_argument("--svm", choices=["linearsvc", "svc"], default="linearsvc")
    p.add_argument("--svm-no-scale", action="store_true", help="Feed raw CNN features to the SVM")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=train.run)


def _evaluate(sub):
    from . import evaluate
    p = sub.add_parser("evaluate", help="Metrics, per-year AUT and latency on an image set")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model-dir", help="Folder written by `train`")
    g.add_argument("--torch-model", help="ViT saved whole with torch.save (RGB, 224x224, ImageNet norm)")
    p.add_argument("--data", required=True, help="One set folder, e.g. images/droidobpack1")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    p.add_argument("--latency", type=int, default=1000, help="APKs timed at batch 1 (0 = all, -1 = skip)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=evaluate.run)


def _ablation(sub):
    from . import ablation
    p = sub.add_parser("ablation", help="Retrain with channels disabled, one variant per GPU")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", choices=["vit", "cnn-svm"], default="vit")
    p.add_argument("--input", choices=["rgb", "trigram", "hybrid"], default="rgb")
    p.add_argument("--variants", nargs="+", help="Subset of variants (default all)")
    p.add_argument("--gpus", default="0", help="Comma list, e.g. 0,1,2,3")
    p.add_argument("--eval", nargs="*", help="Extra set folders to evaluate each variant on")
    p.add_argument("train_args", nargs=argparse.REMAINDER, help="After --, extra arguments for train")
    p.set_defaults(func=ablation.run)


def _report(sub):
    from . import report
    p = sub.add_parser("report", help="Compare metrics.json files with the paper")
    p.add_argument("results", nargs="+", help="model:set=path/metrics.json, e.g. vit:DroidObPack1=runs/.../metrics.json")
    p.add_argument("--output")
    p.set_defaults(func=report.run)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="droidhunter", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for add in (_extract, _split, _vocab, _images, _train, _evaluate, _ablation, _report):
        add(sub)
    args = parser.parse_args(argv)
    if getattr(args, "train_args", None) and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    setup_logging(args.verbose)
    args.func(args)
