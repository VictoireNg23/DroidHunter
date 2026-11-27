"""Side-by-side table: reproduced results vs the values printed in the paper."""
from pathlib import Path

from .utils import read_json

# Accuracy (%) from the ASIA CCS '26 paper.
PAPER = {
    ("vit", "droidtrain"): 99.97, ("cnn-svm", "droidtrain"): 99.08,
    ("vit", "droidobpack1"): 98.98, ("cnn-svm", "droidobpack1"): 98.20,
    ("vit", "droidobpack2"): 98.76, ("cnn-svm", "droidobpack2"): 98.51,
}
PAPER_AUT = {"vit": 0.89, "cnn-svm": 0.87}


def run(args):
    lines = ["| Model | Set | Accuracy (%) | Paper (%) | Gap (pts) | F1 | AUT | Paper AUT |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in args.results:
        label, path = item.split("=", 1)
        model, set_name = label.split(":")
        data = read_json(Path(path))
        m = data.get("metrics") or data.get("test")
        acc = 100 * m["accuracy"]
        ref = PAPER.get((model, set_name.lower()))
        gap = f"{acc - ref:+.2f}" if ref is not None else ""
        aut = data.get("per_year", {}).get("aut_accuracy")
        lines.append(f"| {model} | {set_name} | {acc:.2f} | {ref if ref is not None else ''} | {gap} | "
                     f"{m['f1']:.4f} | {'' if aut is None else f'{aut:.3f}'} | "
                     f"{PAPER_AUT.get(model, '') if aut is not None else ''} |")
    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
