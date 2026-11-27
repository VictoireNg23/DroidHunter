"""Torch datasets over the memory-mapped image arrays written by `images`."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

INPUTS = {"rgb": ["rgb"], "trigram": ["tri"], "hybrid": ["rgb", "tri"]}
NORMS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet statistics
    "half": ([0.5] * 3, [0.5] * 3),                               # ViT-in21k preprocessing
    "unit": ([0.0] * 3, [1.0] * 3),                               # plain /255, used for the CNN
}
CHANNEL_NAMES = {"rgb": ["opcode", "op1", "op2"], "trigram": ["tri_opcode", "tri_op1", "tri_op2"]}
CHANNEL_NAMES["hybrid"] = CHANNEL_NAMES["rgb"] + CHANNEL_NAMES["trigram"]


def n_channels(input_kind):
    return 3 * len(INPUTS[input_kind])


def parse_mask(mask, input_kind):
    """'101' -> keep opcode and op2, zero op1 (the ablation of §5.6)."""
    c = n_channels(input_kind)
    if not mask:
        return None
    if len(mask) != c or set(mask) - {"0", "1"}:
        raise SystemExit(f"--mask must be {c} characters of 0/1 for input '{input_kind}'")
    return [int(x) for x in mask]


class ImageArrays(Dataset):
    def __init__(self, set_dir, input_kind, size, norm, mask=None):
        set_dir = Path(set_dir)
        self.arrays = []
        for feat in INPUTS[input_kind]:
            path = set_dir / f"{feat}_{size}.npy"
            if not path.exists():
                raise SystemExit(f"Missing {path}: run `images` with --features {feat} --sizes {size}")
            self.arrays.append(np.load(path, mmap_mode="r"))
        self.labels = np.load(set_dir / "labels.npy")
        with (set_dir / "index.csv").open(newline="", encoding="utf-8") as fh:
            self.index = list(csv.DictReader(fh))
        mean, std = NORMS[norm]
        c = n_channels(input_kind)
        reps = c // 3
        self.mean = torch.tensor(mean * reps).view(c, 1, 1)
        self.std = torch.tensor(std * reps).view(c, 1, 1)
        self.keep = None if mask is None else torch.tensor(mask, dtype=torch.float32).view(c, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = np.concatenate([a[i] for a in self.arrays], axis=2)
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div_(255)
        if self.keep is not None:
            x = x * self.keep  # a disabled channel is a black (0) channel, as in §5.6
        return (x - self.mean) / self.std, int(self.labels[i])
