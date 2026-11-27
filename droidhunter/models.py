"""Classifiers of §3.3: ViT-B/16 and the CNN used as feature extractor for the SVM."""
import math

import torch
import torch.nn.functional as F
from torch import nn


def build_vit(source, channels, config=None):
    from transformers import ViTConfig, ViTForImageClassification
    if config is not None:  # reload a trained model without downloading anything
        cfg = ViTConfig.from_dict(config)
        return ViTForImageClassification(cfg)
    model = ViTForImageClassification.from_pretrained(source, num_labels=2, ignore_mismatched_sizes=True)
    if channels != 3:
        # Hybrid input: widen the patch embedding. Extra channels start from the
        # pretrained RGB filters, everything is rescaled so activations keep their range.
        emb = model.vit.embeddings.patch_embeddings
        old = emb.projection
        new = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride)
        with torch.no_grad():
            reps = math.ceil(channels / 3)
            new.weight.copy_(old.weight.repeat(1, reps, 1, 1)[:, :channels] * 3 / channels)
            new.bias.copy_(old.bias)
        emb.projection = new
        emb.num_channels = channels
        model.config.num_channels = channels
    return model


class SamePad(nn.Module):
    """'same' padding for strided convolutions (output size = ceil(input / stride))."""

    def __init__(self, kernel, stride):
        super().__init__()
        self.k, self.s = kernel, stride

    def forward(self, x):
        pads = []
        for size in (x.shape[-1], x.shape[-2]):
            out = math.ceil(size / self.s)
            total = max((out - 1) * self.s + self.k - size, 0)
            pads += [total // 2, total - total // 2]
        return F.pad(x, pads)


def _dw(channels, kernel, stride):
    return nn.Sequential(SamePad(kernel, stride), nn.Conv2d(channels, channels, kernel, stride, groups=channels))


class DroidHunterCNN(nn.Module):
    """DroidHunter CNN feature extractor (§3.3).

    Conv 3x3 (16) -> pool -> depthwise 32x32 stride 3 -> pointwise (64) -> pool
    -> depthwise 64x64 stride 3 -> pointwise (128) -> pool -> flatten -> MLP head.
    The large strided depthwise kernels give a wide receptive field over long
    instruction streams at low cost. The flattened maps (3,200 values at 400x400)
    are the features passed to the SVM.
    """

    def __init__(self, channels=3, size=400):
        super().__init__()
        self.features = nn.Sequential(
            SamePad(3, 1), nn.Conv2d(channels, 16, 3), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            _dw(16, 32, 3), nn.BatchNorm2d(16), nn.Conv2d(16, 64, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            _dw(64, 64, 3), nn.BatchNorm2d(64), nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dim = self.features(torch.zeros(1, channels, size, size)).shape[1]
        self.head = nn.Sequential(nn.Linear(dim, 1024), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 2))

    def forward(self, x):
        return self.head(self.features(x))


class CompactCNN(nn.Module):
    """Lighter variant: 3x3 depthwise-separable blocks and global average pooling."""

    def __init__(self, channels=3, size=400):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.Conv2d(16, 64, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 2))

    def forward(self, x):
        return self.head(self.features(x))


CNNS = {"droidhunter": DroidHunterCNN, "compact": CompactCNN}


def build_model(kind, channels, size, vit_source=None, cnn_arch="droidhunter", vit_config=None):
    if kind == "vit":
        return build_vit(vit_source, channels, vit_config)
    return CNNS[cnn_arch](channels, size)


def logits(model, x):
    out = model(x)
    return out.logits if hasattr(out, "logits") else out
