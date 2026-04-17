"""EfficientNet-B0 encoder + heads, with independent deterministic initialisation streams."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .utils import derive_seed

FEAT_DIM = 1280


class Encoder(nn.Module):
    """EfficientNet-B0 features + global average pool -> [B, 1280]."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # explicit weight enum; download failure raises (never a silent random init)
        net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


def _seeded(fn, seed: int):
    """Build a module under a private RNG so it never perturbs other streams."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return fn()


def linear_head(n_classes: int, seed: int, purpose: str) -> nn.Linear:
    return _seeded(lambda: nn.Linear(FEAT_DIM, n_classes), derive_seed(seed, purpose))


def projection_head(hidden: int, out: int, seed: int) -> nn.Module:
    return _seeded(lambda: nn.Sequential(nn.Linear(FEAT_DIM, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out)),
                   derive_seed(seed, "projector"))


class Classifier(nn.Module):
    def __init__(self, encoder: Encoder, head: nn.Linear):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, x):
        return self.head(self.encoder(x))


def freeze_bn_stats(model: nn.Module) -> None:
    """Put BatchNorm layers in eval mode (running stats frozen); affine params stay trainable."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def build_classifier(n_classes: int, seed: int = 0, pretrained: bool = True) -> Classifier:
    return Classifier(Encoder(pretrained), linear_head(n_classes, seed, "classifier"))
