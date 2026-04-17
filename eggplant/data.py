"""Datasets and transforms. Augmentation randomness is a pure function of (seed, epoch, index, view)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from .utils import derive_seed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_class_map(manifest_dir: str | Path) -> list[dict]:
    return json.loads((Path(manifest_dir) / "class_map.json").read_text())


def train_transform(cfg: dict) -> T.Compose:
    t, a = cfg["train"], cfg["train"]["aug"]
    return T.Compose([
        T.RandomResizedCrop(t["image_size"], scale=tuple(a["rrc_scale"]), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(a["rotation_deg"]),
        T.ColorJitter(**a["color_jitter"]),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_geometry(size: int) -> T.Compose:
    """Deterministic geometry shared by evaluation, XAI and the demo (PIL -> PIL)."""
    return T.Compose([T.Resize(size, antialias=True), T.CenterCrop(size)])


def to_tensor_normalized() -> T.Compose:
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def open_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


class ImageTable(Dataset):
    """Rows from a split/support CSV. mode: 'two_view' | 'one_view' (train aug) | 'eval'."""

    def __init__(self, df: pd.DataFrame, cache_dir: str | Path, cfg: dict, mode: str, seed: int = 0):
        if df.empty:
            raise RuntimeError("empty dataset table")
        self.df = df.reset_index(drop=True)
        self.cache = Path(cache_dir)
        self.mode, self.seed, self.epoch = mode, seed, 0
        self.aug = train_transform(cfg)
        self.eval_tf = T.Compose([eval_geometry(cfg["train"]["image_size"]), to_tensor_normalized()])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.df)

    def _aug(self, img, idx: int, view: int) -> torch.Tensor:
        # pure function of (seed, epoch, sample, view): identical across methods, resumable, worker-independent
        torch.manual_seed(derive_seed(self.seed, "aug", self.epoch, self.df.at[idx, "rel_path"], view))
        return self.aug(img)

    def __getitem__(self, idx: int):
        img = open_rgb(self.cache / self.df.at[idx, "cache_path"])
        y = int(self.df.at[idx, "class_id"])
        if self.mode == "two_view":
            return torch.stack([self._aug(img, idx, 0), self._aug(img, idx, 1)]), y, idx
        if self.mode == "one_view":
            return self._aug(img, idx, 0), y, idx
        return self.eval_tf(img), y, idx
