"""Save a grid of augmented training views (support images only) for visual inspection."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggplant.data import IMAGENET_MEAN, IMAGENET_STD, ImageTable  # noqa: E402
from eggplant.utils import load_config  # noqa: E402

cfg = load_config("configs/experiment.yaml")
sup = pd.read_csv("data/manifests/supports/support_f0.10_s42.csv").groupby("class_id").head(1)
ds = ImageTable(sup, cfg["data"]["cache_dir"], cfg, "two_view", seed=42)
mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
fig, ax = plt.subplots(len(ds), 5, figsize=(10, 2 * len(ds)))
for i in range(len(ds)):
    ax[i, 0].imshow(plt.imread(Path(cfg["data"]["cache_dir"]) / ds.df.at[i, "cache_path"]))
    ax[i, 0].set_title(ds.df.at[i, "label_original"][:22], fontsize=7)
    for e in range(2):
        ds.set_epoch(e)
        views, _, _ = ds[i]
        for v in range(2):
            ax[i, 1 + 2 * e + v].imshow((views[v] * std + mean).clamp(0, 1).permute(1, 2, 0))
            ax[i, 1 + 2 * e + v].set_title(f"ep{e} view{v}", fontsize=7)
for a in ax.ravel():
    a.axis("off")
Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig("results/figures/augmentation_preview.png", dpi=80)
print("wrote results/figures/augmentation_preview.png")
