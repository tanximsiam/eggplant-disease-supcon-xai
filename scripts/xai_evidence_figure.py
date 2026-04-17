"""Compact CE-vs-SupCon Grad-CAM/occlusion figure for the README.

Rows are predetermined: the first image of each class in the saved XAI subset (no selection by outcome).
Target = ground-truth class for both models (same rule as the quantitative XAI evaluation).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggplant.data import load_class_map  # noqa: E402
from eggplant.evaluate import load_best  # noqa: E402
from eggplant.train import run_spec  # noqa: E402
from eggplant.utils import load_config  # noqa: E402
from eggplant.xai import gradcam, load_image01, mean_fill, normalize, occlusion  # noqa: E402

cfg = load_config("configs/experiment.yaml")
x = cfg["xai"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names = [c["display_name"] for c in load_class_map(cfg["data"]["manifest_dir"])]
subset = pd.read_csv("results/xai/xai_subset.csv").groupby("class_id").head(1).reset_index(drop=True)
imgs = [load_image01(Path(cfg["data"]["cache_dir"]) / p, cfg["train"]["image_size"]).to(dev) for p in subset["cache_path"]]

cols = ["input", "CE Grad-CAM", "CE occlusion", "SupCon Grad-CAM", "SupCon occlusion"]
fig, ax = plt.subplots(len(subset), 5, figsize=(10, 2.15 * len(subset)))
for ci, m in enumerate(("ce", "supcon")):
    model, _ = load_best(run_spec(cfg, m, x["train_fraction"], x["seed"]), dev)
    model.float()
    for r, img in enumerate(imgs):
        gt = int(subset.at[r, "class_id"])
        with torch.no_grad():
            p = model(normalize(img[None])).float().softmax(1)[0].cpu()
        pred = int(p.argmax())
        cam, _ = gradcam(model, img, gt)
        occ = occlusion(model, img, gt, x["occlusion_patch"], x["occlusion_stride"], mean_fill(img))
        base = img.permute(1, 2, 0).cpu().numpy()
        for k, (mp, label) in enumerate(((cam, "Grad-CAM"), (occ, "occlusion"))):
            a = ax[r, 1 + 2 * ci + k]
            a.imshow(base)
            a.imshow(mp, cmap="jet", alpha=0.45)
            verdict = "OK" if pred == gt else f"WRONG→{names[pred][:14]}"
            a.set_title(f"{m.upper()} {label}\np(GT)={p[gt]:.2f} {verdict}", fontsize=7)
        if ci == 0:
            ax[r, 0].imshow(base)
            ax[r, 0].set_title(f"GT: {names[gt]}", fontsize=8)
    del model
for a in ax.ravel():
    a.axis("off")
fig.suptitle("Grad-CAM and occlusion (target = ground-truth class), 20% fraction, seed 42 — first XAI-subset image per class.\n"
             "Scores uncalibrated; heatmaps show model evidence, not validated lesion locations.", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.975))
out = Path("results/xai/evidence_ce_vs_supcon.png")
fig.savefig(out, dpi=85)
print(f"wrote {out}")
