"""Grad-CAM, occlusion sensitivity, deletion-curve faithfulness and a randomised-head sanity check."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from .data import IMAGENET_MEAN, IMAGENET_STD, eval_geometry, load_class_map, open_rgb
from .model import linear_head
from .train import read_status, run_spec
from .utils import atomic_json, derive_seed, log

OUT = Path("results/xai")
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def normalize(x01: torch.Tensor) -> torch.Tensor:
    """x01: [...,3,H,W] in [0,1] image space -> ImageNet-normalised model input."""
    return (x01 - _MEAN.to(x01.device)) / _STD.to(x01.device)


def load_image01(path, size: int = 224) -> torch.Tensor:
    img = eval_geometry(size)(open_rgb(path))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


# ----------------------------------------------------------------------------- explanations
def gradcam(model, x01: torch.Tensor, target: int) -> tuple[np.ndarray, bool]:
    """Grad-CAM on the encoder's last conv feature map (1280x7x7). Returns ([H,W] in [0,1], is_constant)."""
    model.eval()
    acts = {}

    def fwd_hook(_, __, out):
        acts["a"] = out
        out.register_hook(lambda g: acts.__setitem__("g", g))

    h = model.encoder.features.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            x = normalize(x01.unsqueeze(0)).requires_grad_(False)
            logits = model(x)
            model.zero_grad(set_to_none=True)
            logits[0, target].backward()
    finally:
        h.remove()
    a, g = acts["a"][0].detach(), acts["g"][0].detach()
    weights = g.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * a).sum(0))
    cam = F.interpolate(cam[None, None], size=x01.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    lo, hi = cam.min(), cam.max()
    if (hi - lo) <= 1e-12:
        return torch.zeros_like(cam).cpu().numpy(), True
    return ((cam - lo) / (hi - lo)).cpu().numpy(), False


@torch.no_grad()
def target_probs(model, x01_batch: torch.Tensor, target: int, chunk: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(x01_batch), chunk):
        out.append(model(normalize(x01_batch[i:i + chunk])).float().softmax(1)[:, target].cpu())
    return torch.cat(out).numpy()


def mean_fill(x01: torch.Tensor) -> torch.Tensor:
    return x01.mean(dim=(1, 2), keepdim=True).expand_as(x01).clone()


def blur_fill(x01: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(2 * round(3 * sigma) + 1)
    return gaussian_blur(x01.unsqueeze(0), [k, k], [sigma, sigma])[0]


@torch.no_grad()
def occlusion(model, x01: torch.Tensor, target: int, patch: int, stride: int, fill: torch.Tensor) -> np.ndarray:
    """Per-pixel mean drop in target probability over all patches covering it (masks applied in image space)."""
    _, H, W = x01.shape
    base = target_probs(model, x01[None], target)[0]
    coords = [(i, j) for i in range(0, H - patch + 1, stride) for j in range(0, W - patch + 1, stride)]
    batch = x01.unsqueeze(0).repeat(len(coords), 1, 1, 1)
    for k, (i, j) in enumerate(coords):
        batch[k, :, i:i + patch, j:j + patch] = fill[:, i:i + patch, j:j + patch]
    p = target_probs(model, batch, target)
    heat, cnt = np.zeros((H, W)), np.zeros((H, W))
    for k, (i, j) in enumerate(coords):
        heat[i:i + patch, j:j + patch] += base - p[k]
        cnt[i:i + patch, j:j + patch] += 1
    return heat / np.maximum(cnt, 1)


@torch.no_grad()
def deletion_curve(model, x01: torch.Tensor, order: np.ndarray, fill: torch.Tensor, target: int, steps: int) -> np.ndarray:
    """Target probability after removing the first k% of pixels in `order` (k = 0..100% in `steps` steps)."""
    _, H, W = x01.shape
    n = H * W
    fr = np.linspace(0, 1, steps + 1)
    batch = x01.unsqueeze(0).repeat(len(fr), 1, 1, 1)
    flat_fill = fill.reshape(3, -1)
    for k, f in enumerate(fr):
        idx = torch.from_numpy(order[: int(round(f * n))]).to(x01.device)
        b = batch[k].reshape(3, -1)
        b[:, idx] = flat_fill[:, idx]
    return target_probs(model, batch, target)


def auc(curve: np.ndarray) -> float:
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(curve, np.linspace(0, 1, len(curve))))


def saliency_order(sal: np.ndarray, tie_seed: int = 0) -> np.ndarray:
    """Pixels sorted by descending saliency; ties broken by a fixed random key."""
    tie = np.random.default_rng(tie_seed).random(sal.size)
    return np.lexsort((tie, -sal.ravel()))


def _down(m: np.ndarray, s: int = 14) -> np.ndarray:
    return F.adaptive_avg_pool2d(torch.from_numpy(np.ascontiguousarray(m)).float()[None, None], s)[0, 0].numpy().ravel()


def top_overlap(a: np.ndarray, b: np.ndarray, q: float = 0.2) -> float:
    ta, tb = a >= np.quantile(a, 1 - q), b >= np.quantile(b, 1 - q)
    return float((ta & tb).sum() / max((ta | tb).sum(), 1))


# ----------------------------------------------------------------------------- subset + driver
def select_xai_subset(test: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Deterministic, model-independent: per class, one image per distinct group first (random order)."""
    rows = []
    for c in sorted(test["class_id"].unique()):
        sub = test[test["class_id"] == c].sort_values("rel_path")
        rng = np.random.default_rng(derive_seed(seed, "xai_subset", c))
        sub = sub.iloc[rng.permutation(len(sub))]
        firsts = sub.drop_duplicates("group_id")
        rest = sub[~sub.index.isin(firsts.index)]
        rows.append(pd.concat([firsts, rest]).head(per_class))
    return pd.concat(rows).reset_index(drop=True)


def _overlay(ax, img01, m, title):
    ax.imshow(img01.permute(1, 2, 0).cpu().numpy())
    ax.imshow(m, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=7); ax.axis("off")


def run_xai(cfg: dict) -> None:
    from .evaluate import load_best
    x = cfg["xai"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = Path(cfg["data"]["manifest_dir"])
    names = [c["display_name"] for c in load_class_map(mdir)]
    OUT.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(mdir / "split.csv")
    test = split[split["partition"] == "test"]
    sub_path = OUT / "xai_subset.csv"
    subset = select_xai_subset(test, x["per_class"], x["seed"])
    if sub_path.exists() and pd.read_csv(sub_path)["rel_path"].tolist() != subset["rel_path"].tolist():
        raise RuntimeError("saved XAI subset differs from the deterministic selection")
    subset.to_csv(sub_path, index=False)
    log(f"XAI subset: {len(subset)} test images ({subset.groupby('class_id').size().to_dict()} per class id), "
        f"{subset['group_id'].nunique()} distinct groups")

    specs = {m: run_spec(cfg, m, x["train_fraction"], x["seed"]) for m in ("ce", "supcon")}
    missing = [m for m, s in specs.items() if read_status(s)["state"] != "completed"]
    if missing:
        atomic_json(OUT / "xai_summary.json", {"status": "PENDING", "reason": f"runs not completed: {missing}"})
        log(f"XAI pending: predeclared runs not completed: {missing}"); return

    cache = Path(cfg["data"]["cache_dir"])
    imgs = [load_image01(cache / p, cfg["train"]["image_size"]).to(device) for p in
            tqdm(subset["cache_path"], desc="load XAI images", leave=False)]
    steps, P, S = x["deletion_steps"], x["occlusion_patch"], x["occlusion_stride"]
    records, curves = [], {}
    gallery = {}
    for mth, spec in specs.items():
        model, _ = load_best(spec, device)
        model.float()
        for k, (img, row) in enumerate(tqdm(list(zip(imgs, subset.itertuples())), desc=f"XAI {mth}", unit="img")):
            gt = int(row.class_id)
            with torch.no_grad():
                p_all = model(normalize(img[None])).float().softmax(1)[0].cpu().numpy()
            pred = int(p_all.argmax())
            cam, const = gradcam(model, img, gt)
            fills = {"mean": mean_fill(img), "blur": blur_fill(img, x["blur_sigma"])}
            occ = occlusion(model, img, gt, P, S, fills["mean"])
            rec = {"method": mth, "rel_path": row.rel_path, "true": gt, "pred": pred, "correct": gt == pred,
                   "p_true": float(p_all[gt]), "cam_constant": const,
                   "spearman_cam_occ": float(spearmanr(_down(cam), _down(occ)).statistic),
                   "top20_overlap_cam_occ": top_overlap(cam, occ)}
            for fname, fill in fills.items():
                c_cam = deletion_curve(model, img, saliency_order(cam), fill, gt, steps)
                c_occ = deletion_curve(model, img, saliency_order(occ), fill, gt, steps)
                c_rnd = np.stack([deletion_curve(model, img, np.random.default_rng(rs).permutation(img[0].numel()),
                                                 fill, gt, steps) for rs in x["random_control_seeds"]])
                curves[(mth, fname, "gradcam")] = curves.get((mth, fname, "gradcam"), []) + [c_cam]
                curves[(mth, fname, "occlusion")] = curves.get((mth, fname, "occlusion"), []) + [c_occ]
                curves[(mth, fname, "random")] = curves.get((mth, fname, "random"), []) + [c_rnd.mean(0)]
                rec[f"auc_gradcam_{fname}"] = auc(c_cam)
                rec[f"auc_occlusion_{fname}"] = auc(c_occ)
                rec[f"auc_random_{fname}"] = float(np.mean([auc(c) for c in c_rnd]))
            records.append(rec)
            gallery[(mth, k)] = {"cam": cam, "occ": occ, "pred": pred, "p": p_all,
                                 "cam_pred": gradcam(model, img, pred)[0] if pred != gt else None}

        # sanity: randomised classifier head (fixed seeds) -> Grad-CAM should change
        sanity = []
        rand_head = linear_head(len(names), 12345, "sanity_random_head").to(device)
        orig_head = model.head
        n_s = min(x["sanity_examples"], len(imgs))
        for k in range(n_s):
            gt = int(subset.at[k, "class_id"])
            cam0, _ = gradcam(model, imgs[k], gt)
            model.head = rand_head
            cam1, const1 = gradcam(model, imgs[k], gt)
            model.head = orig_head
            sanity.append({"method": mth, "rel_path": subset.at[k, "rel_path"],
                           "spearman_trained_vs_random_head": float(spearmanr(_down(cam0), _down(cam1)).statistic),
                           "random_head_constant": const1})
            gallery[(mth, "sanity", k)] = (cam0, cam1)
        pd.DataFrame(sanity).to_csv(OUT / f"sanity_random_head_{mth}.csv", index=False)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv(OUT / "per_image_xai.csv", index=False)
    np.savez(OUT / "deletion_curves.npz", **{f"{m}__{f}__{s}": np.stack(v) for (m, f, s), v in curves.items()})

    summary = {"status": "completed", "runs": {m: s["run_id"] for m, s in specs.items()},
               "subset_size": len(subset), "auc_direction": "deletion AUC of target-class probability; LOWER = "
               "probability falls faster when removing top-ranked pixels (more faithful ranking)",
               "per_method": {}}
    for mth in ("ce", "supcon"):
        d = df[df["method"] == mth]
        s = {"n": len(d), "accuracy_on_subset": float(d["correct"].mean()), "cam_constant_maps": int(d["cam_constant"].sum()),
             "spearman_cam_occ_median": float(d["spearman_cam_occ"].median()),
             "top20_overlap_cam_occ_mean": float(d["top20_overlap_cam_occ"].mean())}
        for f in ("mean", "blur"):
            s[f"auc_gradcam_{f}_mean"] = float(d[f"auc_gradcam_{f}"].mean())
            s[f"auc_occlusion_{f}_mean"] = float(d[f"auc_occlusion_{f}"].mean())
            s[f"auc_random_{f}_mean"] = float(d[f"auc_random_{f}"].mean())
            s[f"frac_images_gradcam_beats_random_{f}"] = float((d[f"auc_gradcam_{f}"] < d[f"auc_random_{f}"]).mean())
            s[f"mean_gap_random_minus_gradcam_{f}"] = float((d[f"auc_random_{f}"] - d[f"auc_gradcam_{f}"]).mean())
        san = pd.read_csv(OUT / f"sanity_random_head_{mth}.csv")
        s["sanity_spearman_trained_vs_random_head_median"] = float(san["spearman_trained_vs_random_head"].median())
        summary["per_method"][mth] = s
    w = df.pivot_table(index="rel_path", columns="method", values=["auc_gradcam_mean", "auc_gradcam_blur"])
    for f in ("mean", "blur"):
        diff = w[(f"auc_gradcam_{f}", "supcon")] - w[(f"auc_gradcam_{f}", "ce")]
        summary[f"paired_auc_gradcam_{f}_supcon_minus_ce"] = {"mean": float(diff.mean()), "median": float(diff.median()),
                                                              "frac_supcon_lower": float((diff < 0).mean())}
    atomic_json(OUT / "xai_summary.json", summary)
    print(json.dumps(summary, indent=2))

    # deletion-curve figure
    fr = np.linspace(0, 1, steps + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, f in zip(axes, ("mean", "blur")):
        for mth, col in (("ce", "#4C72B0"), ("supcon", "#DD8452")):
            for src, ls in (("gradcam", "-"), ("occlusion", ":"), ("random", "--")):
                ax.plot(fr, np.stack(curves[(mth, f, src)]).mean(0), ls, color=col, label=f"{mth} {src}")
        ax.set_title(f"deletion curves ({f} fill), GT target, n={len(subset)}", fontsize=9)
        ax.set_xlabel("fraction of pixels removed"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean target probability"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "deletion_curves.png", dpi=150); plt.close(fig)

    # galleries: predetermined first correct + first incorrect per class where available (per model)
    for mth in ("ce", "supcon"):
        d = df[df["method"] == mth].reset_index(drop=True)
        picks = []
        for c in sorted(subset["class_id"].unique()):
            idx = d.index[d["true"] == c]
            ok = [i for i in idx if d.at[i, "correct"]][:1]
            bad = [i for i in idx if not d.at[i, "correct"]][:1]
            picks += ok + bad
        fig, axes = plt.subplots(len(picks), 4, figsize=(9, 2.3 * len(picks)))
        axes = np.atleast_2d(axes)
        for r, k in enumerate(picks):
            g, row = gallery[(mth, k)], d.iloc[k]
            img = imgs[k]
            axes[r, 0].imshow(img.permute(1, 2, 0).cpu().numpy()); axes[r, 0].axis("off")
            axes[r, 0].set_title(f"GT {names[row.true][:20]}\npred {names[row.pred][:20]} "
                                 f"({'OK' if row.correct else 'WRONG'})", fontsize=7)
            _overlay(axes[r, 1], img, g["cam"], f"Grad-CAM [GT target]")
            _overlay(axes[r, 2], img, g["occ"], f"occlusion [GT target]")
            if g["cam_pred"] is not None:
                _overlay(axes[r, 3], img, g["cam_pred"], f"Grad-CAM [predicted: {names[g['pred']][:16]}]")
            else:
                axes[r, 3].axis("off"); axes[r, 3].set_title("(correct: pred = GT)", fontsize=7)
        fig.suptitle(f"{mth.upper()} f=0.20 seed=42 — scores uncalibrated; heatmaps are not lesion validation", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / f"gallery_{mth}.png", dpi=110); plt.close(fig)
        # sanity figure
        n_s = min(x["sanity_examples"], len(imgs))
        fig, axes = plt.subplots(2, n_s, figsize=(1.8 * n_s, 4))
        for k in range(n_s):
            cam0, cam1 = gallery[(mth, "sanity", k)]
            _overlay(axes[0, k], imgs[k], cam0, "trained head")
            _overlay(axes[1, k], imgs[k], cam1, "random head")
        fig.tight_layout(); fig.savefig(OUT / f"sanity_random_head_{mth}.png", dpi=110); plt.close(fig)
    log(f"wrote {OUT}/")
