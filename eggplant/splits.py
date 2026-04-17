"""Group-aware stratified split and nested, paired support-subset selection."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import atomic_json, derive_seed, file_sha256, log

PARTS = ("train_pool", "val", "test")


def group_stratified_split(df: pd.DataFrame, fractions: dict, seed: int) -> pd.Series:
    """Assign whole groups to partitions, per class, greedily toward the target image counts.

    df must contain eligible rows with columns class_id, group_id (single-label groups)."""
    assign = pd.Series(index=df.index, dtype=object)
    for c in sorted(df["class_id"].unique()):
        sub = df[df["class_id"] == c]
        groups = sub.groupby("group_id").size()
        order = np.random.default_rng(derive_seed(seed, "split", c)).permutation(sorted(groups.index))
        # large groups first (stable on the random order) so they don't wreck small partitions late
        order = sorted(order, key=lambda g: -groups[g])
        total = groups.sum()
        target = {p: fractions[p] * total for p in PARTS}
        have = dict.fromkeys(PARTS, 0)
        for g in order:
            deficit = {p: (target[p] - have[p]) / max(target[p], 1e-9) for p in PARTS}
            # every partition must end up non-empty: prioritise empty val/test while groups remain
            empty = [p for p in ("val", "test") if have[p] == 0]
            p = empty[0] if empty else max(PARTS, key=lambda q: deficit[q])
            assign[sub.index[sub["group_id"] == g]] = p
            have[p] += groups[g]
    return assign


def select_support(pool: pd.DataFrame, fraction: float, seed: int, min_per_class: int = 1) -> pd.DataFrame:
    """Deterministic group-aware subset of the training pool.

    Per class, groups are put in a seed-specific random order; groups are taken as a prefix of that
    order while the cumulative image count stays <= floor(fraction * n_class), with at least
    `min_per_class` images (whole first group). Because selection is a prefix of the same order,
    smaller fractions are nested in larger ones for a given seed. Method-independent by construction."""
    chosen = []
    for c in sorted(pool["class_id"].unique()):
        sub = pool[pool["class_id"] == c]
        groups = sub.groupby("group_id").size()
        order = np.random.default_rng(derive_seed(seed, "support", c)).permutation(sorted(groups.index))
        target = max(min_per_class, math.floor(fraction * len(sub)))
        taken, cum = [], 0
        for g in order:
            if cum + groups[g] > target and cum >= min_per_class:
                break
            taken.append(g)
            cum += groups[g]
            if cum >= target:
                break
        chosen.append(sub[sub["group_id"].isin(taken)])
    return pd.concat(chosen).sort_values("rel_path")


def make_splits(cfg: dict) -> None:
    mdir = Path(cfg["data"]["manifest_dir"])
    man = pd.read_csv(mdir / "image_manifest.csv", keep_default_na=False)
    man["eligible"] = man["eligible"].astype(str) == "True"
    el = man[man["eligible"]].copy()
    if el.groupby("group_id")["class_id"].nunique().max() > 1:
        raise RuntimeError("multi-label group among eligible images; audit quarantine failed")
    seed = cfg["split"]["seed"]
    el["partition"] = group_stratified_split(el, cfg["split"]["fractions"], seed)
    ncls = man["class_id"].nunique()
    for p in PARTS:
        present = el.loc[el["partition"] == p, "class_id"].nunique()
        if present != ncls:
            raise RuntimeError(f"partition {p} has {present}/{ncls} classes")
    if el.groupby("group_id")["partition"].nunique().max() > 1:
        raise RuntimeError("group leakage across partitions")
    cols = ["rel_path", "cache_path", "class_id", "label_original", "group_id", "group_type", "sha256"]
    split_path = mdir / "split.csv"
    el.sort_values("rel_path")[cols + ["partition"]].to_csv(split_path, index=False)
    split_hash = file_sha256(split_path)

    counts = el.pivot_table(index="label_original", columns="partition", values="rel_path", aggfunc="size", fill_value=0)
    gcounts = el.pivot_table(index="label_original", columns="partition", values="group_id", aggfunc="nunique", fill_value=0)
    log(f"split locked (seed {seed}) sha256={split_hash[:16]}")
    print(counts[list(PARTS)].assign(total=counts.sum(1)).to_string())

    pool = el[el["partition"] == "train_pool"]
    supports = {}
    sdir = mdir / "supports"
    sdir.mkdir(exist_ok=True)
    for s in cfg["experiment"]["seeds"]:
        prev = None
        for f in sorted(cfg["experiment"]["train_fractions"]):
            sup = select_support(pool, f, s, cfg["experiment"]["min_per_class"])
            if prev is not None and not set(prev["rel_path"]) <= set(sup["rel_path"]):
                raise RuntimeError(f"support for seed {s} fraction {f} is not nested")
            prev = sup
            path = sdir / f"support_f{f:.2f}_s{s}.csv"
            sup[cols].to_csv(path, index=False)
            per_class = []
            for c, lab in sorted(pool.groupby("class_id")["label_original"].first().items()):
                n_pool = int((pool["class_id"] == c).sum())
                n_sel = int((sup["class_id"] == c).sum())
                per_class.append({"class_id": int(c), "label": lab, "pool_images": n_pool,
                                  "target_floor": max(cfg["experiment"]["min_per_class"], math.floor(f * n_pool)),
                                  "selected_images": n_sel,
                                  "selected_groups": int(sup.loc[sup["class_id"] == c, "group_id"].nunique()),
                                  "realized_fraction": round(n_sel / n_pool, 4)})
            supports[f"f{f:.2f}_s{s}"] = {"fraction": f, "seed": s, "file": path.name, "sha256": file_sha256(path),
                                          "total_images": int(len(sup)),
                                          "total_groups": int(sup["group_id"].nunique()),
                                          "realized_fraction": round(len(sup) / len(pool), 4),
                                          "per_class": per_class}
            log(f"support f={f:.2f} seed={s}: {len(sup)} images ({len(sup) / len(pool):.3%} of pool)")
    summary = {
        "split_seed": seed, "split_file": split_path.name, "split_sha256": split_hash,
        "rule": "per class, groups shuffled with derive_seed(2026,'split',class), larger groups first; each group "
                "assigned to the partition with the largest relative deficit (val/test forced non-empty first)",
        "support_rule": "per class & seed, groups in derive_seed(seed,'support',class) order; prefix while "
                        "cumulative <= floor(fraction*n_pool_class), min 1 group; nested by construction",
        "image_counts": {p: {k: int(v) for k, v in counts[p].items()} for p in PARTS},
        "group_counts": {p: {k: int(v) for k, v in gcounts[p].items()} for p in PARTS},
        "totals": {p: int((el["partition"] == p).sum()) for p in PARTS},
        "group_totals": {p: int(el.loc[el["partition"] == p, "group_id"].nunique()) for p in PARTS},
        "excluded_before_split": int((~man["eligible"]).sum()),
        "supports": supports,
    }
    atomic_json(mdir / "split_summary.json", summary)
    log(f"wrote {split_path}, {sdir}/, split_summary.json")
