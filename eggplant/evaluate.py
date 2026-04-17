"""Single reproducible final-evaluation stage on the locked test split."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import ImageTable, load_class_map
from .metrics import classification_metrics
from .model import build_classifier
from .train import ledger, make_loader, predict, run_dir, run_spec
from .utils import atomic_json, file_sha256, log

RES = Path("results")


def load_best(spec: dict, device) -> tuple[torch.nn.Module, dict]:
    ck_path = run_dir(spec) / "checkpoints" / "best.pt"
    status = json.loads((run_dir(spec) / "status.json").read_text())
    if file_sha256(ck_path) != status["best_checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {ck_path}")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    if ck["run_id"] != spec["run_id"]:
        raise RuntimeError("checkpoint run_id mismatch")
    model = build_classifier(len(ck["class_map"]), pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def plot_confusion(cm: np.ndarray, names: list[str], title: str, path: Path, normalized: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalized else int(v), ha="center", va="center", fontsize=7,
                    color="white" if v > (0.5 if normalized else cm.max() / 2) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate_all(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = Path(cfg["data"]["manifest_dir"])
    class_map = load_class_map(mdir)
    names = [c["display_name"] for c in class_map]
    n = len(class_map)
    split = pd.read_csv(mdir / "split.csv")
    test = split[split["partition"] == "test"].reset_index(drop=True)
    split_hash = file_sha256(mdir / "split.csv")
    ds = ImageTable(test, cfg["data"]["cache_dir"], cfg, "eval")
    dl = make_loader(ds, 64, None, cfg["train"]["num_workers"])
    led = ledger(cfg)
    done = led[led["state"] == "completed"]
    (RES / "predictions").mkdir(parents=True, exist_ok=True)
    (RES / "metrics").mkdir(parents=True, exist_ok=True)
    (RES / "figures").mkdir(parents=True, exist_ok=True)
    log(f"final test evaluation: {len(done)}/{len(led)} completed runs, {len(test)} test images")
    rows = []
    for _, r in tqdm(done.iterrows(), total=len(done), desc="evaluate runs", unit="run"):
        spec = run_spec(cfg, r["method"], r["fraction"], r["seed"])
        if spec["split_sha256"] != split_hash:
            raise RuntimeError("split hash changed since training")
        model, ck = load_best(spec, device)
        if [c["display_name"] for c in ck["class_map"]] != names:
            raise RuntimeError("class map mismatch between checkpoint and manifest")
        probs, ys, idx, loss = predict(model, dl, device, cfg["train"]["amp"], desc=f"  test {spec['run_id']}")
        pred = probs.argmax(1)
        m = classification_metrics(ys, pred, n)
        m.update(run_id=spec["run_id"], method=spec["method"], fraction=spec["fraction"], seed=spec["seed"],
                 test_loss=loss, best_val=ck["best"], split_sha256=split_hash, support_sha256=spec["support_sha256"],
                 checkpoint_sha256=file_sha256(run_dir(spec) / "checkpoints" / "best.pt"))
        atomic_json(RES / "metrics" / f"{spec['run_id']}.json", m)
        pdf = pd.DataFrame({"rel_path": test.loc[idx, "rel_path"].values, "true": ys, "pred": pred,
                            "correct": ys == pred, "run_id": spec["run_id"], "support_sha256": spec["support_sha256"],
                            "split_sha256": split_hash, "checkpoint_sha256": m["checkpoint_sha256"]})
        for c in range(n):
            pdf[f"p_{c}"] = probs[:, c]
        pdf.to_csv(RES / "predictions" / f"{spec['run_id']}.csv", index=False)
        rows.append({k: m[k] for k in ("run_id", "method", "fraction", "seed", "macro_f1", "accuracy",
                                       "balanced_accuracy", "test_loss")} | {"best_val_macro_f1": ck["best"]["val_macro_f1"],
                                                                             "best_ft_epoch": ck["best"]["epoch"]})
        log(f"  {spec['run_id']}: test macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} bal-acc {m['balanced_accuracy']:.4f}")
    if not rows:
        log("no completed runs to evaluate"); return
    per_run = pd.DataFrame(rows).sort_values(["fraction", "seed", "method"])
    per_run.to_csv(RES / "test_metrics_per_run.csv", index=False)
    complete = len(done) == len(led)

    agg = per_run.groupby(["method", "fraction"]).agg(
        n_runs=("macro_f1", "size"), macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
        bal_acc_mean=("balanced_accuracy", "mean"), bal_acc_sd=("balanced_accuracy", "std")).reset_index()
    agg["status"] = "final" if complete else "PROVISIONAL (incomplete matrix)"
    agg.to_csv(RES / "summary_by_method_fraction.csv", index=False)
    wide = per_run.pivot_table(index=["fraction", "seed"], columns="method", values="macro_f1").dropna()
    paired = wide.assign(diff_supcon_minus_ce=wide.get("supcon") - wide.get("ce")).reset_index() \
        if {"ce", "supcon"} <= set(wide.columns) else pd.DataFrame()
    paired.to_csv(RES / "paired_differences.csv", index=False)
    print("\n" + agg.round(4).to_string(index=False))
    if not paired.empty:
        print("\npaired SupCon - CE test macro-F1:\n" + paired.round(4).to_string(index=False))
        print(paired.groupby("fraction")["diff_supcon_minus_ce"].agg(["mean", "std", "min", "max", "size"]).round(4).to_string())

    # fraction vs macro-F1 plot (each seed shown; mean +- sample SD)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for mth, color, off in (("ce", "#4C72B0", -0.004), ("supcon", "#DD8452", 0.004)):
        d = per_run[per_run["method"] == mth]
        if d.empty:
            continue
        ax.scatter(d["fraction"] + off, d["macro_f1"], color=color, alpha=0.45, s=22)
        a = agg[agg["method"] == mth]
        ax.errorbar(a["fraction"] + off, a["macro_f1_mean"], yerr=a["macro_f1_sd"].fillna(0), color=color,
                    marker="o", capsize=4, label=f"{'CE' if mth == 'ce' else 'SupCon'} (mean ± sample SD, n={int(a['n_runs'].min())} seeds)")
    ax.set_xticks(sorted(per_run["fraction"].unique()), [f"{f:.0%}" for f in sorted(per_run["fraction"].unique())])
    ax.set_xlabel("fraction of eligible training pool used"); ax.set_ylabel("test macro-F1")
    ax.set_title("Test macro-F1 vs training fraction" + ("" if complete else " (PROVISIONAL)"), fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "figures" / "fraction_vs_macro_f1.png", dpi=150); plt.close(fig)

    # predeclared confusion matrices: seed 42, both fractions, both methods
    for _, r in per_run[per_run["seed"] == 42].iterrows():
        m = json.loads((RES / "metrics" / f"{r['run_id']}.json").read_text())
        tag = f"{r['method']}_f{r['fraction']:.2f}_s42"
        plot_confusion(np.array(m["confusion"]), names, f"{tag} raw counts", RES / "figures" / f"cm_{tag}_raw.png", False)
        plot_confusion(np.array(m["confusion_row_normalized"]), names, f"{tag} row-normalized",
                       RES / "figures" / f"cm_{tag}_norm.png", True)

    # error analysis inputs: per-class F1 by method/fraction, most frequent confusions
    pc = []
    for _, r in per_run.iterrows():
        m = json.loads((RES / "metrics" / f"{r['run_id']}.json").read_text())
        for c in m["per_class"]:
            pc.append({"method": r["method"], "fraction": r["fraction"], "seed": r["seed"], "class": names[c["class_id"]],
                       **{k: c[k] for k in ("precision", "recall", "f1", "support")}})
        cm = np.array(m["confusion"])
    pcdf = pd.DataFrame(pc)
    pcdf.groupby(["class", "method", "fraction"])[["precision", "recall", "f1"]].mean().round(4) \
        .unstack(["method", "fraction"]).to_csv(RES / "per_class_f1_mean.csv")
    conf = []
    for _, r in per_run.iterrows():
        cm = np.array(json.loads((RES / "metrics" / f"{r['run_id']}.json").read_text())["confusion"])
        for i in range(n):
            for j in range(n):
                if i != j and cm[i, j]:
                    conf.append({"method": r["method"], "fraction": r["fraction"], "true": names[i], "pred": names[j], "count": int(cm[i, j])})
    cdf = pd.DataFrame(conf)
    if not cdf.empty:
        top = cdf.groupby(["true", "pred"])["count"].sum().sort_values(ascending=False).head(15)
        top.to_csv(RES / "top_confusions_all_runs.csv")
        print("\nmost frequent confusions (summed over all evaluated runs):\n" + top.to_string())
    atomic_json(RES / "evaluation_meta.json", {"complete_matrix": complete, "n_completed": int(len(done)),
                                                "n_total": int(len(led)), "test_images": int(len(test)),
                                                "split_sha256": split_hash, "device": str(device)})
    log(f"wrote results/ (status: {'final' if complete else 'PROVISIONAL'})")
