"""Export the demo checkpoints and a results bundle (backup route for results + checkpoints)."""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import torch

from .train import ledger, run_dir, run_spec
from .utils import atomic_json, file_sha256, log


def export(cfg: dict, out: str = "exports") -> None:
    out = Path(out)
    (out / "demo").mkdir(parents=True, exist_ok=True)
    x = cfg["xai"]
    exported = []
    for m in ("ce", "supcon"):
        spec = run_spec(cfg, m, x["train_fraction"], x["seed"])
        ck = run_dir(spec) / "checkpoints" / "best.pt"
        if not ck.exists():
            log(f"skip demo export for {m}: {ck} missing")
            continue
        state = torch.load(ck, map_location="cpu", weights_only=False)
        dst = out / "demo" / f"{m}_f{x['train_fraction']:.2f}_s{x['seed']}.pt"
        torch.save({"model": state["model"], "class_map": state["class_map"], "run_id": spec["run_id"],
                    "best": state["best"], "image_size": cfg["train"]["image_size"]}, dst)
        exported.append({"method": m, "file": dst.name, "run_id": spec["run_id"], "sha256": file_sha256(dst),
                         "best_val": state["best"]})
        log(f"exported demo checkpoint {dst}")
    atomic_json(out / "demo" / "demo_models.json", exported)

    # full backup: results, manifests, configs, run metadata + all best checkpoints
    led = ledger(cfg)
    bundle = out / "eggplant_results_and_checkpoints.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for p in ("results", "data/manifests", "configs", "README.md", "PROGRESS.md"):
            if Path(p).exists():
                tar.add(p)
        for _, r in led[led["state"] == "completed"].iterrows():
            rd = run_dir(run_spec(cfg, r["method"], r["fraction"], r["seed"]))
            for f in ("spec.json", "status.json", "history.csv", "train.log", "checkpoints/best.pt"):
                if (rd / f).exists():
                    tar.add(rd / f)
    log(f"wrote backup bundle {bundle} ({bundle.stat().st_size / 2**20:.1f} MiB)")
