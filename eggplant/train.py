"""Matched two-stage training: representation phase (CE head or SupCon projector) -> shared CE fine-tuning."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .data import ImageTable, load_class_map
from .metrics import classification_metrics
from .model import Classifier, Encoder, freeze_bn_stats, linear_head, projection_head
from .supcon import supcon_loss
from .utils import (atomic_json, atomic_torch_save, derive_seed, environment_info, file_sha256, log, seed_everything,
                    set_log_file, stable_hash)

RUNS_DIR = Path("runs")


# ----------------------------------------------------------------------------- run specification
def run_spec(cfg: dict, method: str, fraction: float, seed: int) -> dict:
    mdir = Path(cfg["data"]["manifest_dir"])
    summary = json.loads((mdir / "split_summary.json").read_text())
    sup = summary["supports"][f"f{fraction:.2f}_s{seed}"]
    spec = {"method": method, "fraction": fraction, "seed": seed, "train": cfg["train"],
            "split_sha256": summary["split_sha256"], "support_sha256": sup["sha256"], "support_file": sup["file"],
            "class_map_sha256": file_sha256(mdir / "class_map.json"), "protocol_version": 1}
    spec["run_id"] = f"{method}_f{fraction:.2f}_s{seed}_{stable_hash(spec, 10)}"
    return spec


def run_dir(spec: dict) -> Path:
    return RUNS_DIR / spec["run_id"]


def read_status(spec: dict) -> dict:
    p = run_dir(spec) / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"state": "pending"}


# ----------------------------------------------------------------------------- data plumbing
class EpochPermutation(Sampler):
    """Source-image order = f(seed, stage, epoch) only; identical across paired methods."""

    def __init__(self, n: int, seed: int, stage: str):
        self.n, self.seed, self.stage, self.epoch = n, seed, stage, 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __iter__(self):
        g = torch.Generator().manual_seed(derive_seed(self.seed, "order", self.stage, self.epoch))
        return iter(torch.randperm(self.n, generator=g).tolist())

    def __len__(self):
        return self.n


def _worker_init(_):
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)


def make_loader(ds, batch_size, sampler=None, num_workers=4):
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=_worker_init)


def load_tables(cfg: dict, spec: dict):
    mdir = Path(cfg["data"]["manifest_dir"])
    split = pd.read_csv(mdir / "split.csv")
    sup = pd.read_csv(mdir / "supports" / spec["support_file"])
    if file_sha256(mdir / "split.csv") != spec["split_sha256"]:
        raise RuntimeError("split.csv hash differs from run spec; refusing to train on a changed split")
    if file_sha256(mdir / "supports" / spec["support_file"]) != spec["support_sha256"]:
        raise RuntimeError("support file hash differs from run spec")
    if not set(sup["rel_path"]) <= set(split.loc[split["partition"] == "train_pool", "rel_path"]):
        raise RuntimeError("support contains non-train-pool images")
    val = split[split["partition"] == "val"]
    return sup, val


# ----------------------------------------------------------------------------- optimisation helpers
def make_optimizer(encoder: nn.Module, head: nn.Module, t: dict, total_steps: int):
    opt = torch.optim.AdamW([{"params": encoder.parameters(), "lr": t["lr_encoder"]},
                             {"params": head.parameters(), "lr": t["lr_head"]}], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    return opt, sched


def _check_finite(loss: torch.Tensor, where: str) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss at {where}: {loss.item()}")


@torch.no_grad()
def predict(model: Classifier, loader, device, amp: bool, desc: str = "eval"):
    model.eval()
    probs, ys, idxs, losses = [], [], [], []
    for x, y, idx in tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True):
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        losses.append(F.cross_entropy(logits, y, reduction="sum").item())
        probs.append(logits.softmax(1).cpu())
        ys.append(y.cpu())
        idxs.append(idx)
    probs = torch.cat(probs).numpy()
    return probs, torch.cat(ys).numpy(), torch.cat(idxs).numpy(), sum(losses) / len(probs)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


# ----------------------------------------------------------------------------- one pipeline
def train_pipeline(cfg: dict, spec: dict, device: torch.device, max_batches: int | None = None,
                   out_dir: Path | None = None, sup_override: pd.DataFrame | None = None,
                   val_override: pd.DataFrame | None = None) -> dict:
    """Runs (or resumes) one full pipeline. Returns final status dict."""
    t = spec["train"]
    rdir = out_dir or run_dir(spec)
    ckdir = rdir / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    set_log_file(rdir / "train.log")
    seed, method = spec["seed"], spec["method"]
    seed_everything(derive_seed(seed, "global"))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if sup_override is not None:
        sup, val = sup_override, val_override
    else:
        sup, val = load_tables(cfg, spec)
    class_map = load_class_map(cfg["data"]["manifest_dir"])
    n_classes = len(class_map)
    if sorted(sup["class_id"].unique()) != list(range(n_classes)):
        raise RuntimeError(f"support is missing classes: {sorted(sup['class_id'].unique())}")
    cache, nw, bs, amp = cfg["data"]["cache_dir"], t["num_workers"], t["batch_size"], t["amp"]
    ds_rep = ImageTable(sup, cache, cfg, "two_view", seed)
    ds_ft = ImageTable(sup, cache, cfg, "one_view", seed)
    ds_val = ImageTable(val, cache, cfg, "eval")
    s_rep, s_ft = EpochPermutation(len(ds_rep), seed, "rep"), EpochPermutation(len(ds_ft), seed, "ft")
    dl_rep, dl_ft = make_loader(ds_rep, bs, s_rep, nw), make_loader(ds_ft, bs, s_ft, nw)
    dl_val = make_loader(ds_val, 64, None, nw)
    steps_rep, steps_ft = len(dl_rep), len(dl_ft)
    if max_batches:
        steps_rep, steps_ft = min(steps_rep, max_batches), min(steps_ft, max_batches)

    # identical pretrained encoder for both methods; heads from independent RNG streams
    encoder = Encoder(pretrained=True).to(device)
    rep_head = (linear_head(n_classes, seed, "temp_ce_head") if method == "ce"
                else projection_head(t["proj_hidden"], t["proj_dim"], seed)).to(device)
    clf_head = linear_head(n_classes, seed, "classifier").to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    state = {"stage": "rep", "epoch": 0, "history": [], "best": None, "optimizer_steps": 0, "image_views": 0,
             "elapsed_s": 0.0}
    opt, sched = make_optimizer(encoder, rep_head, t, t["rep_epochs"] * steps_rep)
    last = ckdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck["run_id"] != spec["run_id"] or ck["spec"] != spec:
            raise RuntimeError(f"checkpoint metadata mismatch in {last}; refusing to resume")
        state = ck["state"]
        encoder.load_state_dict(ck["encoder"])
        if state["stage"] == "rep":
            rep_head.load_state_dict(ck["head"])
            opt, sched = make_optimizer(encoder, rep_head, t, t["rep_epochs"] * steps_rep)
        else:
            clf_head.load_state_dict(ck["head"])
            opt, sched = make_optimizer(encoder, clf_head, t, t["ft_epochs"] * steps_ft)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"])
        log(f"resumed {spec['run_id']} at stage={state['stage']} next_epoch={state['epoch']}")

    def save_last(head):
        atomic_torch_save({"run_id": spec["run_id"], "spec": spec, "state": state, "encoder": encoder.state_dict(),
                           "head": head.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                           "scaler": scaler.state_dict(), "rng": _rng_state()}, last)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time() - state["elapsed_s"]
    total_epochs = t["rep_epochs"] + t["ft_epochs"]

    # ---------------- representation phase
    if state["stage"] == "rep":
        for epoch in range(state["epoch"], t["rep_epochs"]):
            encoder.train(); rep_head.train()
            if t["freeze_bn_stats"]:
                freeze_bn_stats(encoder)
            ds_rep.set_epoch(epoch); s_rep.set_epoch(epoch)
            run_loss, nb = 0.0, 0
            bar = tqdm(dl_rep, total=steps_rep, leave=False, unit="batch", dynamic_ncols=True,
                       desc=f"  rep[{method}] ep {epoch + 1}/{t['rep_epochs']}")
            for bi, (x, y, _) in enumerate(bar):
                if max_batches and bi >= max_batches:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device)
                b = x.shape[0]
                with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                    feats = encoder(x.flatten(0, 1))                      # [B*2, 1280], image-major
                    out = rep_head(feats)
                if method == "ce":   # mean over both views of every image
                    loss = F.cross_entropy(out.float(), y.repeat_interleave(2))
                else:
                    loss = supcon_loss(out.float().view(b, 2, -1), y, t["temperature"])
                _check_finite(loss, f"rep epoch {epoch} batch {bi}")
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                state["optimizer_steps"] += 1
                state["image_views"] += 2 * b
                run_loss += loss.item(); nb += 1
                bar.set_postfix(loss=f"{run_loss / nb:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
            state["history"].append({"stage": "rep", "epoch": epoch + 1, "train_loss": run_loss / max(nb, 1),
                                     "elapsed_s": time.time() - t_start})
            state["epoch"] = epoch + 1
            state["elapsed_s"] = time.time() - t_start
            done = epoch + 1
            eta = state["elapsed_s"] / done * (total_epochs - done)
            log(f"  [{spec['run_id']}] rep {done}/{t['rep_epochs']} loss={run_loss / max(nb, 1):.4f} "
                f"elapsed={state['elapsed_s'] / 60:.1f}m eta~{eta / 60:.1f}m")
            if state["epoch"] == t["rep_epochs"]:   # switch: discard temp head/projector, fresh classifier + optimiser
                state["stage"], state["epoch"] = "ft", 0
                opt, sched = make_optimizer(encoder, clf_head, t, t["ft_epochs"] * steps_ft)
                scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
                atomic_torch_save({"encoder": encoder.state_dict(), "run_id": spec["run_id"]}, ckdir / "encoder_after_rep.pt")
                save_last(clf_head)
                log(f"  [{spec['run_id']}] representation phase done -> shared CE fine-tuning")
            else:
                save_last(rep_head)
        del rep_head

    # ---------------- shared classification phase
    model = Classifier(encoder, clf_head)
    for epoch in range(state["epoch"], t["ft_epochs"]):
        model.train()
        if t["freeze_bn_stats"]:
            freeze_bn_stats(model)
        ds_ft.set_epoch(epoch); s_ft.set_epoch(epoch)
        run_loss, nb = 0.0, 0
        bar = tqdm(dl_ft, total=steps_ft, leave=False, unit="batch", dynamic_ncols=True,
                   desc=f"  ft[{method}] ep {epoch + 1}/{t['ft_epochs']}")
        for bi, (x, y, _) in enumerate(bar):
            if max_batches and bi >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            _check_finite(loss, f"ft epoch {epoch} batch {bi}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            state["optimizer_steps"] += 1
            state["image_views"] += x.shape[0]
            run_loss += loss.item(); nb += 1
            bar.set_postfix(loss=f"{run_loss / nb:.4f}")
        probs, ys, _, vloss = predict(model, dl_val, device, amp, desc="  validate")
        vm = classification_metrics(ys, probs.argmax(1), n_classes)
        rec = {"stage": "ft", "epoch": epoch + 1, "train_loss": run_loss / max(nb, 1), "val_loss": vloss,
               "val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"], "elapsed_s": time.time() - t_start}
        state["history"].append(rec)
        b = state["best"]
        better = (b is None or vm["macro_f1"] > b["val_macro_f1"]
                  or (vm["macro_f1"] == b["val_macro_f1"] and vloss < b["val_loss"]))
        if better:
            state["best"] = {"epoch": epoch + 1, "val_macro_f1": vm["macro_f1"], "val_loss": vloss,
                             "val_accuracy": vm["accuracy"]}
            atomic_torch_save({"run_id": spec["run_id"], "spec": spec, "class_map": class_map,
                               "model": model.state_dict(), "best": state["best"]}, ckdir / "best.pt")
        state["epoch"] = epoch + 1
        state["elapsed_s"] = time.time() - t_start
        done = t["rep_epochs"] + epoch + 1
        eta = state["elapsed_s"] / done * (total_epochs - done)
        log(f"  [{spec['run_id']}] ft {epoch + 1}/{t['ft_epochs']} train_loss={rec['train_loss']:.4f} "
            f"val_loss={vloss:.4f} val_macroF1={vm['macro_f1']:.4f} val_acc={vm['accuracy']:.4f}"
            f"{'  *best*' if better else ''}  eta~{eta / 60:.1f}m")
        save_last(clf_head)

    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    pd.DataFrame(state["history"]).to_csv(rdir / "history.csv", index=False)
    result = {"state": "completed", "run_id": spec["run_id"], "best": state["best"],
              "optimizer_steps": state["optimizer_steps"], "image_views": state["image_views"],
              "runtime_min": round(state["elapsed_s"] / 60, 2), "peak_mem_gib_this_session": peak,
              "support_images": len(sup), "best_checkpoint_sha256": file_sha256(ckdir / "best.pt"),
              "environment": environment_info(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    set_log_file(None)
    return result


# ----------------------------------------------------------------------------- matrix / ledger
def _device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def run_matrix(cfg: dict, methods=None, fractions=None, seeds=None, tag: str = "main") -> None:
    e = cfg["experiment"]
    methods, fractions, seeds = methods or e["methods"], fractions or e["train_fractions"], seeds or e["seeds"]
    device = _device()
    if device.type != "cuda":
        raise RuntimeError("CUDA not available; refusing to launch full training on CPU (use `smoke` for CPU tests)")
    log(f"device: {torch.cuda.get_device_name(0)} | tag={tag}")
    # paired order: for each (fraction, seed) run CE then SupCon back to back
    specs = [run_spec(cfg, m, f, s) for f in fractions for s in seeds for m in methods]
    for i, spec in enumerate(specs, 1):
        rdir = run_dir(spec)
        rdir.mkdir(parents=True, exist_ok=True)
        atomic_json(rdir / "spec.json", spec)
        st = read_status(spec)
        tags = sorted(set(st.get("tags", [])) | {tag})
        if st["state"] == "completed" and (rdir / "checkpoints" / "best.pt").exists():
            if file_sha256(rdir / "checkpoints" / "best.pt") != st["best_checkpoint_sha256"]:
                raise RuntimeError(f"best.pt hash mismatch for completed run {spec['run_id']}")
            st["tags"] = tags
            atomic_json(rdir / "status.json", st)
            log(f"[{i}/{len(specs)}] {spec['run_id']} already completed (val macro-F1 "
                f"{st['best']['val_macro_f1']:.4f}) -> reused")
            continue
        log(f"===== [{i}/{len(specs)}] {spec['method'].upper()} | fraction {spec['fraction']:.0%} | seed {spec['seed']} "
            f"| run {spec['run_id']} =====")
        atomic_json(rdir / "status.json", {"state": "running", "run_id": spec["run_id"], "tags": tags,
                                           "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        try:
            res = train_pipeline(cfg, spec, device)
        except torch.cuda.OutOfMemoryError as ex:
            torch.cuda.empty_cache()
            atomic_json(rdir / "status.json", {"state": "failed", "run_id": spec["run_id"], "tags": tags,
                                               "reason": f"CUDA OOM at batch_size={spec['train']['batch_size']}: {ex}"})
            log("CUDA OOM: a common smaller batch size must be frozen as a NEW configuration; stopping.")
            raise
        except BaseException as ex:
            atomic_json(rdir / "status.json", {"state": "failed" if not isinstance(ex, KeyboardInterrupt) else "interrupted",
                                               "run_id": spec["run_id"], "tags": tags,
                                               "reason": f"{type(ex).__name__}: {ex}"})
            raise
        res["tags"] = tags
        atomic_json(rdir / "status.json", res)
        log(f"[{i}/{len(specs)}] done {spec['run_id']}: best val macro-F1 {res['best']['val_macro_f1']:.4f} "
            f"@ft-epoch {res['best']['epoch']} | {res['runtime_min']:.1f} min | peak {res['peak_mem_gib_this_session']:.2f} GiB")
    print_ledger(cfg)


def ledger(cfg: dict) -> pd.DataFrame:
    e = cfg["experiment"]
    rows = []
    for f in e["train_fractions"]:
        for s in e["seeds"]:
            for m in e["methods"]:
                spec = run_spec(cfg, m, f, s)
                st = read_status(spec)
                rows.append({"run_id": spec["run_id"], "method": m, "fraction": f, "seed": s, "state": st["state"],
                             "val_macro_f1": (st.get("best") or {}).get("val_macro_f1"),
                             "best_epoch": (st.get("best") or {}).get("epoch"),
                             "runtime_min": st.get("runtime_min"), "optimizer_steps": st.get("optimizer_steps"),
                             "image_views": st.get("image_views"), "peak_mem_gib": st.get("peak_mem_gib_this_session"),
                             "reason": st.get("reason", "")})
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/run_ledger.csv", index=False)
    return df


def print_ledger(cfg: dict) -> None:
    df = ledger(cfg)
    log("run ledger (results/run_ledger.csv):")
    print(df.drop(columns=["reason"]).to_string(index=False))
    print(df["state"].value_counts().to_string())


# ----------------------------------------------------------------------------- smoke / diagnostics
def smoke(cfg: dict, device: str = "auto", batch_size: int | None = None) -> None:
    """Tiny end-to-end pass on real data (2 images/class) + overfit diagnostic. Not a reported run."""
    dev = _device(device)
    log(f"smoke test on {dev}")
    mdir = Path(cfg["data"]["manifest_dir"])
    split = pd.read_csv(mdir / "split.csv")
    pool = split[split["partition"] == "train_pool"]
    tiny = pool.groupby("class_id").head(2)
    val = split[split["partition"] == "val"].groupby("class_id").head(2)
    n_classes = len(load_class_map(mdir))

    if batch_size:  # memory probe for the SupCon two-view batch (largest per-step footprint)
        enc = Encoder().to(dev); proj = projection_head(cfg["train"]["proj_hidden"], cfg["train"]["proj_dim"], 0).to(dev)
        enc.train(); freeze_bn_stats(enc)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-4)
        scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
        torch.cuda.reset_peak_memory_stats()
        for _ in tqdm(range(5), desc=f"memory probe bs={batch_size}"):
            x = torch.randn(batch_size * 2, 3, 224, 224, device=dev)
            y = torch.randint(0, n_classes, (batch_size,), device=dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                z = proj(enc(x))
            loss = supcon_loss(z.float().view(batch_size, 2, -1), y, 0.1)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        log(f"batch_size={batch_size} ({2 * batch_size} views): peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, loss {loss.item():.3f}")
        return

    out = Path("runs/_diagnostics")
    for m in ("ce", "supcon"):
        c = json.loads(json.dumps(cfg))
        c["train"].update(rep_epochs=1, ft_epochs=1, batch_size=8, num_workers=2)
        spec = {"method": m, "fraction": 0, "seed": 0, "train": c["train"], "run_id": f"smoke_{m}"}
        import shutil
        shutil.rmtree(out / f"smoke_{m}", ignore_errors=True)
        res = train_pipeline(c, spec, dev, max_batches=2, out_dir=out / f"smoke_{m}", sup_override=tiny, val_override=val)
        log(f"smoke {m}: ok, steps={res['optimizer_steps']} views={res['image_views']}")

    # overfit diagnostic: classifier must be able to fit 20 training images
    log("overfit diagnostic: fit 2 images/class with CE, eval on the same images")
    torch.manual_seed(0)
    from .model import build_classifier
    model = build_classifier(n_classes).to(dev)
    ds = ImageTable(tiny, cfg["data"]["cache_dir"], cfg, "eval")
    dl = make_loader(ds, 20, None, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y, _ = next(iter(dl)); x, y = x.to(dev), y.to(dev)
    model.train(); freeze_bn_stats(model)
    for step in tqdm(range(60), desc="overfit"):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    log(f"overfit diagnostic: final loss {loss.item():.4f}, train acc {acc:.2%} ({'PASS' if acc == 1.0 else 'CHECK'})")
    atomic_json(out / "overfit_diagnostic.json", {"final_loss": loss.item(), "train_acc": acc, "n_images": len(tiny)})
