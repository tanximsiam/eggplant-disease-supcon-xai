"""Thin CLI: python -m eggplant <command> [--config configs/experiment.yaml]."""
from __future__ import annotations

import argparse

from .utils import load_config


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m eggplant")
    p.add_argument("--config", default="configs/experiment.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="download + verify + extract the original-image archive")
    dl.add_argument("--local-zip", help="use an already downloaded 'Original Dataset.zip' (SHA-256 still verified)")
    sub.add_parser("audit", help="build image manifest, integrity + duplicate audit, contact sheets, cache")
    sub.add_parser("split", help="lock train-pool/val/test split + nested support subsets")
    sm = sub.add_parser("smoke", help="tiny smoke test + overfit diagnostic (not a reported run)")
    sm.add_argument("--device", default="auto")
    sm.add_argument("--batch-size", type=int, default=None, help="probe a batch size (memory check)")
    tr = sub.add_parser("train", help="run training pipelines (resumable)")
    tr.add_argument("--methods", nargs="+", default=None)
    tr.add_argument("--fractions", nargs="+", type=float, default=None)
    tr.add_argument("--seeds", nargs="+", type=int, default=None)
    tr.add_argument("--tag", default="main", help="run group: 'pilot' or 'main'")
    sub.add_parser("ledger", help="print run ledger (completed / failed / pending)")
    sub.add_parser("evaluate", help="final test evaluation of frozen, validation-selected checkpoints")
    sub.add_parser("xai", help="Grad-CAM / occlusion / deletion / sanity evaluation")
    ex = sub.add_parser("export", help="export demo checkpoints + results bundle")
    ex.add_argument("--out", default="exports")
    a = p.parse_args()
    cfg = load_config(a.config)

    if a.cmd == "download":
        from .download import acquire
        acquire(cfg, local_zip=a.local_zip)
    elif a.cmd == "audit":
        from .audit import run_audit
        run_audit(cfg)
    elif a.cmd == "split":
        from .splits import make_splits
        make_splits(cfg)
    elif a.cmd == "smoke":
        from .train import smoke
        smoke(cfg, device=a.device, batch_size=a.batch_size)
    elif a.cmd == "train":
        from .train import run_matrix
        run_matrix(cfg, methods=a.methods, fractions=a.fractions, seeds=a.seeds, tag=a.tag)
    elif a.cmd == "ledger":
        from .train import print_ledger
        print_ledger(cfg)
    elif a.cmd == "evaluate":
        from .evaluate import evaluate_all
        evaluate_all(cfg)
    elif a.cmd == "xai":
        from .xai import run_xai
        run_xai(cfg)
    elif a.cmd == "export":
        from .export import export
        export(cfg, a.out)


if __name__ == "__main__":
    main()
