# Progress log (resume point across sessions)

Execution order: inspect → acquire/audit → lock manifests → implement/test → paired pilot → freeze recipe → matrix → evaluation → XAI → demo → report.

- 2026-04-17 — environment: WSL2 Ubuntu, Python 3.12 venv (`.venv`), torch 2.5.1+cu124, RTX 3060 Ti 8 GB, WSL RAM 7.6 GB. Installed `python3.12-venv` via apt.
- 2026-04-17 — code + 13 unit tests passing (synthetic fixtures, CPU).
- 2026-04-17 — dataset: Mendeley API metadata fetched (CC BY 4.0, v1, published 2024-07-10); original archive SHA-256 verified; 3116 JPGs / 10 classes, all 800x800 RGB, EXIF orientation 1.
- 2026-04-17 — audit: 128 exact-duplicate pairs (all "Fresh Eggplant Leaf") -> 128 redundant copies excluded; 1 cross-label near-dup (Defect vs Fresh Eggplant, same photo) quarantined (2 images); 41 heuristic pHash groups kept within one partition. 2986 eligible.
- 2026-04-17 — split locked (sha256 87cc8263...), supports f0.10 = 203 imgs, f0.20 = 412 imgs per seed (nested).
- 2026-04-17 — smoke OK (both methods, overfit diag 100%); memory: bs16 1.47 GiB, bs32 2.81 GiB peak.
- 2026-04-17 — paired pilot f=0.10 seed 42: CE val mF1 0.8047, SupCon 0.8082; no NaNs; 585 steps / 15225 views each, ~1.2 min each.
- 2026-04-17 — RECIPE FROZEN unchanged (configs/experiment.yaml, batch 16). Pilot runs reused by the matrix (identical run IDs). Next: `bash scripts/run.sh train` (resumable).
- 2026-04-17 — matrix 12/12 completed, 0 failed. Final eval: macro-F1 CE 0.776±0.027 / SupCon 0.803±0.015 (10%); CE 0.843±0.018 / SupCon 0.845±0.013 (20%).
- 2026-04-17 — XAI done (results/xai/), export done (exports/), Streamlit demo verified in browser (upload, Grad-CAM, occlusion, corrupt-file rejection). README report written.
- STATUS: all stages complete. Committed to `main` and pushed to GitHub (checkpoints/raw data excluded; backup bundle in `exports/`).
