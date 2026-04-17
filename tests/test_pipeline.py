"""Tiny train / resume / checkpoint / inference / XAI pass on SYNTHETIC images (CPU, test only)."""
import torch

from eggplant import train as T
from eggplant.data import load_class_map
from eggplant.model import build_classifier
from eggplant.xai import deletion_curve, gradcam, mean_fill, occlusion, saliency_order, target_probs


def _spec(cfg, method):
    return {"method": method, "fraction": 0, "seed": 7, "train": cfg["train"], "run_id": f"test_{method}"}


def _run(cfg, df, out, method):
    return T.train_pipeline(cfg, _spec(cfg, method), torch.device("cpu"), out_dir=out, sup_override=df,
                            val_override=df)


def test_both_methods_train_and_checkpoint(synthetic_cfg):
    cfg, df, tmp = synthetic_cfg
    for m in ("ce", "supcon"):
        res = _run(cfg, df, tmp / m, m)
        assert res["state"] == "completed"
        ck = torch.load(tmp / m / "checkpoints" / "best.pt", weights_only=False)
        assert ck["class_map"] == load_class_map(cfg["data"]["manifest_dir"])
        assert ck["spec"]["run_id"] == f"test_{m}"
        model = build_classifier(3, pretrained=False)
        model.load_state_dict(ck["model"])  # loads cleanly into the inference architecture


def test_resume_is_deterministic(synthetic_cfg, monkeypatch):
    cfg, df, tmp = synthetic_cfg
    _run(cfg, df, tmp / "full", "supcon")
    full = torch.load(tmp / "full" / "checkpoints" / "last.pt", weights_only=False)

    calls = {"n": 0}
    orig = T._check_finite

    def interrupt(loss, where):
        calls["n"] += 1
        if where.startswith("rep epoch 1"):  # die at the start of the 2nd representation epoch
            raise KeyboardInterrupt
        orig(loss, where)

    monkeypatch.setattr(T, "_check_finite", interrupt)
    try:
        _run(cfg, df, tmp / "res", "supcon")
    except KeyboardInterrupt:
        pass
    monkeypatch.setattr(T, "_check_finite", orig)
    _run(cfg, df, tmp / "res", "supcon")
    resumed = torch.load(tmp / "res" / "checkpoints" / "last.pt", weights_only=False)
    for k, v in full["encoder"].items():
        assert torch.allclose(v.float(), resumed["encoder"][k].float(), atol=1e-6), k
    assert full["state"]["optimizer_steps"] == resumed["state"]["optimizer_steps"]


def test_resume_rejects_mismatched_spec(synthetic_cfg):
    cfg, df, tmp = synthetic_cfg
    _run(cfg, df, tmp / "x", "ce")
    bad = _spec(cfg, "ce") | {"seed": 8}
    try:
        T.train_pipeline(cfg, bad, torch.device("cpu"), out_dir=tmp / "x", sup_override=df, val_override=df)
    except RuntimeError as e:
        assert "mismatch" in str(e)
    else:
        raise AssertionError("resume with a different spec must fail")


def test_xai_tiny_pass():
    torch.manual_seed(0)
    model = build_classifier(3, pretrained=False).eval()
    x = torch.rand(3, 64, 64)
    cam, const = gradcam(model, x, 1)
    assert cam.shape == (64, 64) and (const or (cam.min() == 0 and abs(cam.max() - 1) < 1e-6))
    assert not any(m._forward_hooks for m in model.modules())  # hooks removed
    occ = occlusion(model, x, 1, 32, 16, mean_fill(x))
    assert occ.shape == (64, 64)
    curve = deletion_curve(model, x, saliency_order(cam), mean_fill(x), 1, 10)
    assert len(curve) == 11
    assert abs(curve[0] - target_probs(model, x[None], 1)[0]) < 1e-5
    full = target_probs(model, mean_fill(x)[None], 1)[0]
    assert abs(curve[-1] - full) < 1e-5
