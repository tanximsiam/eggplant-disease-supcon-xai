import torch
import torch.nn.functional as F

from eggplant.supcon import supcon_loss


def naive_supcon(features, labels, t):
    b, v, _ = features.shape
    z = F.normalize(features.float(), dim=-1)
    items = [(z[i, k], labels[i].item()) for k in range(v) for i in range(b)]
    total = 0.0
    for a, (za, ya) in enumerate(items):
        denom = sum(torch.exp(za @ zb / t) for b_, (zb, _) in enumerate(items) if b_ != a)
        pos = [b_ for b_, (_, yb) in enumerate(items) if b_ != a and yb == ya]
        total += -sum(torch.log(torch.exp(za @ items[p][0] / t) / denom) for p in pos) / len(pos)
    return total / len(items)


def test_matches_naive_reference():
    torch.manual_seed(0)
    f = torch.randn(6, 2, 8)
    y = torch.tensor([0, 1, 0, 2, 1, 2])
    assert torch.allclose(supcon_loss(f, y, 0.1), naive_supcon(f, y, 0.1), atol=1e-4)


def test_other_view_is_positive_for_singleton_class():
    # label 3 appears once: its only positive is its own second view -> loss still finite
    torch.manual_seed(1)
    f = torch.randn(4, 2, 8)
    y = torch.tensor([0, 0, 1, 3])
    assert torch.isfinite(supcon_loss(f, y, 0.1))
    assert torch.allclose(supcon_loss(f, y, 0.1), naive_supcon(f, y, 0.1), atol=1e-4)


def test_permutation_invariance_and_label_alignment():
    torch.manual_seed(2)
    f = torch.randn(8, 2, 16)
    y = torch.randint(0, 3, (8,))
    perm = torch.randperm(8)
    assert torch.allclose(supcon_loss(f, y, 0.1), supcon_loss(f[perm], y[perm], 0.1), atol=1e-5)
    # mis-aligned labels must change the loss
    assert not torch.allclose(supcon_loss(f, y, 0.1), supcon_loss(f, y.roll(1), 0.1))


def test_finite_gradients_and_extreme_scale():
    f = (torch.randn(16, 2, 128) * 1e3).requires_grad_()
    y = torch.randint(0, 10, (16,))
    loss = supcon_loss(f, y, 0.05)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(f.grad).all()


def test_identical_views_lower_loss_than_random():
    torch.manual_seed(3)
    base = torch.randn(8, 1, 32)
    y = torch.arange(8)
    aligned = supcon_loss(base.repeat(1, 2, 1), y, 0.1)
    random = supcon_loss(torch.randn(8, 2, 32), y, 0.1)
    assert aligned < random
