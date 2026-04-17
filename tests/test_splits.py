"""Split / support selection on a SYNTHETIC manifest (test fixture only)."""
import numpy as np
import pandas as pd

from eggplant.splits import group_stratified_split, select_support


def synthetic_manifest(n_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_classes):
        n = int(rng.integers(150, 450))
        g = 0
        i = 0
        while i < n:
            size = int(rng.choice([1, 1, 1, 1, 2, 3]))
            for _ in range(size):
                rows.append({"rel_path": f"c{c}/{i:04d}.jpg", "class_id": c, "group_id": f"c{c}_g{g}"})
                i += 1
            g += 1
    return pd.DataFrame(rows)


FR = {"train_pool": 0.70, "val": 0.15, "test": 0.15}


def test_no_group_leakage_and_all_classes_everywhere():
    df = synthetic_manifest()
    df["partition"] = group_stratified_split(df, FR, 2026)
    assert df.groupby("group_id")["partition"].nunique().max() == 1
    for p in FR:
        assert df.loc[df.partition == p, "class_id"].nunique() == 10
    share = df["partition"].value_counts(normalize=True)
    assert abs(share["train_pool"] - 0.70) < 0.02 and abs(share["test"] - 0.15) < 0.02


def test_split_deterministic():
    df = synthetic_manifest()
    assert group_stratified_split(df, FR, 2026).equals(group_stratified_split(df, FR, 2026))


def test_support_nested_paired_and_within_pool():
    df = synthetic_manifest()
    df["partition"] = group_stratified_split(df, FR, 2026)
    pool = df[df.partition == "train_pool"]
    for seed in (17, 42, 2026):
        s10, s20 = select_support(pool, 0.1, seed), select_support(pool, 0.2, seed)
        assert set(s10.rel_path) <= set(s20.rel_path)
        assert set(s20.rel_path) <= set(pool.rel_path)
        assert s10.equals(select_support(pool, 0.1, seed))  # identical for both methods (method-free)
        for c in range(10):
            n = (pool.class_id == c).sum()
            k10 = (s10.class_id == c).sum()
            assert 1 <= k10 <= max(1, int(np.floor(0.1 * n))) + 2  # group indivisibility tolerance
        # whole groups only
        assert set(s20.group_id).isdisjoint(set(pool[~pool.rel_path.isin(s20.rel_path)].group_id))
    assert not set(select_support(pool, 0.1, 17).rel_path) == set(select_support(pool, 0.1, 42).rel_path)


def test_min_one_per_class():
    pool = pd.DataFrame({"rel_path": [f"{i}" for i in range(6)], "class_id": [0, 0, 0, 1, 1, 1],
                         "group_id": [f"g{i}" for i in range(6)]})
    s = select_support(pool, 0.1, 0)
    assert sorted(s.class_id.unique()) == [0, 1]
