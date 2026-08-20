import numpy as np

from sc_cap.catalog.loader import MusicCatalog
from sc_cap.evaluation import METHODS, plan_baseline


def test_all_baselines_produce_unique_plans():
    rng = np.random.default_rng(4)
    n = 100
    c = MusicCatalog(np.array([f"s{i}" for i in range(n)]), np.array(["test"] * n),
        rng.uniform(-1, 1, (n, 2)).astype("float32"), rng.random((n, 87), dtype="float32"),
        rng.random((n, 40), dtype="float32"), rng.random((n, 56), dtype="float32"),
        rng.random((n, 183), dtype="float32"), rng.normal(size=(n, 256)).astype("float32"),
        rng.normal(size=(n, 256)).astype("float32"))
    for method in METHODS:
        plan = plan_baseline(c, [0, 0], "energize", method, steps=3)
        assert len(plan) == 3
        assert len({x["song_id"] for x in plan}) == 3
