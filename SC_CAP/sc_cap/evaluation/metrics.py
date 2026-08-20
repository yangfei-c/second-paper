from __future__ import annotations

from typing import Any

import numpy as np

from ..strategies.constraints import Strategy


def _direction(strategy: Strategy, delta: np.ndarray) -> int:
    axis = 1 if strategy in {Strategy.ENERGIZE, Strategy.CALM} else 0
    value = float(delta[axis])
    if strategy in {Strategy.CALM, Strategy.COMFORT}:
        value = -value if strategy is Strategy.CALM else value
    return 1 if value > 0.02 else -1 if value < -0.02 else 0


def evaluate_dual_trajectory(planned: Any, actual: Any, strategy: str,
                             epsilon_p: float = 0.26,
                             strategy_ratings: list[float] | None = None) -> dict[str, float | int | bool | None]:
    """Compare planned music VA and reported user felt VA without equating them."""
    p = np.asarray(planned, dtype=float)
    u = np.asarray(actual, dtype=float)
    if p.ndim != 2 or u.ndim != 2 or p.shape[1] != 2 or u.shape[1] != 2 or len(p) != len(u):
        raise ValueError("planned and actual must be equal [T,2] trajectories")
    s = Strategy.parse(strategy)
    dp, du = np.diff(p, axis=0), np.diff(u, axis=0)
    agreement = [_direction(s, a) == _direction(s, b) for a, b in zip(dp, du)]
    planned_rel, actual_rel = p - p[0], u - u[0]
    rtd = float(np.sqrt(np.mean(np.sum((planned_rel - actual_rel) ** 2, axis=1))))
    drift = np.linalg.norm(actual_rel, axis=1)
    if s is Strategy.ENERGIZE:
        success = bool(u[-1, 1] - u[0, 1] > 0)
    elif s is Strategy.CALM:
        success = bool(u[-1, 1] - u[0, 1] < 0)
    elif s is Strategy.MAINTAIN:
        success = bool(float(drift.max()) <= epsilon_p)
    else:
        success = bool(u[-1, 0] - u[0, 0] >= 0)
    rating_mean = None if not strategy_ratings else float(np.mean(strategy_ratings))
    return {"strategy_success": success, "strategy_success_rate": float(success),
            "stepwise_direction_agreement": float(np.mean(agreement)) if agreement else 0.0,
            "relative_trajectory_deviation": rtd,
            "user_max_prefix_drift": float(drift.max()),
            "user_prefix_violation": bool(drift.max() > epsilon_p),
            "strategy_rating_mean": rating_mean}
