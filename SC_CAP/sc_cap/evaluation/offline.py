from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..catalog import MusicCatalog
from ..planner import PlannerConfig, SCCAPPlanner


METHODS = ("endpoint_knn", "linear_waypoint", "open_loop_sc_cap")


def _nearest(catalog: MusicCatalog, target: np.ndarray, used: set[int]) -> int:
    distances = np.linalg.norm(catalog.pred_va - target, axis=1)
    for index in np.argsort(distances, kind="stable"):
        if int(index) not in used:
            return int(index)
    raise RuntimeError("catalog exhausted")


def plan_baseline(catalog: MusicCatalog, initial_va: Any, strategy: str,
                  method: str, *, steps: int = 4,
                  config: PlannerConfig | None = None) -> list[dict[str, Any]]:
    """Generate one open-loop plan using a named ablation/baseline."""
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"unknown method {method}; expected {METHODS}")
    cfg = config or PlannerConfig(total_steps=steps)
    cfg = replace(cfg, total_steps=steps)
    initial = np.asarray(initial_va, dtype=float).reshape(2)
    used: set[int] = set()
    history: list[str] = []
    current = initial.copy()
    if method == "open_loop_sc_cap":
        planner = SCCAPPlanner(catalog, cfg)
        return [{**item, "method": method}
                for item in planner.plan_open_loop(initial, strategy, steps)]

    output: list[dict[str, Any]] = []
    if method == "endpoint_knn":
        # Endpoint target is a strategy-oriented target, not a user-truth label.
        target = initial.copy()
        if strategy == "energize": target[1] = min(1.0, target[1] + steps * cfg.delta_a)
        elif strategy == "calm": target[1] = max(-1.0, target[1] - steps * cfg.delta_a)
        elif strategy == "comfort": target[0] = min(1.0, target[0] + steps * cfg.delta_v)
    for step in range(1, steps + 1):
        if method == "linear_waypoint":
            rho = step / steps
            target = initial.copy()
            if strategy == "energize": target[1] += rho * steps * cfg.delta_a
            elif strategy == "calm": target[1] -= rho * steps * cfg.delta_a
            elif strategy == "comfort": target[0] += rho * steps * cfg.delta_v
            index = _nearest(catalog, target, used)
        elif method == "endpoint_knn":
            index = _nearest(catalog, target, used)
        else:
            raise AssertionError(f"unhandled method: {method}")
        used.add(index)
        row = catalog.as_public_row(index)
        item = {"song_id": row["song_id"], "music_pred_va": row["pred_va"],
                "step": step, "method": method}
        output.append(item)
        history.append(row["song_id"])
        current = np.asarray(row["pred_va"], dtype=float)
    return output


def evaluate_offline(plan: list[dict[str, Any]], *, initial_va: Any,
                     strategy: str, true_va_by_song: dict[str, Any] | None = None,
                     epsilon_p: float = 0.26) -> dict[str, float | bool]:
    """Evaluate a plan. `true_va_by_song` is mandatory for true-VA metrics."""
    predicted = np.asarray([initial_va] + [item["music_pred_va"] for item in plan], dtype=float)
    if true_va_by_song is None:
        raise ValueError("Pass hidden true VA labels; predicted VA is not an evaluation target")
    true = np.asarray([initial_va] + [true_va_by_song[item["song_id"]] for item in plan], dtype=float)
    deltas = np.diff(true, axis=0)
    s = strategy.lower()
    primary = deltas[:, 1] if s in {"energize", "calm"} else deltas[:, 0]
    if s == "calm": primary = -primary
    success = bool(primary.sum() > 0) if s != "maintain" else bool(np.max(np.linalg.norm(true - true[0], axis=1)) <= epsilon_p)
    reversals = np.sign(primary[1:]) != np.sign(primary[:-1]) if len(primary) > 1 else np.array([], dtype=bool)
    content_sim = [float(item["content"]["adjacent_similarity"])
                   for item in plan if item.get("content", {}).get("adjacent_similarity") is not None]
    return {"strategy_success": success,
            "final_true_va_error": float(np.linalg.norm(true[-1] - true[0])),
            "reversal_rate": float(np.mean(reversals)) if len(reversals) else 0.0,
            "max_prefix_drift": float(np.max(np.linalg.norm(true - true[0], axis=1))),
            "prefix_violation": bool(np.max(np.linalg.norm(true - true[0], axis=1)) > epsilon_p),
            "predicted_true_final_gap": float(np.linalg.norm(predicted[-1] - true[-1])),
            "content_continuity": float(np.mean(content_sim)) if content_sim else None,
            "diversity": float(len({item["song_id"] for item in plan}) / max(1, len(plan)))}
