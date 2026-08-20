from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Strategy(str, Enum):
    COMFORT = "comfort"
    CALM = "calm"
    ENERGIZE = "energize"
    MAINTAIN = "maintain"

    @classmethod
    def parse(cls, value: str | "Strategy") -> "Strategy":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {item.value: item for item in cls}
        if normalized not in aliases:
            raise ValueError(f"Unknown strategy {value!r}; expected {sorted(aliases)}")
        return aliases[normalized]


@dataclass(frozen=True)
class PrefixState:
    """Observed and reference means for one complete music prefix."""

    count: int
    mean_va: np.ndarray
    target_mean_va: np.ndarray
    anchor_drift: float
    violation: bool
    checks: dict[str, bool]


@dataclass(frozen=True)
class StrategyConstraint:
    """Canonical SC-CAP strategy and all-prefix constraints.

    Let q0 be Text-VA and p_j the selected Music-VA at step j.  Each
    strategy defines a clipped reference waypoint r_j.  Prefix feasibility
    compares mean(p_1, ..., p_t) with mean(r_1, ..., r_t); it never compares
    a prefix mean with the endpoint q0 + t * delta.
    """

    strategy: Strategy
    delta_a: float = 0.08
    delta_v: float = 0.06
    epsilon_v: float = 0.20
    epsilon_a: float = 0.22
    epsilon_p: float = 0.26
    congruence_radius: float = 0.30
    total_steps: int = 4

    @staticmethod
    def _va(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=float).reshape(2)

    @staticmethod
    def _clip(value: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(value, dtype=float), -1.0, 1.0)

    def reference_waypoint(self, initial_va: np.ndarray, step: int) -> np.ndarray:
        """Return r_t, the fixed-anchor reference Music-VA at step t."""
        if not 1 <= step <= self.total_steps:
            raise ValueError(f"step must be in [1, {self.total_steps}], got {step}")
        initial = self._va(initial_va)
        offset = np.zeros(2, dtype=float)
        if self.strategy is Strategy.ENERGIZE:
            offset[1] = step * self.delta_a
        elif self.strategy is Strategy.CALM:
            offset[1] = -step * self.delta_a
        elif self.strategy is Strategy.COMFORT:
            # r_1=q0 acknowledges the initial affect; recovery begins at t=2.
            offset[0] = max(0, step - 1) * self.delta_v
        return self._clip(initial + offset)

    def prefix_target(self, initial_va: np.ndarray, step: int) -> np.ndarray:
        """Return mean(r_1, ..., r_t), including clipping near VA bounds."""
        references = [self.reference_waypoint(initial_va, j) for j in range(1, step + 1)]
        return np.mean(np.asarray(references, dtype=float), axis=0)

    def desired_va(self, current_va: np.ndarray, initial_va: np.ndarray, step: int) -> np.ndarray:
        """Return the feedback-adaptive soft VA target used before content reranking."""
        current = self._va(current_va)
        initial = self._va(initial_va)
        if self.strategy is Strategy.ENERGIZE:
            return self._clip(current + np.asarray([0.0, self.delta_a]))
        if self.strategy is Strategy.CALM:
            return self._clip(current - np.asarray([0.0, self.delta_a]))
        if self.strategy is Strategy.COMFORT:
            return initial if step == 1 else self._clip(current + np.asarray([self.delta_v, 0.0]))
        return initial

    def feasible_mask(self, candidate_va: np.ndarray, current_va: np.ndarray,
                      initial_va: np.ndarray, step: int) -> tuple[np.ndarray, dict[str, object]]:
        """Evaluate the strategy-local hard constraint for all catalog items."""
        values = np.asarray(candidate_va, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("candidate_va must have shape [N, 2]")
        current = self._va(current_va)
        initial = self._va(initial_va)
        tolerance = 1.0e-8

        if self.strategy is Strategy.ENERGIZE:
            required_a = min(1.0, current[1] + self.delta_a)
            mask = ((values[:, 1] >= required_a - tolerance) &
                    (values[:, 0] >= initial[0] - self.epsilon_v - tolerance))
            info = {"phase": "arousal_increase", "required_arousal": float(required_a),
                    "valence_floor": float(max(-1.0, initial[0] - self.epsilon_v))}
        elif self.strategy is Strategy.CALM:
            required_a = max(-1.0, current[1] - self.delta_a)
            mask = ((values[:, 1] <= required_a + tolerance) &
                    (values[:, 0] >= initial[0] - self.epsilon_v - tolerance))
            info = {"phase": "arousal_decrease", "required_arousal": float(required_a),
                    "valence_floor": float(max(-1.0, initial[0] - self.epsilon_v))}
        elif self.strategy is Strategy.COMFORT and step == 1:
            mask = np.linalg.norm(values - initial[None, :], axis=1) <= self.congruence_radius + tolerance
            info = {"phase": "affective_acknowledgement",
                    "congruence_radius": float(self.congruence_radius)}
        elif self.strategy is Strategy.COMFORT:
            required_v = min(1.0, current[0] + self.delta_v)
            mask = ((values[:, 0] >= required_v - tolerance) &
                    (np.abs(values[:, 1] - current[1]) <= self.epsilon_a + tolerance))
            info = {"phase": "valence_recovery", "required_valence": float(required_v),
                    "arousal_tolerance": float(self.epsilon_a)}
        else:
            # Maintain is a set-point trajectory strategy.  Stability is imposed
            # on every prefix below; VA distance to q0 remains the soft target.
            mask = np.ones(len(values), dtype=bool)
            info = {"phase": "trajectory_stability"}
        return mask, {"strategy": self.strategy.value, **info}

    def prefix_mask(self, candidate_va: np.ndarray, history_va: np.ndarray,
                    initial_va: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the hard all-prefix constraint after appending each candidate."""
        values = np.asarray(candidate_va, dtype=float)
        history = np.asarray(history_va, dtype=float).reshape(-1, 2)
        if step != len(history) + 1:
            raise ValueError("step must equal len(history_va) + 1")
        initial = self._va(initial_va)
        prior_sum = history.sum(axis=0) if len(history) else np.zeros(2, dtype=float)
        means = (prior_sum[None, :] + values) / float(step)
        target = self.prefix_target(initial, step)
        tolerance = 1.0e-8

        if self.strategy is Strategy.ENERGIZE:
            mask = ((means[:, 1] >= target[1] - self.epsilon_p - tolerance) &
                    (means[:, 0] >= initial[0] - self.epsilon_v - tolerance))
        elif self.strategy is Strategy.CALM:
            mask = ((means[:, 1] <= target[1] + self.epsilon_p + tolerance) &
                    (means[:, 0] >= initial[0] - self.epsilon_v - tolerance))
        elif self.strategy is Strategy.COMFORT:
            mask = ((means[:, 0] >= target[0] - self.epsilon_p - tolerance) &
                    (np.abs(means[:, 1] - initial[1]) <= self.epsilon_a + tolerance))
        else:
            mask = np.linalg.norm(means - initial[None, :], axis=1) <= self.epsilon_p + tolerance
        return mask, target

    def feasible(self, candidate_va: np.ndarray, current_va: np.ndarray,
                 initial_va: np.ndarray, step: int) -> tuple[bool, dict[str, object]]:
        mask, info = self.feasible_mask(
            np.asarray(candidate_va, dtype=float).reshape(1, 2), current_va, initial_va, step
        )
        return bool(mask[0]), {**info, "hard_feasible": bool(mask[0])}

    def prefix(self, history_va: list[np.ndarray], initial_va: np.ndarray,
               *, step: int) -> PrefixState:
        if not history_va:
            initial = self._va(initial_va)
            return PrefixState(0, initial, initial, 0.0, False, {})
        values = np.asarray(history_va, dtype=float).reshape(-1, 2)
        if len(values) != step:
            raise ValueError("step must equal the complete prefix length")
        candidate = values[-1:].copy()
        mask, target = self.prefix_mask(candidate, values[:-1], initial_va, step)
        mean = values.mean(axis=0)
        initial = self._va(initial_va)
        checks: dict[str, bool]
        if self.strategy is Strategy.ENERGIZE:
            checks = {"arousal_prefix_progress": bool(mean[1] >= target[1] - self.epsilon_p),
                      "valence_prefix_safe": bool(mean[0] >= initial[0] - self.epsilon_v)}
        elif self.strategy is Strategy.CALM:
            checks = {"arousal_prefix_progress": bool(mean[1] <= target[1] + self.epsilon_p),
                      "valence_prefix_safe": bool(mean[0] >= initial[0] - self.epsilon_v)}
        elif self.strategy is Strategy.COMFORT:
            checks = {"valence_prefix_progress": bool(mean[0] >= target[0] - self.epsilon_p),
                      "arousal_prefix_stable": bool(abs(mean[1] - initial[1]) <= self.epsilon_a)}
        else:
            checks = {"trajectory_prefix_stable": bool(np.linalg.norm(mean - initial) <= self.epsilon_p)}
        return PrefixState(
            count=step,
            mean_va=mean,
            target_mean_va=target,
            anchor_drift=float(np.linalg.norm(mean - initial)),
            violation=not bool(mask[0]),
            checks=checks,
        )
