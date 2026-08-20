from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from ..strategies.constraints import Strategy, StrategyConstraint


ALGORITHM_VERSION = "sc-cap-v1"
StateSource = Literal["text_va", "user_felt_va", "music_va"]


def _validated_va(value: Any, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain two finite VA values")
    if (array < -1.0).any() or (array > 1.0).any():
        raise ValueError(f"{name} must be in [-1, 1]")
    return float(array[0]), float(array[1])


@dataclass(frozen=True)
class PlannerState:
    """Canonical SC-CAP state.

    q0 is the fixed Text-VA planning anchor.  c_t is q0 at t=1 and the
    previous user felt-VA observation in a closed-loop session for t>=2.
    ``music_va`` is reserved for the explicitly named open-loop baseline.
    """

    q0: tuple[float, float]
    current_va: tuple[float, float]
    history: tuple[str, ...] = ()
    source: StateSource = "text_va"

    def __post_init__(self) -> None:
        object.__setattr__(self, "q0", _validated_va(self.q0, "q0"))
        object.__setattr__(self, "current_va", _validated_va(self.current_va, "current_va"))
        object.__setattr__(self, "history", tuple(str(song) for song in self.history))
        if self.source not in {"text_va", "user_felt_va", "music_va"}:
            raise ValueError(f"unknown state source: {self.source}")
        if not self.history:
            if self.source != "text_va":
                raise ValueError("the first planning state must be sourced from Text-VA")
            if not np.allclose(self.current_va, self.q0, atol=1.0e-8):
                raise ValueError("c1 must equal q0")
        elif self.source == "text_va":
            raise ValueError("steps t>=2 must use user_felt_va or explicit open-loop music_va")

    @property
    def step(self) -> int:
        return len(self.history) + 1

    @classmethod
    def initial(cls, text_va: Any) -> "PlannerState":
        q0 = _validated_va(text_va, "text_va")
        return cls(q0=q0, current_va=q0, history=(), source="text_va")

    @classmethod
    def after_feedback(cls, text_va: Any, user_felt_va: Any,
                       history: list[str] | tuple[str, ...]) -> "PlannerState":
        if not history:
            raise ValueError("feedback state requires at least one selected song")
        return cls(q0=_validated_va(text_va, "text_va"),
                   current_va=_validated_va(user_felt_va, "user_felt_va"),
                   history=tuple(history), source="user_felt_va")

    @classmethod
    def open_loop(cls, text_va: Any, music_va: Any,
                  history: list[str] | tuple[str, ...]) -> "PlannerState":
        if not history:
            return cls.initial(text_va)
        return cls(q0=_validated_va(text_va, "text_va"),
                   current_va=_validated_va(music_va, "music_va"),
                   history=tuple(history), source="music_va")


@dataclass(frozen=True)
class PlannerConfig:
    total_steps: int = 4
    k: int = 200
    delta_a: float = 0.08
    delta_v: float = 0.06
    epsilon_v: float = 0.20
    epsilon_a: float = 0.22
    epsilon_p: float = 0.26
    congruence_radius: float = 0.30
    alpha: float = 0.65

    def __post_init__(self) -> None:
        if self.total_steps < 1 or self.k < 1:
            raise ValueError("total_steps and k must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        for name in ("delta_a", "delta_v", "epsilon_v", "epsilon_a",
                     "epsilon_p", "congruence_radius"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


class InfeasiblePlanError(ValueError):
    """Raised when no catalog item satisfies all canonical hard constraints."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(
            "SC-CAP infeasible at step "
            f"{diagnostics['step']} for strategy={diagnostics['strategy']} "
            f"(hard_feasible_count={diagnostics['candidate_counts']['hard_feasible']})"
        )


class SCCAPPlanner:
    """Canonical strategy-conditioned, all-prefix-constrained SC-CAP planner."""

    def __init__(self, catalog: Any, config: PlannerConfig | None = None):
        self.catalog = catalog
        self.config = config or PlannerConfig()
        self._song_id = np.asarray(catalog.song_id)
        self._va = np.asarray(catalog.pred_va, dtype=np.float32)
        self._tags = np.asarray(catalog.tag_embedding, dtype=np.float32)
        n = len(self._song_id)
        if self._va.shape != (n, 2):
            raise ValueError(f"catalog pred_va must have shape ({n}, 2)")
        if self._tags.ndim != 2 or self._tags.shape[0] != n:
            raise ValueError("catalog tag_embedding must have shape [N, D]")
        if not np.isfinite(self._va).all() or not np.isfinite(self._tags).all():
            raise ValueError("catalog VA/content features must be finite")
        norms = np.linalg.norm(self._tags, axis=1, keepdims=True)
        self._tags = self._tags / np.maximum(norms, 1.0e-8)
        playable = getattr(catalog, "playable", None)
        self._eligible = (np.ones(n, dtype=bool) if playable is None
                          else np.asarray(playable, dtype=bool))
        if self._eligible.shape != (n,):
            raise ValueError("catalog playable mask must have shape [N]")
        self._index = {str(song): i for i, song in enumerate(self._song_id)}
        if len(self._index) != n:
            raise ValueError("catalog song_id values must be unique")

    def _constraint(self, strategy: Strategy) -> StrategyConstraint:
        c = self.config
        return StrategyConstraint(strategy, c.delta_a, c.delta_v, c.epsilon_v,
                                  c.epsilon_a, c.epsilon_p, c.congruence_radius,
                                  c.total_steps)

    def _history_indices(self, history: tuple[str, ...]) -> list[int]:
        missing = [song for song in history if song not in self._index]
        if missing:
            raise ValueError(f"history contains unknown song_id values: {missing[:3]}")
        if len(set(history)) != len(history):
            raise ValueError("history must not contain repeated song_id values")
        return [self._index[song] for song in history]

    def index_for_song(self, song_id: str) -> int:
        try:
            return int(self._index[str(song_id)])
        except KeyError as error:
            raise KeyError(f"unknown song_id: {song_id}") from error

    def _content_scores(self, indices: np.ndarray, history: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Balance adjacent continuity against non-adjacent redundancy.

        Cosine similarities are normalized to [0, 1].  The latest item is
        excluded from the redundancy term so continuity is not counted twice.
        """
        if not history:
            zeros = np.zeros(len(indices), dtype=np.float32)
            return zeros, np.full(len(indices), np.nan), np.full(len(indices), np.nan)
        similarities = self._tags[indices] @ self._tags[history].T
        normalized = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)
        adjacent = normalized[:, -1]
        if len(history) > 1:
            redundancy = normalized[:, :-1].max(axis=1)
        else:
            redundancy = np.zeros(len(indices), dtype=np.float32)
        score = self.config.alpha * (1.0 - adjacent) + (1.0 - self.config.alpha) * redundancy
        return score.astype(np.float32), adjacent.astype(np.float32), redundancy.astype(np.float32)

    def recommend(self, state: PlannerState, strategy: str | Strategy,
                  n_candidates: int | None = None) -> dict[str, Any]:
        if not isinstance(state, PlannerState):
            raise TypeError("recommend requires a canonical PlannerState")
        parsed = Strategy.parse(strategy)
        if state.step > self.config.total_steps:
            raise ValueError(
                f"sequence already has {len(state.history)} songs (T={self.config.total_steps})"
            )
        current = np.asarray(state.current_va, dtype=float)
        initial = np.asarray(state.q0, dtype=float)
        history_indices = self._history_indices(state.history)
        history_va = self._va[history_indices] if history_indices else np.empty((0, 2), dtype=float)
        constraint = self._constraint(parsed)

        used = np.zeros(len(self._song_id), dtype=bool)
        used[history_indices] = True
        available = self._eligible & ~used
        strategy_mask, strategy_info = constraint.feasible_mask(
            self._va, current, initial, state.step
        )
        prefix_mask, prefix_target = constraint.prefix_mask(
            self._va, history_va, initial, state.step
        )
        hard_mask = available & strategy_mask & prefix_mask
        hard_candidates = np.flatnonzero(hard_mask)
        counts = {
            "catalog": int(len(self._song_id)),
            "eligible_unseen": int(np.count_nonzero(available)),
            "strategy_feasible": int(np.count_nonzero(available & strategy_mask)),
            "prefix_feasible": int(np.count_nonzero(available & prefix_mask)),
            "hard_feasible": int(len(hard_candidates)),
        }
        state_record = {
            "q0_text_va": [float(initial[0]), float(initial[1])],
            "current_va": [float(current[0]), float(current[1])],
            "source": state.source,
        }
        if not len(hard_candidates):
            raise InfeasiblePlanError({
                "algorithm_version": ALGORITHM_VERSION,
                "selection_status": "infeasible",
                "reason": "no item satisfies playable/no-repeat, strategy, and prefix constraints",
                "step": state.step,
                "strategy": parsed.value,
                "planning_state": state_record,
                "prefix_target_mean_va": prefix_target.astype(float).tolist(),
                "candidate_counts": counts,
            })

        desired = constraint.desired_va(current, initial, state.step)
        distances = np.linalg.norm(self._va[hard_candidates] - desired[None, :], axis=1)
        va_order = np.argsort(distances, kind="stable")
        shortlist_size = min(
            len(hard_candidates),
            int(n_candidates) if n_candidates is not None else self.config.k,
        )
        if shortlist_size < 1:
            raise ValueError("n_candidates must be positive")
        shortlist = hard_candidates[va_order[:shortlist_size]]
        shortlist_distances = distances[va_order[:shortlist_size]]
        content_score, adjacent, redundancy = self._content_scores(shortlist, history_indices)
        choice_position = int(np.argmin(content_score)) if history_indices else 0
        index = int(shortlist[choice_position])

        selected_prefix = [self._va[i] for i in history_indices] + [self._va[index]]
        prefix = constraint.prefix(selected_prefix, initial, step=state.step)
        local_ok, local_info = constraint.feasible(self._va[index], current, initial, state.step)
        if not local_ok or prefix.violation:
            raise AssertionError("canonical mask/selected-item constraint mismatch")

        adjacent_value = None if np.isnan(adjacent[choice_position]) else float(adjacent[choice_position])
        redundancy_value = None if np.isnan(redundancy[choice_position]) else float(redundancy[choice_position])
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "selection_status": "hard_feasible",
            "song_id": str(self._song_id[index]),
            "music_pred_va": self._va[index].astype(float).tolist(),
            "step": state.step,
            "strategy": parsed.value,
            "planning_state": state_record,
            "desired_music_va": desired.astype(float).tolist(),
            "planner_score": float(content_score[choice_position] if history_indices
                                   else shortlist_distances[choice_position]),
            "va_distance": float(shortlist_distances[choice_position]),
            "strategy_constraint": {**strategy_info, **local_info, "hard_feasible": True},
            "prefix_state": {
                "count": prefix.count,
                "mean_music_va": prefix.mean_va.astype(float).tolist(),
                "target_mean_va": prefix.target_mean_va.astype(float).tolist(),
                "anchor_drift": prefix.anchor_drift,
                "checks": prefix.checks,
                "within_constraint": True,
                "violation": False,
            },
            "content": {
                "adjacent_similarity": adjacent_value,
                "nonadjacent_max_similarity": redundancy_value,
            },
            "candidate_count": counts["hard_feasible"],
            "candidate_counts": counts,
            "shortlist_count": int(shortlist_size),
        }

    def plan_open_loop(self, initial_va: Any, strategy: str | Strategy,
                       steps: int | None = None) -> list[dict[str, Any]]:
        """Explicit open-loop baseline: c_{t+1}=p_t, never user felt-VA."""
        steps = int(steps or self.config.total_steps)
        if steps > self.config.total_steps:
            raise ValueError("steps cannot exceed configured total_steps")
        q0 = _validated_va(initial_va, "initial_va")
        state = PlannerState.initial(q0)
        result: list[dict[str, Any]] = []
        for _ in range(steps):
            item = self.recommend(state, strategy)
            result.append(item)
            history = [*state.history, str(item["song_id"])]
            state = PlannerState.open_loop(q0, item["music_pred_va"], history)
        return result

    @staticmethod
    def config_dict(config: PlannerConfig) -> dict[str, Any]:
        return asdict(config)
