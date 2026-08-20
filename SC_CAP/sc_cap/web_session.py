from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import evaluate_dual_trajectory
from .planner import InfeasiblePlanError, PlannerState, SCCAPPlanner


@dataclass
class SCSession:
    planner: SCCAPPlanner
    session_id: str
    strategy: str
    input_text: str
    text_pred_va: list[float]
    user_initial_va: list[float]
    planned: list[list[float]] = field(default_factory=list)
    actual: list[list[float]] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    pending_result: dict[str, Any] | None = None
    infeasible_events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, planner: SCCAPPlanner, *, input_text: str, strategy: str,
               text_pred_va: list[float], user_initial_va: list[float]) -> "SCSession":
        initial = [float(x) for x in text_pred_va]
        user_initial = [float(x) for x in user_initial_va]
        return cls(planner, uuid.uuid4().hex, strategy, input_text, initial, user_initial,
                   planned=[initial], actual=[user_initial])

    def recommend_next(self) -> dict[str, Any]:
        if not self.history:
            state = PlannerState.initial(self.text_pred_va)
        else:
            if len(self.actual) != len(self.history) + 1:
                raise ValueError("felt-VA feedback is required before the next recommendation")
            state = PlannerState.after_feedback(
                self.text_pred_va, self.actual[-1], self.history
            )
        try:
            result = self.planner.recommend(state, self.strategy)
        except InfeasiblePlanError as error:
            self.infeasible_events.append(error.diagnostics)
            raise
        self.history.append(str(result["song_id"]))
        self.planned.append([float(x) for x in result["music_pred_va"]])
        self.pending_result = result
        return result

    def submit_feedback(self, *, step: int, felt_va: list[float], strategy_rating: float,
                        planner_result: dict[str, Any]) -> dict[str, Any]:
        if step != len(self.steps) + 1:
            raise ValueError(f"feedback step must be {len(self.steps) + 1}, got {step}")
        felt = [float(x) for x in felt_va]
        if len(felt) != 2 or any(x < -1 or x > 1 for x in felt):
            raise ValueError("felt_va must contain two values in [-1, 1]")
        self.actual.append(felt)
        event = {"step": step, "song_id": planner_result["song_id"],
                 "algorithm_version": planner_result["algorithm_version"],
                 "selection_status": planner_result["selection_status"],
                 "music_pred_v": planner_result["music_pred_va"][0],
                 "music_pred_a": planner_result["music_pred_va"][1],
                 "planning_state": planner_result["planning_state"],
                 "desired_music_va": planner_result["desired_music_va"],
                 "planner_score": planner_result["planner_score"],
                 "va_distance": planner_result["va_distance"],
                 "candidate_counts": planner_result["candidate_counts"],
                 "strategy_constraint": planner_result["strategy_constraint"],
                 "prefix_state": planner_result["prefix_state"],
                 "content_similarity": planner_result["content"],
                 "user_felt_v": felt[0], "user_felt_a": felt[1],
                 "strategy_rating": float(strategy_rating), "timestamp": time.time()}
        self.steps.append(event)
        self.pending_result = None
        return event

    def finish(self, *, overall_strategy_fit: float | None = None,
               satisfaction: float | None = None, enjoyment: float | None = None,
               smoothness: float | None = None, willingness_to_use_again: float | None = None) -> dict[str, Any]:
        return {"session_id": self.session_id, "strategy": self.strategy,
                "input_text": self.input_text, "text_pred_v": self.text_pred_va[0],
                "text_pred_a": self.text_pred_va[1], "user_initial_v": self.user_initial_va[0],
                "user_initial_a": self.user_initial_va[1], "steps": self.steps,
                "planned_music_trajectory": self.planned,
                "actual_user_trajectory": self.actual,
                "infeasible_events": self.infeasible_events,
                "overall_strategy_fit": overall_strategy_fit,
                "satisfaction": satisfaction, "enjoyment": enjoyment,
                "smoothness": smoothness, "willingness_to_use_again": willingness_to_use_again,
                "evaluation": evaluate_dual_trajectory(
                    self.planned, self.actual, self.strategy,
                    strategy_ratings=[float(step["strategy_rating"]) for step in self.steps])}

    def save(self, path: str | Path, **ratings: float | None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.finish(**ratings), ensure_ascii=False) + "\n")
