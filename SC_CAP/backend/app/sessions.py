from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .planner import InfeasiblePlanError, RegulationPlanner


STRATEGY_QUESTIONS = {
    "energize": "相比听这首歌前，我现在是否更有精神？",
    "calm": "相比听这首歌前，我现在是否更平静？",
    "maintain": "这首音乐是否帮助我维持想保持的状态？",
    "comfort": "这首音乐是否让我感到被理解、安慰或情绪得到承接？",
}


@dataclass
class RegulationSession:
    session_id: str
    text: str
    strategy: str
    text_pred_va: list[float]
    user_initial_va: list[float]
    total_steps: int
    history: list[str] = field(default_factory=list)
    planned_music_trajectory: list[list[float]] = field(default_factory=list)
    actual_user_trajectory: list[list[float]] = field(default_factory=list)
    step_records: list[dict] = field(default_factory=list)
    infeasible_events: list[dict] = field(default_factory=list)
    pending: dict | None = None
    planner_epsilon_v: float = 0.20
    planner_epsilon_p: float = 0.26

    @classmethod
    def create(cls, text: str, strategy: str, text_pred_va: list[float],
               user_initial_va: list[float], total_steps: int) -> "RegulationSession":
        return cls(uuid.uuid4().hex, text, strategy, text_pred_va, user_initial_va, total_steps,
                   planned_music_trajectory=[list(text_pred_va)], actual_user_trajectory=[list(user_initial_va)])

    def create_first_recommendation(self, planner: RegulationPlanner) -> dict:
        # Text-VA only: user_initial_va is never passed into first planning.
        try:
            self.pending = planner.recommend(
                current_va=self.text_pred_va, initial_text_va=self.text_pred_va,
                strategy=self.strategy, history=[]
            )
        except InfeasiblePlanError as error:
            self.infeasible_events.append(error.diagnostics)
            raise
        self.history.append(self.pending["track"]["song_id"])
        self.planned_music_trajectory.append(self.pending["track"]["pred_va"])
        return self.pending

    def submit_feedback(self, planner: RegulationPlanner, *, user_felt_va: list[float],
                        strategy_rating: int, music_preference: int | None,
                        playback_seconds: float, repeat_count: int) -> tuple[dict, dict | None]:
        if self.pending is None:
            raise ValueError("no pending recommendation")
        if len(self.step_records) >= self.total_steps:
            raise ValueError("sequence is already complete")
        current = [float(user_felt_va[0]), float(user_felt_va[1])]
        if any(value < -1.0 or value > 1.0 for value in current):
            raise ValueError("felt VA must be in [-1, 1]")
        item = self.pending
        record = {
            "step": item["step"], "song_id": item["track"]["song_id"],
            "algorithm_version": item["algorithm_version"],
            "selection_status": item["selection_status"],
            "music_pred_v": item["track"]["pred_va"][0], "music_pred_a": item["track"]["pred_va"][1],
            "planning_state": item["planning_state"], "desired_music_va": item["desired_music_va"],
            "planner_score": item["planner_score"], "strategy_constraint": item["strategy_constraint"],
            "va_distance": item["va_distance"], "candidate_counts": item["candidate_counts"],
            "prefix_state": item["prefix_state"], "content_similarity": item["content"],
            "user_felt_v": current[0], "user_felt_a": current[1],
            "strategy_rating": int(strategy_rating), "music_preference": music_preference,
            "play_duration_seconds": float(playback_seconds), "repeat_count": int(repeat_count),
            "timestamp": time.time(),
        }
        self.step_records.append(record)
        self.actual_user_trajectory.append(current)
        self.pending = None
        if len(self.step_records) == self.total_steps:
            return record, None
        try:
            self.pending = planner.recommend(
                current_va=current, initial_text_va=self.text_pred_va,
                strategy=self.strategy, history=self.history
            )
        except InfeasiblePlanError as error:
            self.infeasible_events.append(error.diagnostics)
            raise
        self.history.append(self.pending["track"]["song_id"])
        self.planned_music_trajectory.append(self.pending["track"]["pred_va"])
        return record, self.pending

    def summary(self, final_ratings: dict | None = None) -> dict:
        planned = np.asarray(self.planned_music_trajectory, dtype=float)
        actual = np.asarray(self.actual_user_trajectory, dtype=float)
        n = min(len(planned), len(actual))
        p, u = planned[:n], actual[:n]
        if n > 1:
            rtd = float(np.sqrt(np.mean(np.sum(((p - p[0]) - (u - u[0])) ** 2, axis=1))))
        else:
            rtd = None
        planned_delta = np.diff(p, axis=0)
        actual_delta = np.diff(u, axis=0)
        if self.strategy in {"energize", "calm"}:
            axis, sign = 1, 1.0 if self.strategy == "energize" else -1.0
        else:
            axis, sign = 0, 1.0
        if len(planned_delta):
            direction_agreement = float(np.mean(
                np.sign(sign * planned_delta[:, axis]) == np.sign(sign * actual_delta[:, axis])
            ))
        else:
            direction_agreement = None
        user_drift = np.linalg.norm(u - u[0], axis=1)
        if self.strategy == "energize":
            strategy_success = bool(u[-1, 1] - u[0, 1] > 0 and u[-1, 0] >= u[0, 0] - self.planner_epsilon_v)
        elif self.strategy == "calm":
            strategy_success = bool(u[-1, 1] - u[0, 1] < 0 and u[-1, 0] >= u[0, 0] - self.planner_epsilon_v)
        elif self.strategy == "maintain":
            strategy_success = bool(float(user_drift.max()) <= self.planner_epsilon_p)
        else:
            # Comfort's primary user outcome is its strategy-specific rating.
            strategy_success = None
        ratings = [record["strategy_rating"] for record in self.step_records]
        return {
            "session_id": self.session_id, "strategy": self.strategy, "input_text": self.text,
            "text_pred_v": self.text_pred_va[0], "text_pred_a": self.text_pred_va[1],
            "user_initial_v": self.user_initial_va[0], "user_initial_a": self.user_initial_va[1],
            "steps": self.step_records, "planned_music_trajectory": self.planned_music_trajectory,
            "actual_user_trajectory": self.actual_user_trajectory,
            "infeasible_events": self.infeasible_events,
            "evaluation": {"strategy_success": strategy_success,
                           "stepwise_direction_agreement": direction_agreement,
                           "relative_trajectory_deviation": rtd,
                           "user_max_prefix_drift": float(user_drift.max()),
                           "user_prefix_violation": bool(user_drift.max() > self.planner_epsilon_p),
                           "strategy_rating_mean": float(np.mean(ratings)) if ratings else None},
            **(final_ratings or {}),
        }


class RegulationSessionStore:
    def __init__(self, planner: RegulationPlanner, log_path: Path) -> None:
        self.planner = planner
        self.log_path = log_path
        self.sessions: dict[str, RegulationSession] = {}

    def start(self, text: str, strategy: str, text_pred_va: list[float], user_initial_va: list[float]) -> tuple[RegulationSession, dict]:
        session = RegulationSession.create(text, strategy, text_pred_va, user_initial_va, self.planner.settings.total_steps)
        session.planner_epsilon_v = self.planner.settings.epsilon_v
        session.planner_epsilon_p = self.planner.settings.epsilon_p
        self.sessions[session.session_id] = session
        try:
            recommendation = session.create_first_recommendation(self.planner)
        except InfeasiblePlanError:
            self.record_infeasible(session)
            raise
        return session, recommendation

    def get(self, session_id: str) -> RegulationSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("session not found")
        return session

    def save(self, session: RegulationSession, final_ratings: dict) -> dict:
        result = session.summary(final_ratings)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    def record_infeasible(self, session: RegulationSession) -> None:
        """Persist hard infeasibility immediately; it is never a recommendation."""
        if not session.infeasible_events:
            return
        payload = {"event_type": "planning_infeasible", "session_id": session.session_id,
                   "strategy": session.strategy, **session.infeasible_events[-1],
                   "timestamp": time.time()}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
