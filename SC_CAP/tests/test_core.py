import numpy as np
import pytest

from sc_cap.catalog.loader import MusicCatalog
from sc_cap.planner import (InfeasiblePlanError, PlannerConfig, PlannerState,
                            SCCAPPlanner)
from sc_cap.strategies import Strategy, StrategyConstraint


def catalog_from_va(values: list[list[float]]) -> MusicCatalog:
    va = np.asarray(values, dtype="float32")
    n = len(va)
    rng = np.random.default_rng(7)
    return MusicCatalog(
        np.asarray([f"s{i}" for i in range(n)]), np.asarray(["test"] * n), va,
        rng.random((n, 87), dtype="float32"),
        rng.random((n, 40), dtype="float32"),
        rng.random((n, 56), dtype="float32"),
        rng.random((n, 183), dtype="float32"),
        rng.normal(size=(n, 256)).astype("float32"),
        rng.normal(size=(n, 256)).astype("float32"),
    )


def test_canonical_state_enforces_text_first_then_feedback() -> None:
    first = PlannerState.initial([-0.2, -0.4])
    assert first.step == 1
    assert first.current_va == first.q0
    second = PlannerState.after_feedback(first.q0, [-0.1, -0.2], ["s0"])
    assert second.step == 2
    assert second.source == "user_felt_va"
    with pytest.raises(ValueError, match="c1 must equal q0"):
        PlannerState(q0=(-0.2, -0.4), current_va=(0.1, 0.1))


def test_prefix_target_is_mean_of_reference_trajectory() -> None:
    q0 = np.asarray([0.0, 0.0])
    energize = StrategyConstraint(Strategy.ENERGIZE, total_steps=4, delta_a=0.08)
    comfort = StrategyConstraint(Strategy.COMFORT, total_steps=4, delta_v=0.06)
    assert np.allclose(energize.prefix_target(q0, 4), [0.0, 0.20])
    assert np.allclose(comfort.prefix_target(q0, 4), [0.09, 0.0])


def test_planner_returns_only_hard_feasible_items_and_uses_feedback_state() -> None:
    catalog = catalog_from_va([
        [-0.20, -0.32], [-0.18, -0.15], [-0.16, 0.05], [-0.14, 0.25],
        [-0.70, -0.10], [0.20, -0.80],
    ])
    planner = SCCAPPlanner(catalog, PlannerConfig(total_steps=4, k=6))
    first_state = PlannerState.initial([-0.2, -0.4])
    first = planner.recommend(first_state, "energize")
    assert first["selection_status"] == "hard_feasible"
    assert first["planning_state"]["source"] == "text_va"
    assert first["strategy_constraint"]["hard_feasible"] is True
    assert first["prefix_state"]["violation"] is False

    second_state = PlannerState.after_feedback(
        first_state.q0, [-0.15, -0.25], [first["song_id"]]
    )
    second = planner.recommend(second_state, "energize")
    assert second["planning_state"]["current_va"] == [-0.15, -0.25]
    assert second["planning_state"]["source"] == "user_felt_va"
    assert second["song_id"] != first["song_id"]


def test_infeasible_is_explicit_and_never_relaxed() -> None:
    catalog = catalog_from_va([[0.1, 0.2], [0.2, 0.4]])
    planner = SCCAPPlanner(catalog, PlannerConfig(k=2))
    with pytest.raises(InfeasiblePlanError) as caught:
        planner.recommend(PlannerState.initial([0.0, 0.95]), "energize")
    diagnostics = caught.value.diagnostics
    assert diagnostics["selection_status"] == "infeasible"
    assert diagnostics["candidate_counts"]["hard_feasible"] == 0


@pytest.mark.parametrize(
    ("strategy", "candidate", "expected_phase"),
    [
        ("energize", [0.0, 0.08], "arousal_increase"),
        ("calm", [0.0, -0.08], "arousal_decrease"),
        ("maintain", [0.0, 0.0], "trajectory_stability"),
        ("comfort", [0.05, 0.05], "affective_acknowledgement"),
    ],
)
def test_four_strategy_definitions(strategy: str, candidate: list[float], expected_phase: str) -> None:
    constraint = StrategyConstraint(Strategy.parse(strategy))
    ok, info = constraint.feasible(np.asarray(candidate), np.zeros(2), np.zeros(2), 1)
    assert ok is True
    assert info["phase"] == expected_phase
