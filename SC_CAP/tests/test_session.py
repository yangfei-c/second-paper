import numpy as np
import pytest

from sc_cap.catalog.loader import MusicCatalog
from sc_cap.planner import InfeasiblePlanError, PlannerConfig, SCCAPPlanner
from sc_cap.web_session import SCSession


def test_session_preserves_dual_trajectories():
    rng = np.random.default_rng(2)
    n = 30
    c = MusicCatalog(np.array([f"s{i}" for i in range(n)]), np.array(["test"] * n),
        rng.uniform(-1, 1, (n, 2)).astype("float32"), rng.random((n, 87), dtype="float32"),
        rng.random((n, 40), dtype="float32"), rng.random((n, 56), dtype="float32"),
        rng.random((n, 183), dtype="float32"), rng.normal(size=(n, 256)).astype("float32"),
        rng.normal(size=(n, 256)).astype("float32"))
    s = SCSession.create(SCCAPPlanner(c, PlannerConfig(k=n)), input_text="x",
                         strategy="maintain", text_pred_va=[0, 0], user_initial_va=[0.1, 0.1])
    result = s.recommend_next()
    assert result["planning_state"]["source"] == "text_va"
    assert result["planning_state"]["current_va"] == [0.0, 0.0]
    s.submit_feedback(step=1, felt_va=[0.2, 0.1], strategy_rating=4, planner_result=result)
    done = s.finish()
    assert len(done["planned_music_trajectory"]) == 2
    assert len(done["actual_user_trajectory"]) == 2


def test_session_records_infeasible_planning_event():
    rng = np.random.default_rng(9)
    n = 2
    c = MusicCatalog(np.array(["s0", "s1"]), np.array(["test"] * n),
        np.array([[0.0, 0.2], [0.1, 0.3]], dtype="float32"),
        rng.random((n, 87), dtype="float32"), rng.random((n, 40), dtype="float32"),
        rng.random((n, 56), dtype="float32"), rng.random((n, 183), dtype="float32"),
        rng.normal(size=(n, 256)).astype("float32"), rng.normal(size=(n, 256)).astype("float32"))
    session = SCSession.create(SCCAPPlanner(c, PlannerConfig(k=n)), input_text="x",
        strategy="energize", text_pred_va=[0.0, 0.95], user_initial_va=[0.0, 0.5])
    with pytest.raises(InfeasiblePlanError):
        session.recommend_next()
    assert session.infeasible_events[0]["selection_status"] == "infeasible"
