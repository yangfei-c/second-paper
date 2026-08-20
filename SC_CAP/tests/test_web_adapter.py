import numpy as np

from backend.app.catalog import Catalog
from backend.app.planner import PlannerSettings, RegulationPlanner
from sc_cap.planner import PlannerConfig, PlannerState


def test_web_adapter_calls_the_same_canonical_core() -> None:
    song_id = np.asarray(["s0", "s1", "s2", "s3"])
    pred_va = np.asarray([
        [-0.20, -0.32], [-0.18, -0.10], [-0.16, 0.10], [-0.60, -0.20]
    ], dtype="float32")
    rng = np.random.default_rng(11)
    tags = rng.normal(size=(4, 256)).astype("float32")
    tags /= np.maximum(np.linalg.norm(tags, axis=1, keepdims=True), 1.0e-8)
    metadata = {song: {"audio_url": f"/{song}.mp3", "display_tags": []} for song in song_id}
    catalog = Catalog(song_id, pred_va, tags, np.zeros((4, 183), dtype="float32"),
                      metadata, np.ones(4, dtype=bool))
    settings = PlannerConfig(k=4)
    assert PlannerSettings is PlannerConfig
    web = RegulationPlanner(catalog, settings)

    canonical = web.core.recommend(PlannerState.initial([-0.2, -0.4]), "energize")
    deployed = web.recommend(current_va=[-0.2, -0.4], initial_text_va=[-0.2, -0.4],
                             strategy="energize", history=[])
    assert deployed["song_id"] == canonical["song_id"]
    assert deployed["music_pred_va"] == canonical["music_pred_va"]
    assert deployed["prefix_state"] == canonical["prefix_state"]
    assert deployed["track"]["song_id"] == canonical["song_id"]

    history = [canonical["song_id"]]
    feedback = [-0.18, -0.20]
    canonical_next = web.core.recommend(
        PlannerState.after_feedback([-0.2, -0.4], feedback, history), "energize"
    )
    deployed_next = web.recommend(
        current_va=feedback, initial_text_va=[-0.2, -0.4],
        strategy="energize", history=history,
    )
    assert deployed_next["song_id"] == canonical_next["song_id"]
    assert deployed_next["planning_state"]["source"] == "user_felt_va"
    assert deployed_next["prefix_state"] == canonical_next["prefix_state"]
