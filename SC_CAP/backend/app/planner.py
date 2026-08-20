from __future__ import annotations

import numpy as np

from .catalog import Catalog

from sc_cap.planner import (InfeasiblePlanError, PlannerConfig, PlannerState,
                            SCCAPPlanner, load_planner_config)
from sc_cap.strategies import Strategy


VALID_STRATEGIES = {item.value for item in Strategy}
PlannerSettings = PlannerConfig


class RegulationPlanner:
    """Web response adapter around the canonical SC-CAP core."""

    def __init__(self, catalog: Catalog, settings: PlannerSettings) -> None:
        self.catalog = catalog
        self.settings = settings
        self.core = SCCAPPlanner(catalog, settings)

    def recommend(self, *, current_va: list[float], initial_text_va: list[float],
                  strategy: str, history: list[str]) -> dict:
        if not history:
            if not np.allclose(current_va, initial_text_va, atol=1.0e-8):
                raise ValueError("c1 must equal q0; first planning is Text-VA only")
            state = PlannerState.initial(initial_text_va)
        else:
            state = PlannerState.after_feedback(initial_text_va, current_va, history)
        result = self.core.recommend(state, strategy)
        index = self.core.index_for_song(str(result["song_id"]))
        return {**result, "track": self.catalog.public_track(index)}


__all__ = [
    "InfeasiblePlanError",
    "PlannerSettings",
    "RegulationPlanner",
    "VALID_STRATEGIES",
    "load_planner_config",
]
