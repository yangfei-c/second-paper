from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Make `python scripts/run_demo.py` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sc_cap.catalog import load_catalog
from sc_cap.models import TextVAInference
from sc_cap.planner import SCCAPPlanner, load_planner_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="I am tired after a long day")
    parser.add_argument("--strategy", default="energize")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    if args.synthetic:
        rng = np.random.default_rng(42)
        n = 2000
        from sc_cap.catalog.loader import MusicCatalog
        catalog = MusicCatalog(np.asarray([f"synthetic-{i}" for i in range(n)]),
            np.asarray(["demo"] * n), rng.uniform(-1, 1, (n, 2)).astype("float32"),
            rng.random((n, 87), dtype="float32"), rng.random((n, 40), dtype="float32"),
            rng.random((n, 56), dtype="float32"), rng.random((n, 183), dtype="float32"),
            rng.normal(size=(n, 256)).astype("float32"), rng.normal(size=(n, 256)).astype("float32"))
        text_result = {"text_pred_v": -0.2, "text_pred_a": -0.4, "fallback": True}
    else:
        catalog = load_catalog(config["catalog_path"])
        text_result = TextVAInference(config["text_checkpoint"], config["msmmr_root"],
                                      config.get("device", "cpu")).predict(args.text)
    planner = SCCAPPlanner(catalog, load_planner_config(root / args.config))
    initial = [text_result["text_pred_v"], text_result["text_pred_a"]]
    plan = planner.plan_open_loop(initial, args.strategy, args.steps)
    print(json.dumps({"input_text": args.text, "strategy": args.strategy,
                      "text": text_result, "planned_music_trajectory": [initial] + [p["music_pred_va"] for p in plan],
                      "recommendations": plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
