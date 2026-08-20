from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sc_cap.catalog import load_catalog
from sc_cap.evaluation import METHODS, plan_baseline
from sc_cap.models import TextVAInference
from sc_cap.planner import load_planner_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="energize")
    parser.add_argument("--text", default="I am tired after a long day")
    parser.add_argument("--method", choices=METHODS, default="open_loop_sc_cap")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    catalog = load_catalog(config["catalog_path"])
    text = TextVAInference(config["text_checkpoint"], config["msmmr_root"], config.get("device", "cpu")).predict(args.text)
    initial = [text["text_pred_v"], text["text_pred_a"]]
    plan = plan_baseline(catalog, initial, args.strategy, args.method,
                         steps=config["planner"]["total_steps"],
                         config=load_planner_config(root / args.config))
    print(json.dumps({"strategy": args.strategy, "method": args.method,
                      "text": text, "planned_music_trajectory": [initial] + [p["music_pred_va"] for p in plan],
                      "recommendations": plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
