#!/usr/bin/env python
"""Run the offline pre-observed trace pipeline."""

from __future__ import annotations

import argparse
import json

from csbf.pipeline import run_preobserved_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pre-observed trace JSONL files.")
    parser.add_argument("path", help="Path to trace JSONL.")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--use-model-selection", action="store_true")
    parser.add_argument("--calibration-mode", choices=["all_steps", "final_step", "em"], default="all_steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_preobserved_pipeline(
        args.path,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        seed=args.seed,
        use_model_selection=args.use_model_selection,
        calibration_mode=args.calibration_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
