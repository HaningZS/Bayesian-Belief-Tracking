#!/usr/bin/env python
"""Run split-seed diagnostics for probe-logit trajectory observations."""

from __future__ import annotations

import argparse
import json

from csbf.probe_trajectory import (
    ProbeTrajectoryConfig,
    run_probe_trajectory_split_seed_sweep,
    write_probe_trajectory_split_seed_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probe-trajectory split-seed robustness diagnostics.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("features", help="Input hidden features JSONL.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--use-model-selection", action="store_true")
    parser.add_argument("--calibration-mode", choices=["all_steps", "final_step", "em"], default="all_steps")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--calibration-epochs", type=int, default=120)
    parser.add_argument("--include-entropy-feature", action="store_true")
    parser.add_argument(
        "--metric",
        choices=[
            "probe_logit",
            "logit_delta",
            "cumulative_logit_change",
            "aligned_logit_change",
            "composite",
            "logit",
            "delta",
            "cumulative",
            "aligned",
        ],
        default="logit_delta",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.seed_count <= 0:
        raise SystemExit("--seed-count must be positive")

    summary = run_probe_trajectory_split_seed_sweep(
        args.records,
        args.features,
        seeds=range(args.seed_start, args.seed_start + args.seed_count),
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        config=ProbeTrajectoryConfig(
            projection_dim=args.projection_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            calibration_epochs=args.calibration_epochs,
            include_entropy_feature=args.include_entropy_feature,
            metric=args.metric,
        ),
        use_model_selection=args.use_model_selection,
        calibration_mode=args.calibration_mode,
    )
    written = write_probe_trajectory_split_seed_sweep(summary, args.output_json, args.output_csv)
    print(json.dumps({"outputs": written, "aggregate": summary["aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
