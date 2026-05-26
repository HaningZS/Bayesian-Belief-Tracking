#!/usr/bin/env python
"""Evaluate latent-trajectory scores over multiple question-level split seeds."""

from __future__ import annotations

import argparse
import json

from csbf.latent_trajectory import (
    LatentTrajectoryConfig,
    run_latent_trajectory_split_seed_sweep,
    write_latent_trajectory_split_seed_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run latent-trajectory split-seed robustness diagnostics.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("features", help="Input hidden features JSONL.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--use-model-selection", action="store_true")
    parser.add_argument("--calibration-mode", choices=["all_steps", "final_step", "em"], default="all_steps")
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--feature-field", default="hidden_last_token")
    parser.add_argument(
        "--metric",
        choices=["net_change", "cumulative_change", "aligned_change", "composite", "net", "cumulative", "aligned"],
        default="composite",
    )
    parser.add_argument("--normalize-vectors", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.seed_count <= 0:
        raise SystemExit("--seed-count must be positive")

    summary = run_latent_trajectory_split_seed_sweep(
        args.records,
        args.features,
        seeds=range(args.seed_start, args.seed_start + args.seed_count),
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        config=LatentTrajectoryConfig(
            projection_dim=args.projection_dim,
            metric=args.metric,
            normalize_vectors=args.normalize_vectors,
            feature_field=args.feature_field,
        ),
        use_model_selection=args.use_model_selection,
        calibration_mode=args.calibration_mode,
    )
    written = write_latent_trajectory_split_seed_sweep(summary, args.output_json, args.output_csv)
    print(json.dumps({"outputs": written, "aggregate": summary["aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
