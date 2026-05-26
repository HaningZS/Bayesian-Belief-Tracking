#!/usr/bin/env python
"""Rewrite trace scores with latent-trajectory hidden-state metrics."""

from __future__ import annotations

import argparse
import json

from csbf.latent_trajectory import LatentTrajectoryConfig, build_latent_trajectory_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute latent-trajectory scores and rewrite observations.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("features", help="Input hidden features JSONL.")
    parser.add_argument("--output-records", required=True, help="Output trace records JSONL with LT scores.")
    parser.add_argument("--report", required=True, help="Output JSON report.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--feature-field", default="hidden_last_token")
    parser.add_argument(
        "--metric",
        choices=["net_change", "cumulative_change", "aligned_change", "composite", "net", "cumulative", "aligned"],
        default="composite",
    )
    parser.add_argument("--normalize-vectors", action="store_true")
    args = parser.parse_args()

    summary = build_latent_trajectory_records(
        args.records,
        args.features,
        output_path=args.output_records,
        report_path=args.report,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        seed=args.seed,
        config=LatentTrajectoryConfig(
            projection_dim=args.projection_dim,
            metric=args.metric,
            normalize_vectors=args.normalize_vectors,
            feature_field=args.feature_field,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
