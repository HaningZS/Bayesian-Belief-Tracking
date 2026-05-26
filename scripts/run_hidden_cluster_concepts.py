#!/usr/bin/env python
"""Rewrite concept codes using train-split hidden-state clusters."""

from __future__ import annotations

import argparse
import json

from csbf.hidden_concepts import HiddenClusterConfig, build_hidden_cluster_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit hidden-vector clusters and rewrite concept codes.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("features", help="Input hidden features JSONL.")
    parser.add_argument("--output-records", required=True, help="Output trace records JSONL with hidden cluster codes.")
    parser.add_argument("--report", required=True, help="Output JSON report.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--cluster-count", type=int, default=16)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--feature-field", default="hidden_last_token")
    parser.add_argument("--include-entropy-feature", action="store_true")
    args = parser.parse_args()

    summary = build_hidden_cluster_records(
        args.records,
        args.features,
        output_path=args.output_records,
        report_path=args.report,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        seed=args.seed,
        config=HiddenClusterConfig(
            cluster_count=args.cluster_count,
            projection_dim=args.projection_dim,
            iterations=args.iterations,
            feature_field=args.feature_field,
            include_entropy_feature=args.include_entropy_feature,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
