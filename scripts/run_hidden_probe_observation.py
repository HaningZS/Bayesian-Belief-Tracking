#!/usr/bin/env python
"""Rewrite trace scores with a question-split hidden-state probe."""

from __future__ import annotations

import argparse
import json

from csbf.hidden_probe import HiddenProbeConfig, build_hidden_probe_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hidden-state probe and rewrite observation scores.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("features", help="Input hidden features JSONL.")
    parser.add_argument("--output-records", required=True, help="Output trace records JSONL with probe scores.")
    parser.add_argument("--report", required=True, help="Output JSON report.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--calibration-epochs", type=int, default=120)
    parser.add_argument(
        "--feature-field",
        action="append",
        default=None,
        help=(
            "Hidden feature row field to use as probe input. Repeat to concatenate fields. "
            "Defaults to hidden_last_token."
        ),
    )
    parser.add_argument("--include-entropy-feature", action="store_true")
    args = parser.parse_args()
    feature_fields = tuple(args.feature_field or ["hidden_last_token"])

    summary = build_hidden_probe_records(
        args.records,
        args.features,
        output_path=args.output_records,
        report_path=args.report,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        seed=args.seed,
        config=HiddenProbeConfig(
            projection_dim=args.projection_dim,
            feature_fields=feature_fields,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            calibration_epochs=args.calibration_epochs,
            include_entropy_feature=args.include_entropy_feature,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
