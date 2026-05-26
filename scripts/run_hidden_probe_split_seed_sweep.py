#!/usr/bin/env python
"""Run leakage-clean split-seed diagnostics for hidden-probe observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from csbf.hidden_probe import (
    HiddenProbeConfig,
    run_hidden_probe_split_seed_sweep,
    write_hidden_probe_split_seed_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain a hidden-state probe for each split seed.")
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
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.seed_count <= 0:
        raise SystemExit("--seed-count must be positive")
    feature_fields = tuple(args.feature_field or ["hidden_last_token"])

    records_path = Path(args.records)
    default_stem = records_path.with_suffix("")
    output_json = (
        Path(args.output_json)
        if args.output_json
        else default_stem.parent / f"{default_stem.name}_hidden_probe_retrained_split_seed.json"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else default_stem.parent / f"{default_stem.name}_hidden_probe_retrained_split_seed.csv"
    )
    summary = run_hidden_probe_split_seed_sweep(
        records_path,
        args.features,
        seeds=range(args.seed_start, args.seed_start + args.seed_count),
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        config=HiddenProbeConfig(
            projection_dim=args.projection_dim,
            feature_fields=feature_fields,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            calibration_epochs=args.calibration_epochs,
            include_entropy_feature=args.include_entropy_feature,
        ),
        use_model_selection=args.use_model_selection,
        calibration_mode=args.calibration_mode,
    )
    written = write_hidden_probe_split_seed_sweep(summary, output_json, output_csv)
    print(json.dumps({"outputs": written, "aggregate": summary["aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
