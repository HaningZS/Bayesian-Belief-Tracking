#!/usr/bin/env python
"""Run leakage-clean split-seed diagnostics for hidden-cluster concepts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from csbf.hidden_concepts import (
    HiddenClusterConfig,
    run_hidden_cluster_split_seed_sweep,
    write_hidden_cluster_split_seed_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit hidden-vector clusters for each split seed.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("features", help="Input hidden features JSONL.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--use-model-selection", action="store_true")
    parser.add_argument("--calibration-mode", choices=["all_steps", "final_step", "em"], default="all_steps")
    parser.add_argument("--cluster-count", type=int, default=16)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--feature-field", default="hidden_last_token")
    parser.add_argument("--include-entropy-feature", action="store_true")
    parser.add_argument(
        "--combine-text-concept",
        choices=["none", "text_pattern", "self_verification"],
        default="none",
        help=(
            "Optionally join each hidden-cluster id with a deterministic text "
            "concept computed from the same prefix text."
        ),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.seed_count <= 0:
        raise SystemExit("--seed-count must be positive")

    records_path = Path(args.records)
    default_stem = records_path.with_suffix("")
    output_json = (
        Path(args.output_json)
        if args.output_json
        else default_stem.parent / f"{default_stem.name}_hidden_cluster_retrained_split_seed.json"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else default_stem.parent / f"{default_stem.name}_hidden_cluster_retrained_split_seed.csv"
    )
    summary = run_hidden_cluster_split_seed_sweep(
        records_path,
        args.features,
        seeds=range(args.seed_start, args.seed_start + args.seed_count),
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        config=HiddenClusterConfig(
            cluster_count=args.cluster_count,
            projection_dim=args.projection_dim,
            iterations=args.iterations,
            feature_field=args.feature_field,
            include_entropy_feature=args.include_entropy_feature,
            text_concept_mode=None if args.combine_text_concept == "none" else args.combine_text_concept,
        ),
        use_model_selection=args.use_model_selection,
        calibration_mode=args.calibration_mode,
    )
    written = write_hidden_cluster_split_seed_sweep(summary, output_json, output_csv)
    print(json.dumps({"outputs": written, "aggregate": summary["aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
