#!/usr/bin/env python
"""Aggregate online utility diagnostics over question-level split seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from csbf.utility_table import (
    DEFAULT_FALSE_POSITIVE_RATES,
    DEFAULT_METHODS,
    DEFAULT_RECALLS,
    build_utility_split_seed_sweep,
    write_utility_split_seed_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split-seed utility diagnostics.")
    parser.add_argument("path", help="Trace JSONL path.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--calibration-mode", choices=["all_steps", "final_step", "em"], default="all_steps")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated utility methods.")
    parser.add_argument(
        "--false-positive-rates",
        default=",".join(str(value) for value in DEFAULT_FALSE_POSITIVE_RATES),
        help="Comma-separated calibration FPR constraints.",
    )
    parser.add_argument(
        "--recalls",
        default=",".join(str(value) for value in DEFAULT_RECALLS),
        help="Comma-separated calibration recall constraints.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.seed_count <= 0:
        raise SystemExit("--seed-count must be positive")

    path = Path(args.path)
    default_stem = path.with_suffix("")
    output_json = (
        Path(args.output_json)
        if args.output_json
        else default_stem.parent / f"{default_stem.name}_utility_split_seed_sweep.json"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else default_stem.parent / f"{default_stem.name}_utility_split_seed_sweep.csv"
    )
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    false_positive_rates = [
        float(value.strip()) for value in args.false_positive_rates.split(",") if value.strip()
    ]
    recalls = [float(value.strip()) for value in args.recalls.split(",") if value.strip()]
    summary = build_utility_split_seed_sweep(
        path,
        seeds=range(args.seed_start, args.seed_start + args.seed_count),
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        calibration_mode=args.calibration_mode,
        methods=methods,
        false_positive_rates=false_positive_rates,
        recalls=recalls,
    )
    written = write_utility_split_seed_sweep(summary, output_json, output_csv)
    print(
        json.dumps(
            {
                "outputs": written,
                "valid_seed_count": summary["valid_seed_count"],
                "aggregate_rows": len(summary["aggregate_rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
