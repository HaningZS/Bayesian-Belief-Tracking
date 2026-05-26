#!/usr/bin/env python
"""Export online decision-utility diagnostics from trace records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from csbf.utility_table import (
    DEFAULT_FALSE_POSITIVE_RATES,
    DEFAULT_METHODS,
    DEFAULT_RECALLS,
    build_utility_table,
    write_utility_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an online utility table from trace records.")
    parser.add_argument("path", help="Trace JSONL path.")
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    path = Path(args.path)
    stem = path.with_suffix("")
    output_json = Path(args.output_json) if args.output_json else stem.parent / f"{stem.name}_utility_table.json"
    output_csv = Path(args.output_csv) if args.output_csv else stem.parent / f"{stem.name}_utility_table.csv"
    output_markdown = (
        Path(args.output_markdown)
        if args.output_markdown
        else stem.parent / f"{stem.name}_utility_table.md"
    )
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    false_positive_rates = [
        float(value.strip()) for value in args.false_positive_rates.split(",") if value.strip()
    ]
    recalls = [float(value.strip()) for value in args.recalls.split(",") if value.strip()]

    summary = build_utility_table(
        path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        calibration_mode=args.calibration_mode,
        methods=methods,
        false_positive_rates=false_positive_rates,
        recalls=recalls,
    )
    written = write_utility_table(
        summary,
        output_json=output_json,
        output_csv=output_csv,
        output_markdown=output_markdown,
    )
    print(json.dumps({"outputs": written, "rows": len(summary["rows"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
