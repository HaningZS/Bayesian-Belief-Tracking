#!/usr/bin/env python
"""Export SBBT prefix beliefs for rollout-validation joins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from csbf.prefix_beliefs import (
    SUPPORTED_PREDICTION_METHODS,
    SUPPORTED_SPLIT_FILTERS,
    build_sbbt_prefix_prediction_rows,
)
from csbf.prefix_rollouts import read_jsonl_rows, write_jsonl_rows
from csbf.schema import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SBBT beliefs keyed by rollout prefix.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("tasks", help="Prefix rollout task JSONL.")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-mode", choices=["all_steps", "final_step", "em"], default="all_steps")
    parser.add_argument("--method", choices=sorted(SUPPORTED_PREDICTION_METHODS), default="hmm_hybrid")
    parser.add_argument("--split-filter", choices=sorted(SUPPORTED_SPLIT_FILTERS), default="test")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    tasks = read_jsonl_rows(args.tasks)
    rows, summary = build_sbbt_prefix_prediction_rows(
        records,
        tasks,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        seed=args.seed,
        calibration_mode=args.calibration_mode,
        method=args.method,
        split_filter=args.split_filter,
    )
    write_jsonl_rows(rows, args.output_jsonl)
    if args.summary_json:
        target = Path(args.summary_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
