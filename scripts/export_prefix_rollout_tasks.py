#!/usr/bin/env python
"""Export prefix continuation tasks for rollout-based prefix value validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from csbf.prefix_rollouts import (
    DEFAULT_PREFIX_FRACTIONS,
    build_prefix_rollout_tasks,
    summarize_prefix_rollout_tasks,
    write_jsonl_rows,
)
from csbf.schema import load_jsonl
from csbf.split import split_by_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prefix rollout continuation tasks from trace records.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Output task JSONL.")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON path.")
    parser.add_argument(
        "--prefix-fraction",
        dest="prefix_fractions",
        action="append",
        type=float,
        default=None,
        help="Requested prefix fraction. Repeat for multiple fractions. Defaults to 0.25/0.5/0.75.",
    )
    parser.add_argument("--rollouts-per-prefix", type=int, default=4)
    parser.add_argument("--max-prefix-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-filter",
        choices=["all", "train", "calibration", "test"],
        default="all",
        help="Optionally export tasks only from one question-level split.",
    )
    parser.add_argument("--prompt-dataset-type", choices=["auto", "gsm8k", "math"], default="auto")
    parser.add_argument("--prompt-format", choices=["raw", "chat_auto", "chat_no_think"], default=None)
    parser.add_argument("--prompt-style", choices=["default", "concise", "aime_benchmark"], default=None)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    if args.split_filter != "all":
        splits = split_by_question(
            records,
            train_ratio=args.train_ratio,
            calibration_ratio=args.calibration_ratio,
            seed=args.seed,
        )
        records = splits[args.split_filter]
    tasks = build_prefix_rollout_tasks(
        records,
        prefix_fractions=args.prefix_fractions or DEFAULT_PREFIX_FRACTIONS,
        rollouts_per_prefix=args.rollouts_per_prefix,
        max_prefix_groups=args.max_prefix_groups,
        seed=args.seed,
        prompt_dataset_type=args.prompt_dataset_type,
        prompt_format=args.prompt_format,
        prompt_style=args.prompt_style,
    )
    write_jsonl_rows(tasks, args.output_jsonl)
    summary = summarize_prefix_rollout_tasks(tasks)
    summary.update(
        {
            "split_filter": args.split_filter,
            "train_ratio": args.train_ratio,
            "calibration_ratio": args.calibration_ratio,
            "seed": args.seed,
        }
    )
    if args.summary_json:
        target = Path(args.summary_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
