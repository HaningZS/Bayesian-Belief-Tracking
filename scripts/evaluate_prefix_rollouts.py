#!/usr/bin/env python
"""Evaluate rollout-based prefix value validation outputs."""

from __future__ import annotations

import argparse

from csbf.prefix_rollouts import aggregate_prefix_rollouts, read_jsonl_rows, write_prefix_rollout_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate prefix rollout continuation correctness.")
    parser.add_argument("tasks", help="Prefix rollout task JSONL.")
    parser.add_argument("results", help="Prefix rollout continuation result JSONL.")
    parser.add_argument("--score-bins", type=int, default=10)
    parser.add_argument(
        "--prediction-jsonl",
        help="Optional JSONL with prefix predictions keyed by trace_id and step_index.",
    )
    parser.add_argument(
        "--prediction-field",
        default="sbbt_belief",
        help="Field to read from --prediction-jsonl.",
    )
    parser.add_argument(
        "--prediction-name",
        default="belief",
        help="Metric prefix/name to use for the joined prediction.",
    )
    parser.add_argument(
        "--require-prediction",
        action="store_true",
        help="Restrict aggregate metrics to prefix groups covered by the joined prediction rows.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    tasks = read_jsonl_rows(args.tasks)
    results = read_jsonl_rows(args.results)
    prediction_rows = read_jsonl_rows(args.prediction_jsonl) if args.prediction_jsonl else None
    prediction_specs = {args.prediction_name: args.prediction_field} if prediction_rows is not None else None
    summary = aggregate_prefix_rollouts(
        tasks,
        results,
        score_bins=args.score_bins,
        prediction_rows=prediction_rows,
        prediction_specs=prediction_specs,
        require_predictions=args.require_prediction,
    )
    write_prefix_rollout_summary(
        summary,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_markdown=args.output_markdown,
    )


if __name__ == "__main__":
    main()
