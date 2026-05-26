#!/usr/bin/env python
"""Evaluate prefix observations against external process labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from csbf.process_validation import build_process_prefix_examples, evaluate_process_labels, load_process_labels
from csbf.schema import load_jsonl


PREFIX_FIELDS = [
    "trace_id",
    "question_id",
    "step_index",
    "error_started",
    "predicted_error_probability",
    "observation_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prefix scores against process/error-onset labels.")
    parser.add_argument("records_jsonl", help="Trace records JSONL.")
    parser.add_argument("process_labels_jsonl", help="JSONL with trace_id and first_error_step/null.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-prefix-csv", required=True)
    args = parser.parse_args()

    records = load_jsonl(args.records_jsonl)
    labels = load_process_labels(args.process_labels_jsonl)
    report = evaluate_process_labels(records, labels)
    examples = build_process_prefix_examples(records, labels)

    output_json = Path(args.output_json)
    output_csv = Path(args.output_prefix_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_prefix_csv([example.to_dict() for example in examples], output_csv)
    print(json.dumps({"prefix_count": len(examples), "output_json": str(output_json)}, indent=2, sort_keys=True))


def _write_prefix_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREFIX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in PREFIX_FIELDS})


if __name__ == "__main__":
    main()
