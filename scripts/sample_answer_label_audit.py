#!/usr/bin/env python
"""Sample generated traces for manual answer-label audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from csbf.answer_audit import build_answer_audit_rows, summarize_answer_audit_rows
from csbf.schema import load_jsonl


FIELDS = [
    "audit_id",
    "question_id",
    "trace_id",
    "categories",
    "correct",
    "final_answer",
    "gold_answer",
    "observation_count",
    "question_excerpt",
    "trace_excerpt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample traces likely to expose answer-label issues.")
    parser.add_argument("records_jsonl", help="Trace records JSONL.")
    parser.add_argument("--max-per-category", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--excerpt-chars", type=int, default=500)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    records = load_jsonl(args.records_jsonl)
    rows = build_answer_audit_rows(
        records,
        max_per_category=args.max_per_category,
        seed=args.seed,
        excerpt_chars=args.excerpt_chars,
    )
    summary = summarize_answer_audit_rows(rows, input_count=len(records))

    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    _write_jsonl([row.to_dict() for row in rows], output_jsonl)
    _write_csv([row.to_dict() for row in rows], output_csv)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit_rows": len(rows), "summary": str(summary_json)}, indent=2, sort_keys=True))


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in FIELDS})


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


if __name__ == "__main__":
    main()
