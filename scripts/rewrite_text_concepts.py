#!/usr/bin/env python
"""Rewrite concept codes with deterministic text-marker concepts."""

from __future__ import annotations

import argparse
import json

from csbf.text_concept_records import build_text_concept_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite trace concept codes from observation text.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("--mode", choices=["text_pattern", "self_verification"], default="text_pattern")
    parser.add_argument("--output-records", required=True, help="Output trace records JSONL.")
    parser.add_argument("--report", required=True, help="Output JSON report.")
    args = parser.parse_args()

    summary = build_text_concept_records(
        args.records,
        output_path=args.output_records,
        report_path=args.report,
        mode=args.mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
