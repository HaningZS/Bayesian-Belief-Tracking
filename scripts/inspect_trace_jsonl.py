#!/usr/bin/env python
"""Inspect trace JSONL quality before running experiments."""

from __future__ import annotations

import argparse
import json

from csbf.diagnostics import summarize_trace_records
from csbf.schema import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect trace JSONL structure and quality.")
    parser.add_argument("path", help="Path to trace JSONL.")
    args = parser.parse_args()

    summary = summarize_trace_records(load_jsonl(args.path))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
