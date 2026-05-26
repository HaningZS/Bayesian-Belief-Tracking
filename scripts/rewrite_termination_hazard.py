#!/usr/bin/env python
"""Rewrite scores/concepts with prefix-safe termination-hazard observations."""

from __future__ import annotations

import argparse
import json

from csbf.termination_hazard import DEFAULT_HORIZON_OBSERVATIONS, build_termination_hazard_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite trace observations with elapsed-reasoning hazard signals.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("--horizon-observations", type=int, default=DEFAULT_HORIZON_OBSERVATIONS)
    parser.add_argument("--output-records", required=True, help="Output trace records JSONL.")
    parser.add_argument("--report", required=True, help="Output JSON report.")
    args = parser.parse_args()

    summary = build_termination_hazard_records(
        args.records,
        output_path=args.output_records,
        report_path=args.report,
        horizon_observations=args.horizon_observations,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
