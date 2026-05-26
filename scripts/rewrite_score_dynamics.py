#!/usr/bin/env python
"""Rewrite concept codes from prefix score dynamics."""

from __future__ import annotations

import argparse
import json

from csbf.score_dynamics import build_score_dynamics_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite trace concept codes from score trajectory dynamics.")
    parser.add_argument("records", help="Input trace records JSONL.")
    parser.add_argument("--output-records", required=True, help="Output trace records JSONL.")
    parser.add_argument("--report", required=True, help="Output JSON report.")
    parser.add_argument("--ema-alpha", type=float, default=0.4)
    parser.add_argument("--volatility-window", type=int, default=3)
    parser.add_argument("--delta-threshold", type=float, default=0.05)
    parser.add_argument("--residual-threshold", type=float, default=0.05)
    parser.add_argument("--volatility-threshold", type=float, default=0.08)
    parser.add_argument("--recovery-threshold", type=float, default=0.15)
    args = parser.parse_args()

    summary = build_score_dynamics_records(
        args.records,
        output_path=args.output_records,
        report_path=args.report,
        ema_alpha=args.ema_alpha,
        volatility_window=args.volatility_window,
        delta_threshold=args.delta_threshold,
        residual_threshold=args.residual_threshold,
        volatility_threshold=args.volatility_threshold,
        recovery_threshold=args.recovery_threshold,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
