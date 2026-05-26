"""Offline pipeline for pre-observed trace JSONL files."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from csbf.diagnostics import summarize_trace_records
from csbf.evaluation import evaluate_hmm_prefix_diagnostics, evaluate_trace_methods
from csbf.model_selection import select_hmm_config
from csbf.schema import TraceRecord, load_jsonl
from csbf.split import split_by_question


def run_preobserved_pipeline(
    path: str | Path,
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    seed: int = 0,
    use_model_selection: bool = False,
    calibration_mode: str = "all_steps",
) -> dict[str, Any]:
    """Run evaluation on traces that already contain score/concept observations."""

    records = load_jsonl(path)
    diagnostics = summarize_trace_records(records)
    splits = split_by_question(
        records,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        seed=seed,
    )

    hmm_config = None
    if use_model_selection:
        cal_fit, cal_val = _sub_split_calibration(
            splits["calibration"], seed=seed,
        )
        if cal_val:
            hmm_config = select_hmm_config(
                cal_fit, cal_val, calibration_mode=calibration_mode,
            )

    methods = evaluate_trace_methods(
        splits["calibration"],
        splits["test"],
        hmm_config=hmm_config,
        calibration_mode=calibration_mode,
    )
    prefix_diagnostics = evaluate_hmm_prefix_diagnostics(
        splits["calibration"],
        splits["test"],
        hmm_config=hmm_config,
        calibration_mode=calibration_mode,
    )
    return {
        "num_records": len(records),
        "split_sizes": {name: len(split_records) for name, split_records in splits.items()},
        "calibration_mode": calibration_mode,
        "use_model_selection": bool(use_model_selection),
        "selected_hmm_config": _config_to_dict(hmm_config),
        "diagnostics": diagnostics,
        "methods": methods,
        "prefix_diagnostics": prefix_diagnostics,
    }


def _sub_split_calibration(
    records: list[TraceRecord],
    fit_ratio: float = 0.7,
    seed: int = 0,
) -> tuple[list[TraceRecord], list[TraceRecord]]:
    """Split calibration records into fit (70%) and validation (30%) by question."""
    by_question: dict[str, list[TraceRecord]] = defaultdict(list)
    for record in records:
        by_question[record.question_id].append(record)

    question_ids = sorted(by_question)
    random.Random(seed + 1).shuffle(question_ids)

    n = len(question_ids)
    if n < 2:
        return list(records), list(records)

    fit_count = max(1, int(n * fit_ratio))
    if fit_count >= n:
        fit_count = n - 1

    fit_ids = set(question_ids[:fit_count])
    fit_records: list[TraceRecord] = []
    val_records: list[TraceRecord] = []
    for qid in question_ids:
        bucket = fit_records if qid in fit_ids else val_records
        bucket.extend(by_question[qid])

    return fit_records, val_records


def _config_to_dict(config: Any) -> dict[str, float] | None:
    if config is None:
        return None
    return {
        "p_error": config.p_error,
        "p_recover": config.p_recover,
        "initial_on_track": config.initial_on_track,
    }
