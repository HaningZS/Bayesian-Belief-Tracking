"""Small model-selection helpers for calibration-time tuning."""

from __future__ import annotations

from itertools import product

from csbf.evaluation import evaluate_trace_methods
from csbf.filtering import HMMConfig
from csbf.schema import TraceRecord


def select_hmm_config(
    calibration_records: list[TraceRecord],
    validation_records: list[TraceRecord],
    p_error_grid: list[float] | None = None,
    p_recover_grid: list[float] | None = None,
    initial_on_track_grid: list[float] | None = None,
    calibration_mode: str = "all_steps",
) -> HMMConfig:
    """Select HMM transitions by lowest hybrid-filter Brier score.

    *calibration_records* are used for fitting likelihoods.
    *validation_records* are used for evaluating each candidate config.
    Callers must ensure these two sets are disjoint to avoid data leakage.
    """

    p_error_values = p_error_grid or [0.02, 0.05, 0.10, 0.20]
    p_recover_values = p_recover_grid or [0.05, 0.10, 0.20, 0.35]
    base_rate = _base_rate(calibration_records)
    initial_values = initial_on_track_grid or _unique([0.3, 0.5, 0.7, base_rate])
    best_config = HMMConfig()
    best_brier = float("inf")
    for p_error, p_recover, initial_on_track in product(
        p_error_values, p_recover_values, initial_values,
    ):
        config = HMMConfig(
            p_error=p_error,
            p_recover=p_recover,
            initial_on_track=initial_on_track,
        )
        report = evaluate_trace_methods(
            calibration_records,
            validation_records,
            hmm_config=config,
            calibration_mode=calibration_mode,
        )
        brier = float(report["hmm_hybrid"]["metrics"]["brier"])
        if brier < best_brier:
            best_config = config
            best_brier = brier
    return best_config


def _base_rate(records: list[TraceRecord]) -> float:
    if not records:
        raise ValueError("calibration_records must not be empty")
    return sum(1 for record in records if record.correct) / len(records)


def _unique(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
