"""Split-seed robustness diagnostics for pre-observed trace records."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from csbf.evaluation import evaluate_trace_methods
from csbf.experiment_diagnostics import BASELINE_METHODS, ONLINE_HMM_METHODS
from csbf.model_selection import select_hmm_config
from csbf.pipeline import _sub_split_calibration
from csbf.schema import TraceRecord, load_jsonl
from csbf.split import split_by_question


def run_split_seed_sweep(
    records_or_path: Sequence[TraceRecord] | str | Path,
    seeds: Iterable[int],
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    use_model_selection: bool = False,
    calibration_mode: str = "all_steps",
) -> dict[str, Any]:
    """Evaluate the same records over multiple question-level split seeds."""

    records = _load_records(records_or_path)
    seed_rows = [
        _evaluate_seed(
            records,
            seed=int(seed),
            train_ratio=train_ratio,
            calibration_ratio=calibration_ratio,
            use_model_selection=use_model_selection,
            calibration_mode=calibration_mode,
        )
        for seed in seeds
    ]
    return {
        "num_records": len(records),
        "num_questions": len({record.question_id for record in records}),
        "seed_count": len(seed_rows),
        "train_ratio": train_ratio,
        "calibration_ratio": calibration_ratio,
        "use_model_selection": bool(use_model_selection),
        "calibration_mode": calibration_mode,
        "valid_auroc_seed_count": sum(row["hmm_vs_baseline_auroc_gap"] is not None for row in seed_rows),
        "aggregate": _aggregate(seed_rows),
        "seeds": seed_rows,
    }


def write_split_seed_sweep(
    summary: dict[str, Any],
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, str]:
    """Write split-seed sweep JSON and optional CSV outputs."""

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = {"json": str(json_path)}
    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(summary["seeds"], csv_path)
        written["csv"] = str(csv_path)
    return written


def _load_records(records_or_path: Sequence[TraceRecord] | str | Path) -> list[TraceRecord]:
    if isinstance(records_or_path, str | Path):
        return load_jsonl(Path(records_or_path))
    return list(records_or_path)


def _evaluate_seed(
    records: list[TraceRecord],
    seed: int,
    train_ratio: float,
    calibration_ratio: float,
    use_model_selection: bool,
    calibration_mode: str,
) -> dict[str, Any]:
    splits = split_by_question(
        records,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        seed=seed,
    )
    hmm_config = None
    if use_model_selection:
        cal_fit, cal_val = _sub_split_calibration(splits["calibration"], seed=seed)
        if cal_val:
            hmm_config = select_hmm_config(cal_fit, cal_val, calibration_mode=calibration_mode)

    row: dict[str, Any] = {
        "seed": seed,
        "split": {name: _split_counts(split_records) for name, split_records in splits.items()},
        "selected_hmm_config": _config_to_dict(hmm_config),
    }
    try:
        methods = evaluate_trace_methods(
            splits["calibration"],
            splits["test"],
            hmm_config=hmm_config,
            calibration_mode=calibration_mode,
        )
    except ValueError as error:
        row["error"] = str(error)
        row.update(_empty_method_summary())
        return row

    best_baseline = _best_method_auc(methods, BASELINE_METHODS)
    best_hmm = _best_method_auc(methods, ONLINE_HMM_METHODS)
    baseline_auc = best_baseline["auroc"] if best_baseline else None
    hmm_auc = best_hmm["auroc"] if best_hmm else None
    row.update(
        {
            "best_baseline_method": best_baseline["method"] if best_baseline else None,
            "best_baseline_auroc": baseline_auc,
            "best_online_hmm_method": best_hmm["method"] if best_hmm else None,
            "best_online_hmm_auroc": hmm_auc,
            "hmm_vs_baseline_auroc_gap": _subtract_or_none(hmm_auc, baseline_auc),
            "method_metrics": {
                method: {
                    "auroc": metrics["metrics"].get("auroc"),
                    "auprc": metrics["metrics"].get("auprc"),
                    "brier": metrics["metrics"].get("brier"),
                    "ece": metrics["metrics"].get("ece"),
                }
                for method, metrics in methods.items()
            },
        }
    )
    row.update(_fixed_method_gaps(row["method_metrics"]))
    return row


def _split_counts(records: list[TraceRecord]) -> dict[str, int]:
    return {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "correct": sum(1 for record in records if record.correct),
        "incorrect": sum(1 for record in records if not record.correct),
    }


def _best_method_auc(methods: dict[str, dict[str, Any]], candidates: set[str]) -> dict[str, Any] | None:
    rows = [
        {"method": method, "auroc": float(payload["metrics"]["auroc"])}
        for method, payload in methods.items()
        if method in candidates and payload["metrics"].get("auroc") is not None
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: row["auroc"])


def _aggregate(seed_rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    gaps = _present_floats(row.get("hmm_vs_baseline_auroc_gap") for row in seed_rows)
    baseline_aurocs = _present_floats(row.get("best_baseline_auroc") for row in seed_rows)
    hmm_aurocs = _present_floats(row.get("best_online_hmm_auroc") for row in seed_rows)
    hmm_score_briers = _present_floats(
        _nested(row, "method_metrics.hmm_score.brier") for row in seed_rows
    )
    hmm_score_eces = _present_floats(
        _nested(row, "method_metrics.hmm_score.ece") for row in seed_rows
    )
    hmm_hybrid_auprcs = _present_floats(
        _nested(row, "method_metrics.hmm_hybrid.auprc") for row in seed_rows
    )
    calibrated_last_step_briers = _present_floats(
        _nested(row, "method_metrics.calibrated_last_step.brier") for row in seed_rows
    )
    calibrated_last_step_eces = _present_floats(
        _nested(row, "method_metrics.calibrated_last_step.ece") for row in seed_rows
    )
    ema_briers = _present_floats(_nested(row, "method_metrics.ema.brier") for row in seed_rows)
    ema_eces = _present_floats(_nested(row, "method_metrics.ema.ece") for row in seed_rows)
    ema_auprcs = _present_floats(_nested(row, "method_metrics.ema.auprc") for row in seed_rows)
    temporal_metric_auprcs = _present_floats(
        _nested(row, "method_metrics.temporal_metric.auprc") for row in seed_rows
    )
    hmm_score_brier_minus_ema = _present_floats(
        row.get("hmm_score_brier_minus_ema") for row in seed_rows
    )
    hmm_score_brier_minus_calibrated_last_step = _present_floats(
        row.get("hmm_score_brier_minus_calibrated_last_step") for row in seed_rows
    )
    hmm_score_ece_minus_ema = _present_floats(
        row.get("hmm_score_ece_minus_ema") for row in seed_rows
    )
    hmm_score_ece_minus_calibrated_last_step = _present_floats(
        row.get("hmm_score_ece_minus_calibrated_last_step") for row in seed_rows
    )
    hmm_hybrid_auroc_minus_ema = _present_floats(
        row.get("hmm_hybrid_auroc_minus_ema") for row in seed_rows
    )
    hmm_hybrid_auroc_minus_temporal_metric = _present_floats(
        row.get("hmm_hybrid_auroc_minus_temporal_metric") for row in seed_rows
    )
    hmm_hybrid_auprc_minus_ema = _present_floats(
        row.get("hmm_hybrid_auprc_minus_ema") for row in seed_rows
    )
    hmm_hybrid_auprc_minus_temporal_metric = _present_floats(
        row.get("hmm_hybrid_auprc_minus_temporal_metric") for row in seed_rows
    )
    return {
        "mean_hmm_vs_baseline_auroc_gap": _mean_or_none(gaps),
        "median_hmm_vs_baseline_auroc_gap": _median_or_none(gaps),
        "positive_hmm_gap_fraction": _fraction_positive(gaps),
        "positive_hmm_vs_baseline_auroc_gap_fraction": _fraction_positive(gaps),
        "mean_best_baseline_auroc": _mean_or_none(baseline_aurocs),
        "mean_best_online_hmm_auroc": _mean_or_none(hmm_aurocs),
        "mean_hmm_score_brier": _mean_or_none(hmm_score_briers),
        "mean_hmm_score_ece": _mean_or_none(hmm_score_eces),
        "mean_hmm_hybrid_auprc": _mean_or_none(hmm_hybrid_auprcs),
        "mean_calibrated_last_step_brier": _mean_or_none(calibrated_last_step_briers),
        "mean_calibrated_last_step_ece": _mean_or_none(calibrated_last_step_eces),
        "mean_ema_brier": _mean_or_none(ema_briers),
        "mean_ema_ece": _mean_or_none(ema_eces),
        "mean_ema_auprc": _mean_or_none(ema_auprcs),
        "mean_temporal_metric_auprc": _mean_or_none(temporal_metric_auprcs),
        "hmm_score_brier_minus_ema": _subtract_or_none(
            _mean_or_none(hmm_score_briers),
            _mean_or_none(ema_briers),
        ),
        "median_hmm_score_brier_minus_ema": _median_or_none(hmm_score_brier_minus_ema),
        "negative_hmm_score_brier_minus_ema_fraction": _fraction_negative(hmm_score_brier_minus_ema),
        "hmm_score_brier_minus_calibrated_last_step": _subtract_or_none(
            _mean_or_none(hmm_score_briers),
            _mean_or_none(calibrated_last_step_briers),
        ),
        "median_hmm_score_brier_minus_calibrated_last_step": _median_or_none(
            hmm_score_brier_minus_calibrated_last_step
        ),
        "negative_hmm_score_brier_minus_calibrated_last_step_fraction": _fraction_negative(
            hmm_score_brier_minus_calibrated_last_step
        ),
        "hmm_score_ece_minus_ema": _subtract_or_none(
            _mean_or_none(hmm_score_eces),
            _mean_or_none(ema_eces),
        ),
        "median_hmm_score_ece_minus_ema": _median_or_none(hmm_score_ece_minus_ema),
        "negative_hmm_score_ece_minus_ema_fraction": _fraction_negative(hmm_score_ece_minus_ema),
        "hmm_score_ece_minus_calibrated_last_step": _subtract_or_none(
            _mean_or_none(hmm_score_eces),
            _mean_or_none(calibrated_last_step_eces),
        ),
        "median_hmm_score_ece_minus_calibrated_last_step": _median_or_none(
            hmm_score_ece_minus_calibrated_last_step
        ),
        "negative_hmm_score_ece_minus_calibrated_last_step_fraction": _fraction_negative(
            hmm_score_ece_minus_calibrated_last_step
        ),
        "mean_hmm_hybrid_auroc_minus_ema": _mean_or_none(hmm_hybrid_auroc_minus_ema),
        "median_hmm_hybrid_auroc_minus_ema": _median_or_none(hmm_hybrid_auroc_minus_ema),
        "positive_hmm_hybrid_auroc_minus_ema_fraction": _fraction_positive(hmm_hybrid_auroc_minus_ema),
        "mean_hmm_hybrid_auroc_minus_temporal_metric": _mean_or_none(
            hmm_hybrid_auroc_minus_temporal_metric
        ),
        "median_hmm_hybrid_auroc_minus_temporal_metric": _median_or_none(
            hmm_hybrid_auroc_minus_temporal_metric
        ),
        "positive_hmm_hybrid_auroc_minus_temporal_metric_fraction": _fraction_positive(
            hmm_hybrid_auroc_minus_temporal_metric
        ),
        "hmm_hybrid_auprc_minus_ema": _subtract_or_none(
            _mean_or_none(hmm_hybrid_auprcs),
            _mean_or_none(ema_auprcs),
        ),
        "mean_hmm_hybrid_auprc_minus_ema": _mean_or_none(hmm_hybrid_auprc_minus_ema),
        "median_hmm_hybrid_auprc_minus_ema": _median_or_none(hmm_hybrid_auprc_minus_ema),
        "positive_hmm_hybrid_auprc_minus_ema_fraction": _fraction_positive(hmm_hybrid_auprc_minus_ema),
        "hmm_hybrid_auprc_minus_temporal_metric": _subtract_or_none(
            _mean_or_none(hmm_hybrid_auprcs),
            _mean_or_none(temporal_metric_auprcs),
        ),
        "mean_hmm_hybrid_auprc_minus_temporal_metric": _mean_or_none(
            hmm_hybrid_auprc_minus_temporal_metric
        ),
        "median_hmm_hybrid_auprc_minus_temporal_metric": _median_or_none(
            hmm_hybrid_auprc_minus_temporal_metric
        ),
        "positive_hmm_hybrid_auprc_minus_temporal_metric_fraction": _fraction_positive(
            hmm_hybrid_auprc_minus_temporal_metric
        ),
        "errored_seed_count": sum(1 for row in seed_rows if row.get("error")),
    }


def _empty_method_summary() -> dict[str, None]:
    return {
        "best_baseline_method": None,
        "best_baseline_auroc": None,
        "best_online_hmm_method": None,
        "best_online_hmm_auroc": None,
        "hmm_vs_baseline_auroc_gap": None,
        "method_metrics": None,
    }


def _present_floats(values: Iterable[object]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(mean(values), 12)


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(median(values), 12)


def _fraction_positive(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(1 for value in values if value > 0.0) / len(values), 12)


def _fraction_negative(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(1 for value in values if value < 0.0) / len(values), 12)


def _subtract_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 12)


def _fixed_method_gaps(method_metrics: dict[str, dict[str, object]] | None) -> dict[str, float | None]:
    return {
        "hmm_score_brier_minus_ema": _metric_gap(method_metrics, "hmm_score", "ema", "brier"),
        "hmm_score_brier_minus_calibrated_last_step": _metric_gap(
            method_metrics,
            "hmm_score",
            "calibrated_last_step",
            "brier",
        ),
        "hmm_score_ece_minus_ema": _metric_gap(method_metrics, "hmm_score", "ema", "ece"),
        "hmm_score_ece_minus_calibrated_last_step": _metric_gap(
            method_metrics,
            "hmm_score",
            "calibrated_last_step",
            "ece",
        ),
        "hmm_hybrid_auroc_minus_ema": _metric_gap(method_metrics, "hmm_hybrid", "ema", "auroc"),
        "hmm_hybrid_auroc_minus_temporal_metric": _metric_gap(
            method_metrics,
            "hmm_hybrid",
            "temporal_metric",
            "auroc",
        ),
        "hmm_hybrid_auprc_minus_ema": _metric_gap(method_metrics, "hmm_hybrid", "ema", "auprc"),
        "hmm_hybrid_auprc_minus_temporal_metric": _metric_gap(
            method_metrics,
            "hmm_hybrid",
            "temporal_metric",
            "auprc",
        ),
    }


def _metric_gap(
    method_metrics: dict[str, dict[str, object]] | None,
    method: str,
    baseline: str,
    metric: str,
) -> float | None:
    if method_metrics is None:
        return None
    left = _nested(method_metrics, f"{method}.{metric}")
    right = _nested(method_metrics, f"{baseline}.{metric}")
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 12)


def _nested(row: dict[str, Any], dotted_key: str) -> Any:
    current: Any = row
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _config_to_dict(config: Any) -> dict[str, float] | None:
    if config is None:
        return None
    return {
        "p_error": config.p_error,
        "p_recover": config.p_recover,
        "initial_on_track": config.initial_on_track,
    }


def _write_csv(seed_rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "seed",
        "train_correct",
        "train_incorrect",
        "calibration_correct",
        "calibration_incorrect",
        "test_correct",
        "test_incorrect",
        "best_baseline_method",
        "best_baseline_auroc",
        "best_online_hmm_method",
        "best_online_hmm_auroc",
        "hmm_vs_baseline_auroc_gap",
        "calibrated_last_step_brier",
        "calibrated_last_step_ece",
        "hmm_score_brier",
        "hmm_score_ece",
        "ema_brier",
        "ema_ece",
        "ema_auprc",
        "hmm_hybrid_auprc",
        "hmm_score_brier_minus_ema",
        "hmm_score_brier_minus_calibrated_last_step",
        "hmm_score_ece_minus_ema",
        "hmm_score_ece_minus_calibrated_last_step",
        "hmm_hybrid_auroc_minus_ema",
        "hmm_hybrid_auroc_minus_temporal_metric",
        "hmm_hybrid_auprc_minus_ema",
        "hmm_hybrid_auprc_minus_temporal_metric",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in seed_rows:
            writer.writerow(
                {
                    "seed": row.get("seed"),
                    "train_correct": _nested(row, "split.train.correct"),
                    "train_incorrect": _nested(row, "split.train.incorrect"),
                    "calibration_correct": _nested(row, "split.calibration.correct"),
                    "calibration_incorrect": _nested(row, "split.calibration.incorrect"),
                    "test_correct": _nested(row, "split.test.correct"),
                    "test_incorrect": _nested(row, "split.test.incorrect"),
                    "best_baseline_method": row.get("best_baseline_method"),
                    "best_baseline_auroc": row.get("best_baseline_auroc"),
                    "best_online_hmm_method": row.get("best_online_hmm_method"),
                    "best_online_hmm_auroc": row.get("best_online_hmm_auroc"),
                    "hmm_vs_baseline_auroc_gap": row.get("hmm_vs_baseline_auroc_gap"),
                    "calibrated_last_step_brier": _nested(row, "method_metrics.calibrated_last_step.brier"),
                    "calibrated_last_step_ece": _nested(row, "method_metrics.calibrated_last_step.ece"),
                    "hmm_score_brier": _nested(row, "method_metrics.hmm_score.brier"),
                    "hmm_score_ece": _nested(row, "method_metrics.hmm_score.ece"),
                    "ema_brier": _nested(row, "method_metrics.ema.brier"),
                    "ema_ece": _nested(row, "method_metrics.ema.ece"),
                    "ema_auprc": _nested(row, "method_metrics.ema.auprc"),
                    "hmm_hybrid_auprc": _nested(row, "method_metrics.hmm_hybrid.auprc"),
                    "hmm_score_brier_minus_ema": row.get("hmm_score_brier_minus_ema"),
                    "hmm_score_brier_minus_calibrated_last_step": row.get(
                        "hmm_score_brier_minus_calibrated_last_step"
                    ),
                    "hmm_score_ece_minus_ema": row.get("hmm_score_ece_minus_ema"),
                    "hmm_score_ece_minus_calibrated_last_step": row.get(
                        "hmm_score_ece_minus_calibrated_last_step"
                    ),
                    "hmm_hybrid_auroc_minus_ema": row.get("hmm_hybrid_auroc_minus_ema"),
                    "hmm_hybrid_auroc_minus_temporal_metric": row.get(
                        "hmm_hybrid_auroc_minus_temporal_metric"
                    ),
                    "hmm_hybrid_auprc_minus_ema": row.get("hmm_hybrid_auprc_minus_ema"),
                    "hmm_hybrid_auprc_minus_temporal_metric": row.get(
                        "hmm_hybrid_auprc_minus_temporal_metric"
                    ),
                    "error": row.get("error"),
                }
            )
