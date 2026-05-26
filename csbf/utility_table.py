"""Online decision-utility tables for prefix reliability methods."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from csbf.baselines import exponential_moving_average, moving_average
from csbf.evaluation import (
    _base_rate,
    _fit_learned_prefix_model,
    _fit_likelihoods,
    _paired_scores_codes,
    _predict_learned_prefix_model,
)
from csbf.filtering import BayesianReliabilityFilter, HMMConfig, NonStationaryHMMConfig
from csbf.schema import TraceRecord, load_jsonl
from csbf.split import split_by_question


DEFAULT_METHODS = ("score_prefix", "ema", "moving_average", "learned_prefix_baseline", "hmm_hybrid")
DEFAULT_FALSE_POSITIVE_RATES = (0.05, 0.10, 0.20)
DEFAULT_RECALLS = (0.50, 0.75, 0.90)


def build_utility_table(
    records_or_path: Sequence[TraceRecord] | str | Path,
    seed: int = 0,
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    calibration_mode: str = "all_steps",
    methods: Sequence[str] = DEFAULT_METHODS,
    false_positive_rates: Sequence[float] = DEFAULT_FALSE_POSITIVE_RATES,
    recalls: Sequence[float] = DEFAULT_RECALLS,
    moving_average_window: int = 3,
    ema_alpha: float = 0.4,
    hmm_config: HMMConfig | NonStationaryHMMConfig | None = None,
) -> dict[str, Any]:
    """Build fixed-threshold online error-detection utility rows.

    Thresholds are selected on calibration records and evaluated on held-out
    test records. Risk means probability of error, so a trace is flagged at
    the first prefix whose risk is greater than or equal to the threshold.
    """

    checked_methods = _validate_methods(methods)
    checked_fprs = _validate_targets(false_positive_rates, "false_positive_rates")
    checked_recalls = _validate_targets(recalls, "recalls")
    records = _load_records(records_or_path)
    splits = split_by_question(
        records,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        seed=seed,
    )
    context = _build_context(
        splits["calibration"],
        calibration_mode=calibration_mode,
        moving_average_window=moving_average_window,
        ema_alpha=ema_alpha,
        hmm_config=hmm_config,
    )
    rows: list[dict[str, Any]] = []
    for method in checked_methods:
        calibration_risks = [
            _record_risks(record, method, context, moving_average_window, ema_alpha)
            for record in splits["calibration"]
        ]
        test_risks = [
            _record_risks(record, method, context, moving_average_window, ema_alpha)
            for record in splits["test"]
        ]
        for target in checked_fprs:
            selected = _select_threshold(calibration_risks, "max_fpr", target)
            rows.append(
                _utility_row(
                    method=method,
                    selection_rule="max_fpr",
                    target_false_positive_rate=target,
                    target_recall=None,
                    threshold=selected["threshold"],
                    calibration_metrics=selected["metrics"],
                    test_risks=test_risks,
                )
            )
        for target in checked_recalls:
            selected = _select_threshold(calibration_risks, "min_recall", target)
            rows.append(
                _utility_row(
                    method=method,
                    selection_rule="min_recall",
                    target_false_positive_rate=None,
                    target_recall=target,
                    threshold=selected["threshold"],
                    calibration_metrics=selected["metrics"],
                    test_risks=test_risks,
                )
            )
    return {
        "num_records": len(records),
        "num_questions": len({record.question_id for record in records}),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "calibration_ratio": float(calibration_ratio),
        "calibration_mode": calibration_mode,
        "methods": checked_methods,
        "false_positive_rates": checked_fprs,
        "recalls": checked_recalls,
        "split": {name: _split_counts(split_records) for name, split_records in splits.items()},
        "rows": rows,
    }


def build_utility_split_seed_sweep(
    records_or_path: Sequence[TraceRecord] | str | Path,
    seeds: Sequence[int],
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    calibration_mode: str = "all_steps",
    methods: Sequence[str] = DEFAULT_METHODS,
    false_positive_rates: Sequence[float] = DEFAULT_FALSE_POSITIVE_RATES,
    recalls: Sequence[float] = DEFAULT_RECALLS,
    moving_average_window: int = 3,
    ema_alpha: float = 0.4,
    hmm_config: HMMConfig | NonStationaryHMMConfig | None = None,
) -> dict[str, Any]:
    """Aggregate utility-table operating points over split seeds."""

    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("seeds must not be empty")
    checked_methods = _validate_methods(methods)
    checked_fprs = _validate_targets(false_positive_rates, "false_positive_rates")
    checked_recalls = _validate_targets(recalls, "recalls")
    records = _load_records(records_or_path)
    seed_rows: list[dict[str, Any]] = []
    for seed in seed_values:
        try:
            summary = build_utility_table(
                records,
                seed=seed,
                train_ratio=train_ratio,
                calibration_ratio=calibration_ratio,
                calibration_mode=calibration_mode,
                methods=checked_methods,
                false_positive_rates=checked_fprs,
                recalls=checked_recalls,
                moving_average_window=moving_average_window,
                ema_alpha=ema_alpha,
                hmm_config=hmm_config,
            )
        except ValueError as error:
            seed_rows.append({"seed": seed, "error": str(error), "split": None, "rows": []})
            continue
        seed_rows.append(
            {
                "seed": seed,
                "split": summary["split"],
                "rows": summary["rows"],
            }
        )
    return {
        "num_records": len(records),
        "num_questions": len({record.question_id for record in records}),
        "seed_count": len(seed_rows),
        "valid_seed_count": sum(1 for row in seed_rows if not row.get("error")),
        "train_ratio": float(train_ratio),
        "calibration_ratio": float(calibration_ratio),
        "calibration_mode": calibration_mode,
        "methods": checked_methods,
        "false_positive_rates": checked_fprs,
        "recalls": checked_recalls,
        "aggregate_rows": _aggregate_utility_seed_rows(seed_rows),
        "seeds": seed_rows,
    }


def write_utility_split_seed_sweep(
    summary: dict[str, Any],
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, str]:
    """Write a split-seed utility summary to JSON and optional aggregate CSV."""

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = {"json": str(json_path)}
    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_utility_sweep_csv(summary["aggregate_rows"], csv_path)
        written["csv"] = str(csv_path)
    return written


def write_utility_table(
    summary: dict[str, Any],
    output_json: str | Path,
    output_csv: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> dict[str, str]:
    """Write a utility summary to JSON, CSV, and optional Markdown."""

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = {"json": str(json_path)}
    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(summary["rows"], csv_path)
        written["csv"] = str(csv_path)
    if output_markdown is not None:
        markdown_path = Path(output_markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(summary, markdown_path)
        written["markdown"] = str(markdown_path)
    return written


def _build_context(
    calibration_records: list[TraceRecord],
    calibration_mode: str,
    moving_average_window: int,
    ema_alpha: float,
    hmm_config: HMMConfig | NonStationaryHMMConfig | None,
) -> dict[str, Any]:
    config = hmm_config or HMMConfig(initial_on_track=_base_rate(calibration_records))
    score_likelihood, concept_likelihood, _ = _fit_likelihoods(
        calibration_records,
        config,
        calibration_mode,
    )
    return {
        "hmm_config": config,
        "score_likelihood": score_likelihood,
        "concept_likelihood": concept_likelihood,
        "learned_prefix_model": _fit_learned_prefix_model(
            calibration_records,
            moving_average_window=moving_average_window,
            ema_alpha=ema_alpha,
        ),
    }


def _record_risks(
    record: TraceRecord,
    method: str,
    context: dict[str, Any],
    moving_average_window: int,
    ema_alpha: float,
) -> dict[str, Any]:
    scores = [float(observation.score) for observation in record.observations if observation.score is not None]
    if not scores:
        raise ValueError(f"trace {record.trace_id} has no score observations")
    if method == "score_prefix":
        probabilities = scores
    elif method == "ema":
        probabilities = exponential_moving_average(scores, ema_alpha)
    elif method == "moving_average":
        probabilities = moving_average(scores, moving_average_window)
    elif method == "learned_prefix_baseline":
        probabilities = [
            _predict_learned_prefix_model(
                context["learned_prefix_model"],
                scores[: index + 1],
                moving_average_window=moving_average_window,
                ema_alpha=ema_alpha,
            )
            for index in range(len(scores))
        ]
    elif method == "hmm_hybrid":
        paired_scores, concept_codes = _paired_scores_codes(record)
        probabilities = BayesianReliabilityFilter(
            config=context["hmm_config"],
            score_likelihood=context["score_likelihood"],
            concept_likelihood=context["concept_likelihood"],
        ).run(scores=paired_scores, concept_codes=concept_codes)
    else:
        raise ValueError(f"unsupported utility method: {method}")
    return {
        "trace_id": record.trace_id,
        "question_id": record.question_id,
        "correct": bool(record.correct),
        "risks": [round(1.0 - float(probability), 12) for probability in probabilities],
    }


def _select_threshold(
    calibration_risks: list[dict[str, Any]],
    rule: str,
    target: float,
) -> dict[str, Any]:
    candidates = _candidate_thresholds(calibration_risks)
    metrics_by_threshold = _threshold_metrics_by_threshold(calibration_risks, candidates)
    scored = [
        {"threshold": threshold, "metrics": metrics_by_threshold[threshold]}
        for threshold in candidates
    ]
    if rule == "max_fpr":
        feasible = [
            row for row in scored
            if float(row["metrics"]["false_positive_rate"]) <= target
        ]
        pool = feasible or scored
        return max(
            pool,
            key=lambda row: (
                float(row["metrics"]["recall"]),
                float(row["metrics"]["mean_compute_saved"]),
                -float(row["threshold"]),
            ),
        )
    if rule == "min_recall":
        feasible = [row for row in scored if float(row["metrics"]["recall"]) >= target]
        pool = feasible or scored
        return min(
            pool,
            key=lambda row: (
                float(row["metrics"]["false_positive_rate"]),
                -float(row["metrics"]["mean_compute_saved"]),
                -float(row["threshold"]),
            ),
        )
    raise ValueError("rule must be 'max_fpr' or 'min_recall'")


def _utility_row(
    method: str,
    selection_rule: str,
    target_false_positive_rate: float | None,
    target_recall: float | None,
    threshold: float,
    calibration_metrics: dict[str, Any],
    test_risks: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _threshold_metrics(test_risks, threshold)
    return {
        "method": method,
        "selection_rule": selection_rule,
        "target_false_positive_rate": target_false_positive_rate,
        "target_recall": target_recall,
        "threshold": round(float(threshold), 12),
        "calibration_false_positive_rate": calibration_metrics["false_positive_rate"],
        "calibration_recall": calibration_metrics["recall"],
        **metrics,
    }


def _threshold_metrics(record_risks: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    correct_count = sum(1 for row in record_risks if row["correct"])
    incorrect_count = len(record_risks) - correct_count
    true_positive = 0
    false_positive = 0
    detected = 0
    detection_times: list[float] = []
    compute_saved: list[float] = []
    incorrect_compute_saved: list[float] = []
    for row in record_risks:
        risks = [float(value) for value in row["risks"]]
        crossing = _first_crossing(risks, threshold)
        saved = 0.0
        if crossing is not None:
            detected += 1
            normalized_time = (crossing + 1) / len(risks)
            detection_times.append(normalized_time)
            saved = max(0.0, 1.0 - normalized_time)
            if row["correct"]:
                false_positive += 1
            else:
                true_positive += 1
        compute_saved.append(saved)
        if not row["correct"]:
            incorrect_compute_saved.append(saved)
    precision_denominator = true_positive + false_positive
    return {
        "records": len(record_risks),
        "correct": correct_count,
        "incorrect": incorrect_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_positive_rate": round(false_positive / correct_count, 12) if correct_count else 0.0,
        "recall": round(true_positive / incorrect_count, 12) if incorrect_count else 0.0,
        "precision": round(true_positive / precision_denominator, 12) if precision_denominator else None,
        "detection_rate": round(detected / len(record_risks), 12) if record_risks else 0.0,
        "mean_normalized_detection_time": _mean_or_none(detection_times),
        "mean_compute_saved": _mean(compute_saved),
        "incorrect_mean_compute_saved": _mean(incorrect_compute_saved),
    }


def _threshold_metrics_by_threshold(
    record_risks: list[dict[str, Any]],
    thresholds: Sequence[float],
) -> dict[float, dict[str, Any]]:
    """Compute utility metrics for many thresholds in one pass per record."""

    threshold_values = sorted({round(float(threshold), 12) for threshold in thresholds}, reverse=True)
    correct_count = sum(1 for row in record_risks if row["correct"])
    incorrect_count = len(record_risks) - correct_count
    stats = [
        {
            "true_positive": 0,
            "false_positive": 0,
            "detected": 0,
            "detection_time_sum": 0.0,
            "detection_time_count": 0,
            "compute_saved_sum": 0.0,
            "incorrect_compute_saved_sum": 0.0,
        }
        for _ in threshold_values
    ]
    for row in record_risks:
        risks = [float(value) for value in row["risks"]]
        events = sorted(
            ((risk, index) for index, risk in enumerate(risks)),
            key=lambda item: item[0],
            reverse=True,
        )
        event_index = 0
        crossing: int | None = None
        for threshold_index, threshold in enumerate(threshold_values):
            while event_index < len(events) and events[event_index][0] >= threshold:
                observation_index = events[event_index][1]
                crossing = observation_index if crossing is None else min(crossing, observation_index)
                event_index += 1
            saved = 0.0
            if crossing is not None:
                normalized_time = (crossing + 1) / len(risks)
                saved = max(0.0, 1.0 - normalized_time)
                stats[threshold_index]["detected"] += 1
                stats[threshold_index]["detection_time_sum"] += normalized_time
                stats[threshold_index]["detection_time_count"] += 1
                if row["correct"]:
                    stats[threshold_index]["false_positive"] += 1
                else:
                    stats[threshold_index]["true_positive"] += 1
            stats[threshold_index]["compute_saved_sum"] += saved
            if not row["correct"]:
                stats[threshold_index]["incorrect_compute_saved_sum"] += saved

    metrics_by_threshold: dict[float, dict[str, Any]] = {}
    for threshold, row_stats in zip(threshold_values, stats, strict=True):
        true_positive = int(row_stats["true_positive"])
        false_positive = int(row_stats["false_positive"])
        detected = int(row_stats["detected"])
        precision_denominator = true_positive + false_positive
        detection_time_count = int(row_stats["detection_time_count"])
        metrics_by_threshold[threshold] = {
            "records": len(record_risks),
            "correct": correct_count,
            "incorrect": incorrect_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_positive_rate": round(false_positive / correct_count, 12) if correct_count else 0.0,
            "recall": round(true_positive / incorrect_count, 12) if incorrect_count else 0.0,
            "precision": round(true_positive / precision_denominator, 12)
            if precision_denominator
            else None,
            "detection_rate": round(detected / len(record_risks), 12) if record_risks else 0.0,
            "mean_normalized_detection_time": round(
                float(row_stats["detection_time_sum"]) / detection_time_count,
                12,
            )
            if detection_time_count
            else None,
            "mean_compute_saved": round(float(row_stats["compute_saved_sum"]) / len(record_risks), 12)
            if record_risks
            else 0.0,
            "incorrect_mean_compute_saved": round(
                float(row_stats["incorrect_compute_saved_sum"]) / incorrect_count,
                12,
            )
            if incorrect_count
            else 0.0,
        }
    return metrics_by_threshold


def _aggregate_utility_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float | None, float | None], list[dict[str, Any]]] = {}
    for seed_row in seed_rows:
        if seed_row.get("error"):
            continue
        for row in seed_row.get("rows", []):
            key = (
                row["method"],
                row["selection_rule"],
                row["target_false_positive_rate"],
                row["target_recall"],
            )
            grouped.setdefault(key, []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        method, selection_rule, target_fpr, target_recall = key
        rows = grouped[key]
        aggregate_rows.append(
            {
                "method": method,
                "selection_rule": selection_rule,
                "target_false_positive_rate": target_fpr,
                "target_recall": target_recall,
                "valid_seed_count": len(rows),
                "target_hit_fraction": _target_hit_fraction(rows, selection_rule, target_fpr, target_recall),
                "mean_threshold": _mean_present(row.get("threshold") for row in rows),
                "median_threshold": _median_present(row.get("threshold") for row in rows),
                "mean_calibration_false_positive_rate": _mean_present(
                    row.get("calibration_false_positive_rate") for row in rows
                ),
                "mean_calibration_recall": _mean_present(row.get("calibration_recall") for row in rows),
                "mean_false_positive_rate": _mean_present(row.get("false_positive_rate") for row in rows),
                "median_false_positive_rate": _median_present(row.get("false_positive_rate") for row in rows),
                "mean_recall": _mean_present(row.get("recall") for row in rows),
                "median_recall": _median_present(row.get("recall") for row in rows),
                "mean_precision": _mean_present(row.get("precision") for row in rows),
                "mean_detection_rate": _mean_present(row.get("detection_rate") for row in rows),
                "mean_normalized_detection_time": _mean_present(
                    row.get("mean_normalized_detection_time") for row in rows
                ),
                "mean_compute_saved": _mean_present(row.get("mean_compute_saved") for row in rows),
                "median_compute_saved": _median_present(row.get("mean_compute_saved") for row in rows),
                "mean_incorrect_compute_saved": _mean_present(
                    row.get("incorrect_mean_compute_saved") for row in rows
                ),
                "median_incorrect_compute_saved": _median_present(
                    row.get("incorrect_mean_compute_saved") for row in rows
                ),
            }
        )
    return aggregate_rows


def _target_hit_fraction(
    rows: list[dict[str, Any]],
    selection_rule: str,
    target_fpr: float | None,
    target_recall: float | None,
) -> float | None:
    if not rows:
        return None
    if selection_rule == "max_fpr" and target_fpr is not None:
        hits = sum(1 for row in rows if float(row["false_positive_rate"]) <= float(target_fpr))
        return round(hits / len(rows), 12)
    if selection_rule == "min_recall" and target_recall is not None:
        hits = sum(1 for row in rows if float(row["recall"]) >= float(target_recall))
        return round(hits / len(rows), 12)
    return None


def _candidate_thresholds(record_risks: list[dict[str, Any]]) -> list[float]:
    risks = {
        round(float(value), 12)
        for row in record_risks
        for value in row["risks"]
    }
    return sorted({0.0, 1.0, *risks})


def _first_crossing(risks: Sequence[float], threshold: float) -> int | None:
    for index, risk in enumerate(risks):
        if risk >= threshold:
            return index
    return None


def _validate_methods(methods: Sequence[str]) -> list[str]:
    values = [str(method).strip() for method in methods if str(method).strip()]
    if not values:
        raise ValueError("methods must not be empty")
    unsupported = sorted(set(values) - set(DEFAULT_METHODS))
    if unsupported:
        raise ValueError(f"unsupported utility methods: {', '.join(unsupported)}")
    return values


def _validate_targets(values: Sequence[float], name: str) -> list[float]:
    checked = [float(value) for value in values]
    if not checked:
        raise ValueError(f"{name} must not be empty")
    for value in checked:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} values must be in [0, 1]")
    return checked


def _split_counts(records: list[TraceRecord]) -> dict[str, int]:
    return {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "correct": sum(1 for record in records if record.correct),
        "incorrect": sum(1 for record in records if not record.correct),
    }


def _load_records(records_or_path: Sequence[TraceRecord] | str | Path) -> list[TraceRecord]:
    if isinstance(records_or_path, str | Path):
        return load_jsonl(records_or_path)
    return list(records_or_path)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 12)


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _mean(values)


def _present_numbers(values: Iterable[object]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _mean_present(values: Iterable[object]) -> float | None:
    present = _present_numbers(values)
    if not present:
        return None
    return round(sum(present) / len(present), 12)


def _median_present(values: Iterable[object]) -> float | None:
    present = _present_numbers(values)
    if not present:
        return None
    return round(median(present), 12)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "method",
        "selection_rule",
        "target_false_positive_rate",
        "target_recall",
        "threshold",
        "calibration_false_positive_rate",
        "calibration_recall",
        "records",
        "correct",
        "incorrect",
        "true_positive",
        "false_positive",
        "false_positive_rate",
        "recall",
        "precision",
        "detection_rate",
        "mean_normalized_detection_time",
        "mean_compute_saved",
        "incorrect_mean_compute_saved",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_utility_sweep_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "method",
        "selection_rule",
        "target_false_positive_rate",
        "target_recall",
        "valid_seed_count",
        "target_hit_fraction",
        "mean_threshold",
        "median_threshold",
        "mean_calibration_false_positive_rate",
        "mean_calibration_recall",
        "mean_false_positive_rate",
        "median_false_positive_rate",
        "mean_recall",
        "median_recall",
        "mean_precision",
        "mean_detection_rate",
        "mean_normalized_detection_time",
        "mean_compute_saved",
        "median_compute_saved",
        "mean_incorrect_compute_saved",
        "median_incorrect_compute_saved",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Utility Table",
        "",
        "Thresholds are selected on calibration records and evaluated on the held-out test split.",
        "",
        "| Method | Rule | Target | Threshold | FPR | Recall | Precision | Saved | Bad-trace Saved |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        target = row["target_false_positive_rate"]
        if target is None:
            target = row["target_recall"]
        lines.append(
            "| {method} | {rule} | {target} | {threshold} | {fpr} | {recall} | {precision} | {saved} | {bad_saved} |".format(
                method=row["method"],
                rule=row["selection_rule"],
                target=_fmt(target),
                threshold=_fmt(row["threshold"]),
                fpr=_fmt(row["false_positive_rate"]),
                recall=_fmt(row["recall"]),
                precision=_fmt(row["precision"]),
                saved=_fmt(row["mean_compute_saved"]),
                bad_saved=_fmt(row["incorrect_mean_compute_saved"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"
