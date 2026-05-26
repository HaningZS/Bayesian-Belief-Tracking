"""Experiment-run diagnostics for generated local/API artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from csbf.metrics import roc_auc_score
from csbf.schema import TraceRecord, load_jsonl


BASELINE_METHODS = {
    "last_step",
    "calibrated_last_step",
    "mean_score",
    "moving_average",
    "calibrated_moving_average",
    "ema",
    "calibrated_ema",
    "temporal_metric",
    "learned_prefix_baseline",
    "prefix_feature_classifier",
}
ONLINE_HMM_METHODS = {"hmm_score", "hmm_concept", "hmm_hybrid", "hmm_joint", "hmm_nonstationary"}


def summarize_run_directory(run_dir: str | Path, target_error_band: tuple[float, float] = (0.3, 0.5)) -> dict[str, Any]:
    """Summarize one experiment run directory into custom diagnostic metrics."""

    path = Path(run_dir)
    config = _read_json(path / "config.json", default={})
    inspect = _read_json(path / "inspect.json", default={})
    pipeline = _read_json(path / "pipeline_report.json", default={})
    split_scan = _read_json(path / "split_seed_scan.json", default={})
    split_seed_sweep = _read_split_seed_sweep(path)
    utility = _read_utility_table(path)

    correctness = inspect.get("correctness", {})
    correct = int(correctness.get("correct", 0) or 0)
    incorrect = int(correctness.get("incorrect", 0) or 0)
    total = correct + incorrect
    error_rate = round(incorrect / total, 12) if total else None
    target_error_gap = _target_band_gap(error_rate, target_error_band)

    method_metrics = _method_metrics(pipeline)
    prefix_hmm = _prefix_hmm_metrics(pipeline)
    best_baseline = _best_method_auc(method_metrics, BASELINE_METHODS)
    best_online_hmm = _best_method_auc(method_metrics, ONLINE_HMM_METHODS)
    baseline_auc = best_baseline["auroc"] if best_baseline else None
    hmm_auc = best_online_hmm["auroc"] if best_online_hmm else None
    hmm_gap = _subtract_or_none(hmm_auc, baseline_auc)
    smoothing_gap = _subtract_or_none(
        _metric(method_metrics, "hmm_smooth", "auroc"),
        _metric(method_metrics, "hmm_hybrid", "auroc"),
    )

    score_summary = inspect.get("score_summary", {})
    observation_summary = inspect.get("observations_per_trace", {})
    trajectory = _trajectory_diagnostics(path / "records.jsonl")
    split = _primary_split(split_scan)

    flags = _custom_flags(
        error_rate=error_rate,
        target_error_gap=target_error_gap,
        score_summary=score_summary,
        baseline_auc=baseline_auc,
        hmm_auc=hmm_auc,
        smoothing_gap=smoothing_gap,
        prefix_hmm=prefix_hmm,
        inspect_warnings=inspect.get("warnings", []),
    )

    return {
        "run_id": path.name,
        "run_dir": str(path),
        "levels": config.get("math_levels"),
        "temperature": config.get("temperature"),
        "traces_per_question": config.get("traces_per_question"),
        "records": inspect.get("num_records"),
        "questions": inspect.get("num_questions"),
        "correct": correct,
        "incorrect": incorrect,
        "error_rate": error_rate,
        "target_error_gap": target_error_gap,
        "score_saturated_fraction": score_summary.get("saturated_fraction"),
        "score_unique_count": score_summary.get("unique_count"),
        "score_mean": score_summary.get("mean"),
        "observations_mean": observation_summary.get("mean"),
        "observations_min": observation_summary.get("min"),
        "observations_max": observation_summary.get("max"),
        "best_baseline_method": best_baseline["method"] if best_baseline else None,
        "best_baseline_auroc": baseline_auc,
        "best_online_hmm_method": best_online_hmm["method"] if best_online_hmm else None,
        "best_online_hmm_auroc": hmm_auc,
        "hmm_vs_baseline_auroc_gap": hmm_gap,
        "offline_smoothing_auroc_gap": smoothing_gap,
        "method_metrics": method_metrics,
        "prefix_hmm": prefix_hmm,
        "trajectory": trajectory,
        "split": split,
        "split_seed_sweep": split_seed_sweep,
        "utility": utility,
        "readiness": _readiness(error_rate, target_error_gap, score_summary, total, flags),
        "flags": flags,
    }


def collect_run_diagnostics(run_dirs: list[str | Path]) -> list[dict[str, Any]]:
    """Collect diagnostics from multiple run directories."""

    return [summarize_run_directory(run_dir) for run_dir in run_dirs]


def write_diagnostic_outputs(
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
    make_plots: bool = True,
) -> dict[str, str]:
    """Write JSON, CSV, Markdown, and optional SVG diagnostic summaries."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "experiment_diagnostics.json"
    csv_path = target / "experiment_diagnostics.csv"
    md_path = target / "experiment_diagnostics.md"

    payload = {
        "runs": summaries,
        "best_run": _best_summary(summaries),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(summaries, csv_path)
    _write_markdown(summaries, md_path)

    written = {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}
    if make_plots:
        for name, metric, title in [
            ("error_rate_by_run.svg", "error_rate", "Error Rate By Run"),
            ("hmm_gap_by_run.svg", "hmm_vs_baseline_auroc_gap", "Online HMM AUROC Gap vs Best Baseline"),
            ("final_score_auroc_by_run.svg", "trajectory.final_score_auroc", "Final Score AUROC By Run"),
        ]:
            svg_path = target / name
            _write_bar_svg(summaries, svg_path, metric_key=metric, title=title)
            written[name] = str(svg_path)
    return written


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_split_seed_sweep(path: Path) -> dict[str, Any]:
    for candidate in _split_seed_sweep_candidates(path):
        payload = _read_json(candidate, default={})
        if not isinstance(payload, dict):
            continue
        aggregate = payload.get("aggregate")
        if not isinstance(aggregate, dict):
            continue
        return {
            "seed_count": payload.get("seed_count"),
            "valid_auroc_seed_count": payload.get("valid_auroc_seed_count"),
            "use_model_selection": payload.get("use_model_selection"),
            "calibration_mode": payload.get("calibration_mode"),
            "aggregate": aggregate,
        }
    return {}


def _read_utility_table(path: Path) -> dict[str, Any]:
    for candidate in _utility_table_candidates(path):
        payload = _read_json(candidate, default={})
        if not isinstance(payload, dict):
            continue
        rows = payload.get("rows")
        if not isinstance(rows, list):
            continue
        valid_rows = [row for row in rows if isinstance(row, dict)]
        if not valid_rows:
            return {"row_count": 0, "methods": [], "best_compute_saved": None}
        return {
            "row_count": len(valid_rows),
            "methods": sorted({str(row.get("method")) for row in valid_rows if row.get("method") is not None}),
            "best_compute_saved": max(
                valid_rows,
                key=lambda row: float(row.get("mean_compute_saved") or 0.0),
            ),
        }
    return {}


def _utility_table_candidates(path: Path) -> list[Path]:
    named = [
        path / "utility_table.json",
        path / "records_utility_table.json",
        path / "diagnostics" / "utility_table.json",
        path / "diagnostics" / "records_utility_table.json",
    ]
    globbed = sorted(path.glob("*utility_table*.json"))
    globbed.extend(sorted((path / "diagnostics").glob("*utility_table*.json")))
    candidates: list[Path] = []
    for candidate in named + globbed:
        if candidate.exists() and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _split_seed_sweep_candidates(path: Path) -> list[Path]:
    named = [
        path / "split_seed_sweep_50.json",
        path / "split_seed_sweep.json",
        path / "records_split_seed_sweep.json",
        path / "diagnostics" / "split_seed_sweep_50.json",
        path / "diagnostics" / "split_seed_sweep.json",
        path / "diagnostics" / "records_split_seed_sweep.json",
    ]
    globbed = sorted(path.glob("*split_seed_sweep*.json"))
    globbed.extend(sorted((path / "diagnostics").glob("*split_seed_sweep*.json")))
    candidates: list[Path] = []
    for candidate in named + globbed:
        if candidate.exists() and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _method_metrics(pipeline: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    methods: dict[str, dict[str, float | None]] = {}
    for name, payload in pipeline.get("methods", {}).items():
        metrics = payload.get("metrics", {})
        methods[name] = {
            "auroc": _as_optional_float(metrics.get("auroc")),
            "auprc": _as_optional_float(metrics.get("auprc")),
            "brier": _as_optional_float(metrics.get("brier")),
            "ece": _as_optional_float(metrics.get("ece")),
        }
    return methods


def _prefix_hmm_metrics(pipeline: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    rows: dict[str, dict[str, float | None]] = {}
    for fraction_key, payload in pipeline.get("prefix_diagnostics", {}).items():
        rows[fraction_key] = {
            "online_auroc": _nested_float(payload, "hmm_hybrid_online.metrics.auroc"),
            "smooth_auroc": _nested_float(payload, "hmm_hybrid_smooth.metrics.auroc"),
            "viterbi_auroc": _nested_float(payload, "hmm_hybrid_viterbi.metrics.auroc"),
            "smooth_minus_online_auroc": _as_optional_float(payload.get("smooth_minus_online_auroc")),
        }
    return rows


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _nested_float(payload: dict[str, Any], key: str) -> float | None:
    current: Any = payload
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return _as_optional_float(current)


def _best_method_auc(methods: dict[str, dict[str, float | None]], candidates: set[str]) -> dict[str, float] | None:
    rows = [
        {"method": name, "auroc": float(metrics["auroc"])}
        for name, metrics in methods.items()
        if name in candidates and metrics.get("auroc") is not None
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: row["auroc"])


def _metric(methods: dict[str, dict[str, float | None]], method: str, metric: str) -> float | None:
    return methods.get(method, {}).get(metric)


def _subtract_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 12)


def _target_band_gap(error_rate: float | None, target_error_band: tuple[float, float]) -> float | None:
    if error_rate is None:
        return None
    lo, hi = target_error_band
    if lo <= error_rate <= hi:
        return 0.0
    if error_rate < lo:
        return round(lo - error_rate, 12)
    return round(error_rate - hi, 12)


def _trajectory_diagnostics(records_path: Path) -> dict[str, Any]:
    if not records_path.exists():
        return {}
    records = load_jsonl(records_path)
    labels = [1 if record.correct else 0 for record in records]
    early_scores = [_score_at_fraction(record, 0.05) for record in records]
    mid_scores = [_score_at_fraction(record, 0.50) for record in records]
    final_scores = [_score_at_fraction(record, 1.00) for record in records]
    observation_counts = [len(record.observations) for record in records]
    long_threshold = 50
    long_records = [record for record in records if len(record.observations) >= long_threshold]

    early_auc = _optional_auc(labels, early_scores)
    mid_auc = _optional_auc(labels, mid_scores)
    final_auc = _optional_auc(labels, final_scores)
    return {
        "early_score_auroc": early_auc,
        "mid_score_auroc": mid_auc,
        "final_score_auroc": final_auc,
        "mid_minus_early_auroc": _subtract_or_none(mid_auc, early_auc),
        "final_minus_mid_auroc": _subtract_or_none(final_auc, mid_auc),
        "early_score_label_gap": _label_gap(labels, early_scores),
        "mid_score_label_gap": _label_gap(labels, mid_scores),
        "final_score_label_gap": _label_gap(labels, final_scores),
        "long_trace_threshold": long_threshold,
        "long_trace_count": len(long_records),
        "long_trace_error_rate": _error_rate(long_records),
        "mean_observations_correct": _mean([count for count, label in zip(observation_counts, labels) if label == 1]),
        "mean_observations_incorrect": _mean([count for count, label in zip(observation_counts, labels) if label == 0]),
    }


def _score_at_fraction(record: TraceRecord, fraction: float) -> float:
    scores = [float(observation.score) for observation in record.observations if observation.score is not None]
    if not scores:
        raise ValueError(f"trace {record.trace_id} has no score observations")
    index = int((len(scores) - 1) * fraction + 0.5)
    return scores[int(index)]


def _optional_auc(labels: list[int], scores: list[float]) -> float | None:
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return None


def _label_gap(labels: list[int], scores: list[float]) -> float | None:
    correct_scores = [score for label, score in zip(labels, scores) if label == 1]
    incorrect_scores = [score for label, score in zip(labels, scores) if label == 0]
    correct_mean = _mean(correct_scores)
    incorrect_mean = _mean(incorrect_scores)
    if correct_mean is None or incorrect_mean is None:
        return None
    return round(correct_mean - incorrect_mean, 12)


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 12)


def _error_rate(records: list[TraceRecord]) -> float | None:
    if not records:
        return None
    return round(sum(1 for record in records if not record.correct) / len(records), 12)


def _primary_split(split_scan: dict[str, Any]) -> dict[str, Any]:
    seeds = split_scan.get("seeds") or []
    if not seeds:
        return {}
    first_two_class_seed = split_scan.get("first_two_class_seed")
    if first_two_class_seed is not None:
        for seed_summary in seeds:
            if seed_summary.get("seed") == first_two_class_seed:
                return seed_summary
    return seeds[0]


def _custom_flags(
    error_rate: float | None,
    target_error_gap: float | None,
    score_summary: dict[str, Any],
    baseline_auc: float | None,
    hmm_auc: float | None,
    smoothing_gap: float | None,
    prefix_hmm: dict[str, dict[str, float | None]],
    inspect_warnings: list[str],
) -> list[str]:
    flags = list(inspect_warnings)
    if error_rate is None:
        flags.append("missing_error_rate")
    elif target_error_gap and target_error_gap > 0.0:
        flags.append("error_rate_outside_target_band")
    if (score_summary.get("saturated_fraction") or 0.0) >= 0.5:
        flags.append("score_saturation_high")
    if int(score_summary.get("unique_count") or 0) < 5:
        flags.append("score_unique_count_lt_5")
    if baseline_auc is not None and hmm_auc is not None and hmm_auc < baseline_auc - 0.1:
        flags.append("online_hmm_underperforms_best_baseline")
    if smoothing_gap is not None and smoothing_gap > 0.1:
        flags.append("offline_smoothing_improves_over_online_hybrid")
    if any((metrics.get("smooth_minus_online_auroc") or 0.0) > 0.1 for metrics in prefix_hmm.values()):
        flags.append("offline_future_information_available")
    return flags


def _readiness(
    error_rate: float | None,
    target_error_gap: float | None,
    score_summary: dict[str, Any],
    total: int,
    flags: list[str],
) -> str:
    if total == 0 or error_rate is None:
        return "not_ready"
    if "only one correctness class is present" in flags:
        return "not_metric_ready"
    saturated = score_summary.get("saturated_fraction")
    unique_count = int(score_summary.get("unique_count") or 0)
    if target_error_gap is not None and target_error_gap <= 0.05 and (saturated is None or saturated < 0.5) and unique_count >= 5:
        return "scale_candidate"
    return "diagnostic"


def _best_summary(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [summary for summary in summaries if summary.get("target_error_gap") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda summary: (float(summary["target_error_gap"]), -int(summary.get("score_unique_count") or 0)))


def _write_csv(summaries: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "run_id",
        "levels",
        "temperature",
        "records",
        "correct",
        "incorrect",
        "error_rate",
        "target_error_gap",
        "score_saturated_fraction",
        "score_unique_count",
        "best_baseline_method",
        "best_baseline_auroc",
        "best_online_hmm_method",
        "best_online_hmm_auroc",
        "hmm_vs_baseline_auroc_gap",
        "offline_smoothing_auroc_gap",
        "p05_online_auroc",
        "p05_smooth_auroc",
        "p05_viterbi_auroc",
        "p05_smooth_minus_online_auroc",
        "p50_online_auroc",
        "p50_smooth_auroc",
        "p50_viterbi_auroc",
        "p50_smooth_minus_online_auroc",
        "split_seed_count",
        "split_seed_valid_auroc_count",
        "split_seed_use_model_selection",
        "split_seed_mean_hmm_gap",
        "split_seed_positive_hmm_gap_fraction",
        "split_seed_mean_calibrated_last_step_brier",
        "split_seed_hmm_score_brier_minus_ema",
        "split_seed_hmm_score_brier_minus_calibrated_last_step",
        "utility_row_count",
        "utility_best_method",
        "utility_best_mean_compute_saved",
        "readiness",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {field: summary.get(field) for field in fields}
            prefix_hmm = summary.get("prefix_hmm") or {}
            for key, label in [("p05", "p05"), ("p50", "p50")]:
                metrics = prefix_hmm.get(key, {}) if isinstance(prefix_hmm, dict) else {}
                row[f"{label}_online_auroc"] = metrics.get("online_auroc")
                row[f"{label}_smooth_auroc"] = metrics.get("smooth_auroc")
                row[f"{label}_viterbi_auroc"] = metrics.get("viterbi_auroc")
                row[f"{label}_smooth_minus_online_auroc"] = metrics.get("smooth_minus_online_auroc")
            split_seed_sweep = summary.get("split_seed_sweep") or {}
            aggregate = split_seed_sweep.get("aggregate", {}) if isinstance(split_seed_sweep, dict) else {}
            row["split_seed_count"] = split_seed_sweep.get("seed_count") if isinstance(split_seed_sweep, dict) else None
            row["split_seed_valid_auroc_count"] = split_seed_sweep.get("valid_auroc_seed_count") if isinstance(split_seed_sweep, dict) else None
            row["split_seed_use_model_selection"] = split_seed_sweep.get("use_model_selection") if isinstance(split_seed_sweep, dict) else None
            row["split_seed_mean_hmm_gap"] = aggregate.get("mean_hmm_vs_baseline_auroc_gap") if isinstance(aggregate, dict) else None
            row["split_seed_positive_hmm_gap_fraction"] = aggregate.get("positive_hmm_gap_fraction") if isinstance(aggregate, dict) else None
            row["split_seed_mean_calibrated_last_step_brier"] = aggregate.get("mean_calibrated_last_step_brier") if isinstance(aggregate, dict) else None
            row["split_seed_hmm_score_brier_minus_ema"] = aggregate.get("hmm_score_brier_minus_ema") if isinstance(aggregate, dict) else None
            row["split_seed_hmm_score_brier_minus_calibrated_last_step"] = aggregate.get("hmm_score_brier_minus_calibrated_last_step") if isinstance(aggregate, dict) else None
            utility = summary.get("utility") or {}
            best_utility = utility.get("best_compute_saved") if isinstance(utility, dict) else {}
            row["utility_row_count"] = utility.get("row_count") if isinstance(utility, dict) else None
            row["utility_best_method"] = best_utility.get("method") if isinstance(best_utility, dict) else None
            row["utility_best_mean_compute_saved"] = best_utility.get("mean_compute_saved") if isinstance(best_utility, dict) else None
            if isinstance(row["levels"], list):
                row["levels"] = "/".join(str(level) for level in row["levels"])
            writer.writerow(row)


def _write_markdown(summaries: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Experiment Diagnostics",
        "",
        "| Run | Levels | Temp | Error | Saturation | Baseline AUROC | HMM AUROC | HMM Gap | Smoothing Gap | Readiness |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        levels = "/".join(str(level) for level in (summary.get("levels") or []))
        lines.append(
            "| {run_id} | {levels} | {temperature} | {error_rate} | {sat} | {baseline} | {hmm} | {gap} | {smooth_gap} | {readiness} |".format(
                run_id=summary.get("run_id"),
                levels=levels,
                temperature=_fmt(summary.get("temperature")),
                error_rate=_fmt(summary.get("error_rate")),
                sat=_fmt(summary.get("score_saturated_fraction")),
                baseline=_fmt(summary.get("best_baseline_auroc")),
                hmm=_fmt(summary.get("best_online_hmm_auroc")),
                gap=_fmt(summary.get("hmm_vs_baseline_auroc_gap")),
                smooth_gap=_fmt(summary.get("offline_smoothing_auroc_gap")),
                readiness=summary.get("readiness"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_bar_svg(summaries: list[dict[str, Any]], path: Path, metric_key: str, title: str) -> None:
    width = 960
    height = 360
    margin_left = 80
    margin_bottom = 95
    margin_top = 45
    plot_width = width - margin_left - 30
    plot_height = height - margin_top - margin_bottom
    values = [_nested_metric(summary, metric_key) for summary in summaries]
    numeric_values = [float(value) for value in values if value is not None]
    min_value = min(0.0, min(numeric_values) if numeric_values else 0.0)
    max_value = max(1.0, max(numeric_values) if numeric_values else 1.0)
    scale = max_value - min_value or 1.0
    bar_width = max(12, int(plot_width / max(1, len(summaries)) * 0.62))
    slot = plot_width / max(1, len(summaries))
    zero_y = margin_top + plot_height - ((0.0 - min_value) / scale) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="Arial" font-size="18">{_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - 30}" y2="{zero_y:.1f}" stroke="#777" stroke-width="1"/>',
    ]
    for tick in [min_value, (min_value + max_value) / 2, max_value]:
        y = margin_top + plot_height - ((tick - min_value) / scale) * plot_height
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - 30}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick:.2f}</text>')
    for index, summary in enumerate(summaries):
        value = _nested_metric(summary, metric_key)
        x = margin_left + index * slot + (slot - bar_width) / 2
        label = str(summary.get("run_id", f"run-{index}"))
        if value is not None:
            value_f = float(value)
            y = margin_top + plot_height - ((max(value_f, 0.0) - min_value) / scale) * plot_height
            bar_h = abs(((value_f - 0.0) / scale) * plot_height)
            if value_f < 0:
                y = zero_y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" fill="#4f7cac"/>')
            parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value_f:.2f}</text>')
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 18}" text-anchor="end" font-family="Arial" font-size="10" transform="rotate(-35 {x + bar_width / 2:.1f},{height - 18})">{_escape(label)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _nested_metric(summary: dict[str, Any], key: str) -> float | int | None:
    current: Any = summary
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    if current is None:
        return None
    return float(current)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
