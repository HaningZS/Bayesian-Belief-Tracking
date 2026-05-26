"""Latent-trajectory hidden-state dynamics baselines."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from csbf.evaluation import evaluate_predictions, evaluate_trace_methods
from csbf.experiment_diagnostics import BASELINE_METHODS, ONLINE_HMM_METHODS
from csbf.metrics import roc_auc_score
from csbf.model_selection import select_hmm_config
from csbf.pipeline import _sub_split_calibration
from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl
from csbf.split import split_by_question
from csbf.split_seed_sweep import (
    _aggregate as _aggregate_seed_rows,
    _best_method_auc,
    _config_to_dict,
    _empty_method_summary,
    _fixed_method_gaps,
    _nested,
    _split_counts,
    _subtract_or_none,
)

_COMPONENTS = ("net_change", "cumulative_change", "aligned_change")
_METRICS = (*_COMPONENTS, "composite")
_DEFAULT_ORIENTATION = {
    "net_change": 1,
    "cumulative_change": -1,
    "aligned_change": 1,
}


@dataclass(frozen=True)
class LatentTrajectoryConfig:
    """Configuration for hidden-state trajectory metric scoring."""

    projection_dim: int = 64
    metric: str = "composite"
    normalize_vectors: bool = False
    feature_field: str = "hidden_last_token"


@dataclass(frozen=True)
class _FeatureEntry:
    question_id: str
    trace_id: str
    step_index: int
    label: int
    vector: list[float]


@dataclass(frozen=True)
class _ComponentTransform:
    name: str
    orientation: int
    minimum: float
    maximum: float
    train_auroc: float | None


@dataclass(frozen=True)
class _TrajectoryScorer:
    metric: str
    components: dict[str, _ComponentTransform]


def compute_trajectory_metrics(vectors: list[list[float]]) -> dict[str, float]:
    """Compute prefix-safe latent-trajectory metrics over hidden vectors.

    `aligned_change` uses the current prefix endpoint as the drift endpoint,
    which keeps the metric valid for online prefixes.
    """

    if not vectors:
        raise ValueError("vectors must not be empty")
    if len(vectors) == 1:
        return {"net_change": 0.0, "cumulative_change": 0.0, "aligned_change": 0.0}
    dim = len(vectors[0])
    if any(len(vector) != dim for vector in vectors):
        raise ValueError("all vectors must have the same dimension")

    drift = _subtract(vectors[-1], vectors[0])
    drift_norm = _norm(drift)
    updates = [
        _subtract(vectors[index], vectors[index - 1])
        for index in range(1, len(vectors))
    ]
    update_norms = [_norm(update) for update in updates]
    cumulative = sum(update_norms)
    aligned_values = [
        _dot(update, drift) / (update_norm * drift_norm)
        for update, update_norm in zip(updates, update_norms, strict=True)
        if update_norm > 0.0 and drift_norm > 0.0
    ]
    return {
        "net_change": drift_norm / len(vectors),
        "cumulative_change": cumulative,
        "aligned_change": sum(aligned_values) / len(aligned_values) if aligned_values else 0.0,
    }


def build_latent_trajectory_records(
    records_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    seed: int = 0,
    config: LatentTrajectoryConfig | None = None,
) -> dict[str, Any]:
    """Rewrite observation scores with train-oriented latent trajectory metrics."""

    cfg = _validate_config(config or LatentTrajectoryConfig())
    records = load_jsonl(records_path)
    splits = split_by_question(records, train_ratio=train_ratio, calibration_ratio=calibration_ratio, seed=seed)
    split_by_trace = {
        record.trace_id: split_name
        for split_name, split_records in splits.items()
        for record in split_records
    }
    labels_by_trace = {record.trace_id: 1 if record.correct else 0 for record in records}
    question_by_trace = {record.trace_id: record.question_id for record in records}
    required_keys = {
        (record.trace_id, observation.step_index)
        for record in records
        for observation in record.observations
    }
    feature_rows = _load_feature_rows(
        features_path,
        required_keys=required_keys,
        labels_by_trace=labels_by_trace,
        question_by_trace=question_by_trace,
        config=cfg,
    )
    _check_missing_features(required_keys, feature_rows)
    entries_by_trace = _entries_by_trace(feature_rows)
    metric_rows = _prefix_metric_rows(entries_by_trace)
    train_entries = [
        entry
        for entry in feature_rows
        if split_by_trace[entry.trace_id] == "train"
    ]
    train_trace_ids = {entry.trace_id for entry in train_entries}
    scorer = _fit_trajectory_scorer_from_rows(
        [row for row in metric_rows if str(row["trace_id"]) in train_trace_ids],
        cfg,
    )
    scores_by_key = _score_metric_rows(metric_rows, scorer)
    rewritten = _rewrite_records(records, scores_by_key, cfg, scorer)
    save_jsonl(rewritten, output_path)

    rewritten_by_trace = {record.trace_id: record for record in rewritten}
    rewritten_splits = {
        name: [rewritten_by_trace[record.trace_id] for record in split_records]
        for name, split_records in splits.items()
    }
    summary: dict[str, Any] = {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "split": {name: _split_summary(split_records) for name, split_records in splits.items()},
        "feature_rows": {
            "required": len(required_keys),
            "used": len(feature_rows),
            "projection_dim": cfg.projection_dim,
            "normalize_vectors": cfg.normalize_vectors,
            "feature_field": cfg.feature_field,
        },
        "transform": _scorer_to_dict(scorer),
        "lt_final_metrics": {
            name: _final_score_metrics(split_records)
            for name, split_records in rewritten_splits.items()
        },
        "outputs": {"records": str(output_path)},
    }
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        summary["outputs"]["report"] = str(report)
        report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_latent_trajectory_split_seed_sweep(
    records_path: str | Path,
    features_path: str | Path,
    seeds: Iterable[int],
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    config: LatentTrajectoryConfig | None = None,
    use_model_selection: bool = False,
    calibration_mode: str = "all_steps",
) -> dict[str, Any]:
    """Refit LT score orientation/scaling per split seed before HMM evaluation."""

    cfg = _validate_config(config or LatentTrajectoryConfig())
    records = load_jsonl(records_path)
    labels_by_trace = {record.trace_id: 1 if record.correct else 0 for record in records}
    question_by_trace = {record.trace_id: record.question_id for record in records}
    required_keys = {
        (record.trace_id, observation.step_index)
        for record in records
        for observation in record.observations
    }
    feature_rows = _load_feature_rows(
        features_path,
        required_keys=required_keys,
        labels_by_trace=labels_by_trace,
        question_by_trace=question_by_trace,
        config=cfg,
    )
    _check_missing_features(required_keys, feature_rows)
    entries_by_trace = _entries_by_trace(feature_rows)
    metric_rows = _prefix_metric_rows(entries_by_trace)
    metric_rows_by_trace = _metric_rows_by_trace(metric_rows)
    seed_rows = [
        _evaluate_latent_trajectory_split_seed(
            records,
            metric_rows=metric_rows,
            metric_rows_by_trace=metric_rows_by_trace,
            seed=int(seed),
            train_ratio=train_ratio,
            calibration_ratio=calibration_ratio,
            config=cfg,
            use_model_selection=use_model_selection,
            calibration_mode=calibration_mode,
        )
        for seed in seeds
    ]
    return {
        "num_records": len(records),
        "num_questions": len({record.question_id for record in records}),
        "feature_rows": {
            "required": len(required_keys),
            "used": len(feature_rows),
            "projection_dim": cfg.projection_dim,
            "normalize_vectors": cfg.normalize_vectors,
            "feature_field": cfg.feature_field,
        },
        "transform": {
            "metric": cfg.metric,
        },
        "seed_count": len(seed_rows),
        "train_ratio": train_ratio,
        "calibration_ratio": calibration_ratio,
        "use_model_selection": bool(use_model_selection),
        "calibration_mode": calibration_mode,
        "valid_auroc_seed_count": sum(row["hmm_vs_baseline_auroc_gap"] is not None for row in seed_rows),
        "aggregate": _aggregate_seed_rows(seed_rows),
        "seeds": seed_rows,
    }


def write_latent_trajectory_split_seed_sweep(
    summary: dict[str, Any],
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, str]:
    """Write LT split-seed robustness outputs."""

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = {"json": str(json_path)}
    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_latent_trajectory_csv(summary, csv_path)
        written["csv"] = str(csv_path)
    return written


def _evaluate_latent_trajectory_split_seed(
    records: list[TraceRecord],
    metric_rows: list[dict[str, object]],
    metric_rows_by_trace: dict[str, list[dict[str, object]]],
    seed: int,
    train_ratio: float,
    calibration_ratio: float,
    config: LatentTrajectoryConfig,
    use_model_selection: bool,
    calibration_mode: str,
) -> dict[str, Any]:
    splits = split_by_question(
        records,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        seed=seed,
    )
    train_metric_rows = [
        row
        for record in splits["train"]
        for row in metric_rows_by_trace.get(record.trace_id, [])
    ]
    scorer = _fit_trajectory_scorer_from_rows(train_metric_rows, config)
    scores_by_key = _score_metric_rows(metric_rows, scorer)
    rewritten = _rewrite_records(records, scores_by_key, config, scorer)
    rewritten_by_trace = {record.trace_id: record for record in rewritten}
    rewritten_splits = {
        split_name: [rewritten_by_trace[record.trace_id] for record in split_records]
        for split_name, split_records in splits.items()
    }

    hmm_config = None
    if use_model_selection:
        cal_fit, cal_val = _sub_split_calibration(rewritten_splits["calibration"], seed=seed)
        if cal_val:
            hmm_config = select_hmm_config(cal_fit, cal_val, calibration_mode=calibration_mode)

    row: dict[str, Any] = {
        "seed": seed,
        "split": {name: _split_counts(split_records) for name, split_records in rewritten_splits.items()},
        "transform": _scorer_to_dict(scorer),
        "lt_final_metrics": {
            name: _final_score_metrics(split_records)
            for name, split_records in rewritten_splits.items()
        },
        "selected_hmm_config": _config_to_dict(hmm_config),
    }
    try:
        methods = evaluate_trace_methods(
            rewritten_splits["calibration"],
            rewritten_splits["test"],
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


def _validate_config(config: LatentTrajectoryConfig) -> LatentTrajectoryConfig:
    if config.projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if not config.feature_field:
        raise ValueError("feature_field must be non-empty")
    metric = _normalize_metric_name(config.metric)
    return LatentTrajectoryConfig(
        projection_dim=config.projection_dim,
        metric=metric,
        normalize_vectors=bool(config.normalize_vectors),
        feature_field=config.feature_field,
    )


def _normalize_metric_name(metric: str) -> str:
    aliases = {
        "net": "net_change",
        "cumulative": "cumulative_change",
        "aligned": "aligned_change",
    }
    normalized = aliases.get(metric, metric)
    if normalized not in _METRICS:
        raise ValueError(f"metric must be one of {', '.join(_METRICS)}")
    return normalized


def _load_feature_rows(
    path: str | Path,
    required_keys: set[tuple[str, int]],
    labels_by_trace: dict[str, int],
    question_by_trace: dict[str, str],
    config: LatentTrajectoryConfig,
) -> list[_FeatureEntry]:
    entries: list[_FeatureEntry] = []
    seen: set[tuple[str, int]] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
                trace_id = str(row["trace_id"])
                step_index = int(row["step_index"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid hidden feature row at line {line_number}: {error}") from error
            key = (trace_id, step_index)
            if key not in required_keys:
                continue
            if key in seen:
                raise ValueError(f"duplicate hidden feature for trace_id={trace_id} step_index={step_index}")
            vector = row.get(config.feature_field)
            if not isinstance(vector, list):
                raise ValueError(
                    f"missing {config.feature_field} for trace_id={trace_id} step_index={step_index}"
                )
            entries.append(
                _FeatureEntry(
                    question_id=question_by_trace[trace_id],
                    trace_id=trace_id,
                    step_index=step_index,
                    label=labels_by_trace[trace_id],
                    vector=_project_vector(vector, config.projection_dim, normalize=config.normalize_vectors),
                )
            )
            seen.add(key)
    return entries


def _check_missing_features(required_keys: set[tuple[str, int]], feature_rows: list[_FeatureEntry]) -> None:
    missing = required_keys - {(entry.trace_id, entry.step_index) for entry in feature_rows}
    if missing:
        trace_id, step_index = sorted(missing)[0]
        raise ValueError(f"missing hidden feature for trace_id={trace_id} step_index={step_index}")


def _project_vector(vector: list[object], projection_dim: int, normalize: bool) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) <= projection_dim:
        projected = values + [0.0] * (projection_dim - len(values))
    else:
        projected = [0.0] * projection_dim
        for index, value in enumerate(values):
            bucket = ((index * 2654435761) & 0xFFFFFFFF) % projection_dim
            sign = 1.0 if (((index * 2246822519) & 0x2) == 0) else -1.0
            projected[bucket] += sign * value
    if normalize:
        norm = _norm(projected)
        if norm:
            return [value / norm for value in projected]
    return projected


def _entries_by_trace(feature_rows: list[_FeatureEntry]) -> dict[str, list[_FeatureEntry]]:
    grouped: dict[str, list[_FeatureEntry]] = {}
    for entry in feature_rows:
        grouped.setdefault(entry.trace_id, []).append(entry)
    for entries in grouped.values():
        entries.sort(key=lambda entry: entry.step_index)
    return grouped


def _fit_trajectory_scorer(entries: list[_FeatureEntry], config: LatentTrajectoryConfig) -> _TrajectoryScorer:
    if not entries:
        raise ValueError("training split has no hidden feature rows")
    return _fit_trajectory_scorer_from_rows(_prefix_metric_rows(_entries_by_trace(entries)), config)


def _fit_trajectory_scorer_from_rows(
    metric_rows: list[dict[str, object]],
    config: LatentTrajectoryConfig,
) -> _TrajectoryScorer:
    if not metric_rows:
        raise ValueError("training split has no hidden feature rows")
    components = {
        component: _fit_component_transform(
            component,
            [row["metrics"][component] for row in metric_rows],
            [int(row["label"]) for row in metric_rows],
        )
        for component in _COMPONENTS
    }
    return _TrajectoryScorer(metric=config.metric, components=components)


def _fit_component_transform(
    name: str,
    raw_values: list[float],
    labels: list[int],
) -> _ComponentTransform:
    if not raw_values:
        raise ValueError("raw_values must not be empty")
    raw_min = min(raw_values)
    raw_max = max(raw_values)
    train_auroc = None
    orientation = _DEFAULT_ORIENTATION[name]
    if raw_max > raw_min and len(set(labels)) == 2:
        normalized = [(value - raw_min) / (raw_max - raw_min) for value in raw_values]
        train_auroc = roc_auc_score(labels, normalized)
        orientation = -1 if train_auroc < 0.5 else 1
    oriented = [orientation * value for value in raw_values]
    return _ComponentTransform(
        name=name,
        orientation=orientation,
        minimum=min(oriented),
        maximum=max(oriented),
        train_auroc=train_auroc,
    )


def _score_feature_rows(
    entries_by_trace: dict[str, list[_FeatureEntry]],
    scorer: _TrajectoryScorer,
) -> dict[tuple[str, int], float]:
    return _score_metric_rows(_prefix_metric_rows(entries_by_trace), scorer)


def _score_metric_rows(
    metric_rows: list[dict[str, object]],
    scorer: _TrajectoryScorer,
) -> dict[tuple[str, int], float]:
    return {
        (str(row["trace_id"]), int(row["step_index"])): _score_metrics(row["metrics"], scorer)
        for row in metric_rows
    }


def _metric_rows_by_trace(metric_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row["trace_id"]), []).append(row)
    return grouped


def _prefix_metric_rows(entries_by_trace: dict[str, list[_FeatureEntry]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trace_id in sorted(entries_by_trace):
        entries = entries_by_trace[trace_id]
        first_vector: list[float] | None = None
        previous_vector: list[float] | None = None
        cumulative_change = 0.0
        normalized_update_sum: list[float] | None = None
        aligned_count = 0
        for prefix_index, entry in enumerate(entries, start=1):
            if first_vector is None:
                first_vector = entry.vector
                previous_vector = entry.vector
                normalized_update_sum = [0.0] * len(entry.vector)
                metrics = {"net_change": 0.0, "cumulative_change": 0.0, "aligned_change": 0.0}
            else:
                assert previous_vector is not None
                assert normalized_update_sum is not None
                update = _subtract(entry.vector, previous_vector)
                update_norm = _norm(update)
                cumulative_change += update_norm
                if update_norm > 0.0:
                    aligned_count += 1
                    for index, value in enumerate(update):
                        normalized_update_sum[index] += value / update_norm
                drift = _subtract(entry.vector, first_vector)
                drift_norm = _norm(drift)
                aligned_change = (
                    _dot(normalized_update_sum, drift) / (drift_norm * aligned_count)
                    if drift_norm > 0.0 and aligned_count > 0
                    else 0.0
                )
                metrics = {
                    "net_change": drift_norm / prefix_index,
                    "cumulative_change": cumulative_change,
                    "aligned_change": aligned_change,
                }
                previous_vector = entry.vector
            rows.append(
                {
                    "question_id": entry.question_id,
                    "trace_id": entry.trace_id,
                    "step_index": entry.step_index,
                    "label": entry.label,
                    "metrics": metrics,
                }
            )
    return rows


def _score_metrics(metrics: dict[str, float], scorer: _TrajectoryScorer) -> float:
    if scorer.metric == "composite":
        scores = [
            _score_component(metrics[component], scorer.components[component])
            for component in _COMPONENTS
        ]
        return sum(scores) / len(scores)
    return _score_component(metrics[scorer.metric], scorer.components[scorer.metric])


def _score_component(value: float, transform: _ComponentTransform) -> float:
    oriented = transform.orientation * value
    if transform.maximum <= transform.minimum:
        return 0.5
    scaled = (oriented - transform.minimum) / (transform.maximum - transform.minimum)
    return min(1.0, max(0.0, scaled))


def _rewrite_records(
    records: list[TraceRecord],
    scores_by_key: dict[tuple[str, int], float],
    config: LatentTrajectoryConfig,
    scorer: _TrajectoryScorer,
) -> list[TraceRecord]:
    rewritten: list[TraceRecord] = []
    for record in records:
        observations: list[Observation] = []
        for observation in record.observations:
            score = scores_by_key[(record.trace_id, observation.step_index)]
            observations.append(
                Observation(
                    step_index=observation.step_index,
                    text=observation.text,
                    score=score,
                    concept_code=observation.concept_code,
                    entropy=observation.entropy,
                    score_delta=observation.score_delta,
                )
            )
        metadata = dict(record.metadata)
        metadata["score_source"] = "latent_trajectory"
        metadata["latent_trajectory"] = {
            "metric": scorer.metric,
            "projection_dim": config.projection_dim,
            "normalize_vectors": config.normalize_vectors,
            "feature_field": config.feature_field,
            "components": _scorer_to_dict(scorer)["components"],
        }
        rewritten.append(
            TraceRecord(
                question_id=record.question_id,
                question=record.question,
                trace_id=record.trace_id,
                trace_text=record.trace_text,
                final_answer=record.final_answer,
                gold_answer=record.gold_answer,
                correct=record.correct,
                observations=observations,
                metadata=metadata,
            )
        )
    return rewritten


def _final_score_metrics(records: list[TraceRecord]) -> dict[str, object]:
    if not records:
        return {"records": 0, "metrics": None}
    labels = [1 if record.correct else 0 for record in records]
    scores = [float(record.observations[-1].score) for record in records]
    return {"records": len(records), "metrics": evaluate_predictions(labels, scores)}


def _scorer_to_dict(scorer: _TrajectoryScorer) -> dict[str, object]:
    return {
        "metric": scorer.metric,
        "components": {
            name: {
                "orientation": transform.orientation,
                "minimum": transform.minimum,
                "maximum": transform.maximum,
                "train_auroc": transform.train_auroc,
            }
            for name, transform in scorer.components.items()
        },
    }


def _write_latent_trajectory_csv(summary: dict[str, Any], path: Path) -> None:
    aggregate = summary.get("aggregate", {})
    fieldnames = [
        "seed",
        "train_correct",
        "train_incorrect",
        "calibration_correct",
        "calibration_incorrect",
        "test_correct",
        "test_incorrect",
        "lt_train_auroc",
        "lt_calibration_auroc",
        "lt_test_auroc",
        "best_baseline_method",
        "best_baseline_auroc",
        "best_online_hmm_method",
        "best_online_hmm_auroc",
        "hmm_vs_baseline_auroc_gap",
        "calibrated_last_step_brier",
        "hmm_score_brier",
        "ema_brier",
        "hmm_score_brier_minus_ema",
        "hmm_score_brier_minus_calibrated_last_step",
        "hmm_hybrid_auroc_minus_ema",
        "hmm_hybrid_auroc_minus_temporal_metric",
        "mean_hmm_score_brier",
        "mean_hmm_vs_baseline_auroc_gap",
        "mean_best_baseline_auroc",
        "mean_best_online_hmm_auroc",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.get("seeds", []):
            writer.writerow(
                {
                    "seed": row.get("seed"),
                    "train_correct": _nested(row, "split.train.correct"),
                    "train_incorrect": _nested(row, "split.train.incorrect"),
                    "calibration_correct": _nested(row, "split.calibration.correct"),
                    "calibration_incorrect": _nested(row, "split.calibration.incorrect"),
                    "test_correct": _nested(row, "split.test.correct"),
                    "test_incorrect": _nested(row, "split.test.incorrect"),
                    "lt_train_auroc": _nested(row, "lt_final_metrics.train.metrics.auroc"),
                    "lt_calibration_auroc": _nested(row, "lt_final_metrics.calibration.metrics.auroc"),
                    "lt_test_auroc": _nested(row, "lt_final_metrics.test.metrics.auroc"),
                    "best_baseline_method": row.get("best_baseline_method"),
                    "best_baseline_auroc": row.get("best_baseline_auroc"),
                    "best_online_hmm_method": row.get("best_online_hmm_method"),
                    "best_online_hmm_auroc": row.get("best_online_hmm_auroc"),
                    "hmm_vs_baseline_auroc_gap": row.get("hmm_vs_baseline_auroc_gap"),
                    "calibrated_last_step_brier": _nested(row, "method_metrics.calibrated_last_step.brier"),
                    "hmm_score_brier": _nested(row, "method_metrics.hmm_score.brier"),
                    "ema_brier": _nested(row, "method_metrics.ema.brier"),
                    "hmm_score_brier_minus_ema": row.get("hmm_score_brier_minus_ema"),
                    "hmm_score_brier_minus_calibrated_last_step": row.get(
                        "hmm_score_brier_minus_calibrated_last_step"
                    ),
                    "hmm_hybrid_auroc_minus_ema": row.get("hmm_hybrid_auroc_minus_ema"),
                    "hmm_hybrid_auroc_minus_temporal_metric": row.get(
                        "hmm_hybrid_auroc_minus_temporal_metric"
                    ),
                    "mean_hmm_score_brier": aggregate.get("mean_hmm_score_brier"),
                    "mean_hmm_vs_baseline_auroc_gap": aggregate.get("mean_hmm_vs_baseline_auroc_gap"),
                    "mean_best_baseline_auroc": aggregate.get("mean_best_baseline_auroc"),
                    "mean_best_online_hmm_auroc": aggregate.get("mean_best_online_hmm_auroc"),
                    "error": row.get("error"),
                }
            )


def _split_summary(records: list[TraceRecord]) -> dict[str, int]:
    return {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "correct": sum(1 for record in records if record.correct),
        "incorrect": sum(1 for record in records if not record.correct),
    }


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right, strict=True)]


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
