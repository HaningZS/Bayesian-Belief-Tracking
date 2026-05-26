"""Probe-logit trajectory diagnostics for hidden-state observations."""

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
from csbf.hidden_probe import (
    HiddenProbeConfig,
    _FeatureEntry,
    _dot,
    _fit_probe,
    _load_projected_feature_rows,
    _standardize,
)
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

_COMPONENTS = ("probe_logit", "logit_delta", "cumulative_logit_change", "aligned_logit_change")
_METRICS = (*_COMPONENTS, "composite")


@dataclass(frozen=True)
class ProbeTrajectoryConfig:
    """Configuration for probe-logit trajectory scoring."""

    projection_dim: int = 128
    epochs: int = 8
    learning_rate: float = 0.05
    l2: float = 1e-4
    calibration_epochs: int = 120
    include_entropy_feature: bool = False
    metric: str = "logit_delta"


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


def build_probe_trajectory_records(
    records_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    seed: int = 0,
    config: ProbeTrajectoryConfig | None = None,
) -> dict[str, Any]:
    """Fit a hidden probe, score probe-logit dynamics, and rewrite records."""

    cfg = _validate_config(config or ProbeTrajectoryConfig())
    records = load_jsonl(records_path)
    splits = split_by_question(records, train_ratio=train_ratio, calibration_ratio=calibration_ratio, seed=seed)
    split_by_trace = {
        record.trace_id: split_name
        for split_name, split_records in splits.items()
        for record in split_records
    }
    entries_by_trace = _load_entries_by_trace(records, features_path, cfg)
    split_entries = {
        split_name: [
            entry
            for record in split_records
            for entry in entries_by_trace.get(record.trace_id, [])
        ]
        for split_name, split_records in splits.items()
    }
    model = _fit_probe(
        split_entries["train"],
        split_entries["calibration"],
        _hidden_probe_config(cfg),
        seed=seed,
    )
    trajectory_rows = _probe_trajectory_rows(entries_by_trace, model)
    train_trace_ids = {trace_id for trace_id, split_name in split_by_trace.items() if split_name == "train"}
    scorer = _fit_trajectory_scorer(
        [row for row in trajectory_rows if str(row["trace_id"]) in train_trace_ids],
        cfg,
    )
    scores_by_key = _score_rows(trajectory_rows, scorer)
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
            "required": sum(len(record.observations) for record in records),
            "used": sum(len(entries) for entries in entries_by_trace.values()),
            "projection_dim": cfg.projection_dim,
            "include_entropy_feature": cfg.include_entropy_feature,
        },
        "probe_trajectory": _scorer_to_dict(scorer),
        "probe_trajectory_final_metrics": {
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


def run_probe_trajectory_split_seed_sweep(
    records_path: str | Path,
    features_path: str | Path,
    seeds: Iterable[int],
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    config: ProbeTrajectoryConfig | None = None,
    use_model_selection: bool = False,
    calibration_mode: str = "all_steps",
) -> dict[str, Any]:
    """Refit hidden probe and probe-logit trajectory transform per split seed."""

    cfg = _validate_config(config or ProbeTrajectoryConfig())
    records = load_jsonl(records_path)
    entries_by_trace = _load_entries_by_trace(records, features_path, cfg)
    seed_rows = [
        _evaluate_probe_trajectory_split_seed(
            records,
            entries_by_trace=entries_by_trace,
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
            "required": sum(len(record.observations) for record in records),
            "used": sum(len(entries) for entries in entries_by_trace.values()),
            "projection_dim": cfg.projection_dim,
            "include_entropy_feature": cfg.include_entropy_feature,
        },
        "probe_trajectory": {
            "metric": cfg.metric,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "l2": cfg.l2,
            "calibration_epochs": cfg.calibration_epochs,
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


def write_probe_trajectory_split_seed_sweep(
    summary: dict[str, Any],
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, str]:
    """Write probe-trajectory split-seed robustness outputs."""

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = {"json": str(json_path)}
    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_probe_trajectory_csv(summary, csv_path)
        written["csv"] = str(csv_path)
    return written


def _evaluate_probe_trajectory_split_seed(
    records: list[TraceRecord],
    entries_by_trace: dict[str, list[_FeatureEntry]],
    seed: int,
    train_ratio: float,
    calibration_ratio: float,
    config: ProbeTrajectoryConfig,
    use_model_selection: bool,
    calibration_mode: str,
) -> dict[str, Any]:
    splits = split_by_question(records, train_ratio=train_ratio, calibration_ratio=calibration_ratio, seed=seed)
    split_entries = {
        split_name: [
            entry
            for record in split_records
            for entry in entries_by_trace.get(record.trace_id, [])
        ]
        for split_name, split_records in splits.items()
    }
    model = _fit_probe(
        split_entries["train"],
        split_entries["calibration"],
        _hidden_probe_config(config),
        seed=seed,
    )
    trajectory_rows = _probe_trajectory_rows(entries_by_trace, model)
    train_trace_ids = {record.trace_id for record in splits["train"]}
    scorer = _fit_trajectory_scorer(
        [row for row in trajectory_rows if str(row["trace_id"]) in train_trace_ids],
        config,
    )
    scores_by_key = _score_rows(trajectory_rows, scorer)
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
        "probe_trajectory_final_metrics": {
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


def _load_entries_by_trace(
    records: list[TraceRecord],
    features_path: str | Path,
    config: ProbeTrajectoryConfig,
) -> dict[str, list[_FeatureEntry]]:
    labels_by_trace = {record.trace_id: 1 if record.correct else 0 for record in records}
    question_by_trace = {record.trace_id: record.question_id for record in records}
    required_keys = {
        (record.trace_id, observation.step_index)
        for record in records
        for observation in record.observations
    }
    entries = _load_projected_feature_rows(
        features_path,
        required_keys=required_keys,
        labels_by_trace=labels_by_trace,
        question_by_trace=question_by_trace,
        config=_hidden_probe_config(config),
    )
    missing = required_keys - {(entry.trace_id, entry.step_index) for entry in entries}
    if missing:
        trace_id, step_index = sorted(missing)[0]
        raise ValueError(f"missing hidden feature for trace_id={trace_id} step_index={step_index}")
    grouped: dict[str, list[_FeatureEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.trace_id, []).append(entry)
    for trace_entries in grouped.values():
        trace_entries.sort(key=lambda entry: entry.step_index)
    return grouped


def _probe_trajectory_rows(
    entries_by_trace: dict[str, list[_FeatureEntry]],
    model,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trace_id in sorted(entries_by_trace):
        entries = entries_by_trace[trace_id]
        first_logit: float | None = None
        previous_logit: float | None = None
        cumulative_change = 0.0
        normalized_update_sum = 0.0
        aligned_count = 0
        for entry in entries:
            logit = _predict_calibrated_logit(model, entry.features)
            if first_logit is None:
                first_logit = logit
                previous_logit = logit
                metrics = {
                    "probe_logit": logit,
                    "logit_delta": 0.0,
                    "cumulative_logit_change": 0.0,
                    "aligned_logit_change": 0.0,
                }
            else:
                assert previous_logit is not None
                delta = logit - previous_logit
                cumulative_change += abs(delta)
                if delta != 0.0:
                    aligned_count += 1
                    normalized_update_sum += 1.0 if delta > 0.0 else -1.0
                drift = logit - first_logit
                aligned = (
                    normalized_update_sum * (1.0 if drift > 0.0 else -1.0) / aligned_count
                    if drift != 0.0 and aligned_count > 0
                    else 0.0
                )
                metrics = {
                    "probe_logit": logit,
                    "logit_delta": delta,
                    "cumulative_logit_change": cumulative_change,
                    "aligned_logit_change": aligned,
                }
                previous_logit = logit
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


def _fit_trajectory_scorer(
    metric_rows: list[dict[str, object]],
    config: ProbeTrajectoryConfig,
) -> _TrajectoryScorer:
    if not metric_rows:
        raise ValueError("training split has no probe trajectory rows")
    components = {
        component: _fit_component_transform(
            component,
            [row["metrics"][component] for row in metric_rows],
            [int(row["label"]) for row in metric_rows],
        )
        for component in _COMPONENTS
    }
    return _TrajectoryScorer(metric=config.metric, components=components)


def _fit_component_transform(name: str, raw_values: list[float], labels: list[int]) -> _ComponentTransform:
    raw_min = min(raw_values)
    raw_max = max(raw_values)
    orientation = 1
    train_auroc = None
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


def _score_rows(
    metric_rows: list[dict[str, object]],
    scorer: _TrajectoryScorer,
) -> dict[tuple[str, int], float]:
    return {
        (str(row["trace_id"]), int(row["step_index"])): _score_metrics(row["metrics"], scorer)
        for row in metric_rows
    }


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
    config: ProbeTrajectoryConfig,
    scorer: _TrajectoryScorer,
) -> list[TraceRecord]:
    rewritten: list[TraceRecord] = []
    for record in records:
        observations: list[Observation] = []
        for observation in record.observations:
            observations.append(
                Observation(
                    step_index=observation.step_index,
                    text=observation.text,
                    score=scores_by_key[(record.trace_id, observation.step_index)],
                    concept_code=observation.concept_code,
                    entropy=observation.entropy,
                    score_delta=observation.score_delta,
                )
            )
        metadata = dict(record.metadata)
        metadata["score_source"] = "probe_trajectory"
        metadata["probe_trajectory"] = {
            "metric": scorer.metric,
            "projection_dim": config.projection_dim,
            "include_entropy_feature": config.include_entropy_feature,
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


def _write_probe_trajectory_csv(summary: dict[str, Any], path: Path) -> None:
    aggregate = summary.get("aggregate", {})
    fieldnames = [
        "seed",
        "train_correct",
        "train_incorrect",
        "calibration_correct",
        "calibration_incorrect",
        "test_correct",
        "test_incorrect",
        "probe_traj_train_auroc",
        "probe_traj_calibration_auroc",
        "probe_traj_test_auroc",
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
                    "probe_traj_train_auroc": _nested(
                        row, "probe_trajectory_final_metrics.train.metrics.auroc"
                    ),
                    "probe_traj_calibration_auroc": _nested(
                        row, "probe_trajectory_final_metrics.calibration.metrics.auroc"
                    ),
                    "probe_traj_test_auroc": _nested(
                        row, "probe_trajectory_final_metrics.test.metrics.auroc"
                    ),
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


def _validate_config(config: ProbeTrajectoryConfig) -> ProbeTrajectoryConfig:
    if config.projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    if config.calibration_epochs < 0:
        raise ValueError("calibration_epochs must be non-negative")
    metric = _normalize_metric_name(config.metric)
    return ProbeTrajectoryConfig(
        projection_dim=config.projection_dim,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l2=config.l2,
        calibration_epochs=config.calibration_epochs,
        include_entropy_feature=bool(config.include_entropy_feature),
        metric=metric,
    )


def _normalize_metric_name(metric: str) -> str:
    aliases = {
        "logit": "probe_logit",
        "delta": "logit_delta",
        "cumulative": "cumulative_logit_change",
        "aligned": "aligned_logit_change",
    }
    normalized = aliases.get(metric, metric)
    if normalized not in _METRICS:
        raise ValueError(f"metric must be one of {', '.join(_METRICS)}")
    return normalized


def _hidden_probe_config(config: ProbeTrajectoryConfig) -> HiddenProbeConfig:
    return HiddenProbeConfig(
        projection_dim=config.projection_dim,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l2=config.l2,
        calibration_epochs=config.calibration_epochs,
        include_entropy_feature=config.include_entropy_feature,
    )


def _predict_calibrated_logit(model, features: list[float]) -> float:
    standardized = _standardize(features, model.means, model.inv_stds)
    raw_logit = _dot(model.weights, standardized) + model.bias
    return model.calibration_a * raw_logit + model.calibration_b


def _split_summary(records: list[TraceRecord]) -> dict[str, int]:
    return {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "correct": sum(1 for record in records if record.correct),
        "incorrect": sum(1 for record in records if not record.correct),
    }
