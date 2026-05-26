"""Train a lightweight hidden-state probe and rewrite trace scores."""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from csbf.evaluation import evaluate_predictions, evaluate_trace_methods
from csbf.experiment_diagnostics import BASELINE_METHODS, ONLINE_HMM_METHODS
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


@dataclass(frozen=True)
class HiddenProbeConfig:
    """Configuration for the stdlib hidden-state probe."""

    projection_dim: int = 128
    feature_fields: tuple[str, ...] = ("hidden_last_token",)
    epochs: int = 8
    learning_rate: float = 0.05
    l2: float = 1e-4
    calibration_epochs: int = 120
    include_entropy_feature: bool = False


@dataclass(frozen=True)
class _ProbeModel:
    weights: list[float]
    bias: float
    means: list[float]
    inv_stds: list[float]
    calibration_a: float
    calibration_b: float


@dataclass(frozen=True)
class _FeatureEntry:
    question_id: str
    trace_id: str
    step_index: int
    label: int
    features: list[float]


def build_hidden_probe_records(
    records_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    seed: int = 0,
    config: HiddenProbeConfig | None = None,
) -> dict[str, Any]:
    """Train a question-split hidden probe and rewrite observation scores.

    The probe is fit on train questions, Platt-calibrated on calibration
    questions, and applied to all records. The resulting JSONL can be evaluated
    by the existing pre-observed pipeline.
    """

    cfg = config or HiddenProbeConfig()
    _validate_config(cfg)
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
    feature_rows = _load_projected_feature_rows(
        features_path,
        required_keys=required_keys,
        labels_by_trace=labels_by_trace,
        question_by_trace=question_by_trace,
        config=cfg,
    )
    missing = required_keys - {(entry.trace_id, entry.step_index) for entry in feature_rows}
    if missing:
        trace_id, step_index = sorted(missing)[0]
        raise ValueError(f"missing hidden feature for trace_id={trace_id} step_index={step_index}")

    train_entries = [entry for entry in feature_rows if split_by_trace[entry.trace_id] == "train"]
    calibration_entries = [entry for entry in feature_rows if split_by_trace[entry.trace_id] == "calibration"]
    test_entries = [entry for entry in feature_rows if split_by_trace[entry.trace_id] == "test"]
    model = _fit_probe(train_entries, calibration_entries, cfg, seed=seed)
    scores_by_key = {
        (entry.trace_id, entry.step_index): _predict_probability(model, entry.features)
        for entry in feature_rows
    }
    rewritten = _rewrite_records(records, scores_by_key, cfg)
    save_jsonl(rewritten, output_path)

    summary = {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "split": {name: _split_summary(split_records) for name, split_records in splits.items()},
        "feature_rows": {
            "required": len(required_keys),
            "used": len(feature_rows),
            "projection_dim": cfg.projection_dim,
            "feature_fields": list(cfg.feature_fields),
            "projected_dim": cfg.projection_dim * len(cfg.feature_fields),
            "include_entropy_feature": cfg.include_entropy_feature,
        },
        "probe": {
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "l2": cfg.l2,
            "calibration_epochs": cfg.calibration_epochs,
        },
        "prefix_metrics": {
            "train": _entry_metrics(model, train_entries),
            "calibration": _entry_metrics(model, calibration_entries),
            "test": _entry_metrics(model, test_entries),
        },
        "outputs": {"records": str(output_path)},
    }
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        summary["outputs"]["report"] = str(report)
        report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_hidden_probe_split_seed_sweep(
    records_path: str | Path,
    features_path: str | Path,
    seeds: Iterable[int],
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    config: HiddenProbeConfig | None = None,
    use_model_selection: bool = False,
    calibration_mode: str = "all_steps",
) -> dict[str, Any]:
    """Retrain/recalibrate a hidden probe for each question split seed.

    This is the leakage-clean robustness check for hidden-probe observations:
    each seed fits the probe only on that seed's train questions, calibrates it
    on that seed's calibration questions, then evaluates HMM methods on the
    held-out test questions.
    """

    cfg = config or HiddenProbeConfig()
    _validate_config(cfg)
    records = load_jsonl(records_path)
    labels_by_trace = {record.trace_id: 1 if record.correct else 0 for record in records}
    question_by_trace = {record.trace_id: record.question_id for record in records}
    required_keys = {
        (record.trace_id, observation.step_index)
        for record in records
        for observation in record.observations
    }
    feature_rows = _load_projected_feature_rows(
        features_path,
        required_keys=required_keys,
        labels_by_trace=labels_by_trace,
        question_by_trace=question_by_trace,
        config=cfg,
    )
    missing = required_keys - {(entry.trace_id, entry.step_index) for entry in feature_rows}
    if missing:
        trace_id, step_index = sorted(missing)[0]
        raise ValueError(f"missing hidden feature for trace_id={trace_id} step_index={step_index}")

    entries_by_trace: dict[str, list[_FeatureEntry]] = {}
    for entry in feature_rows:
        entries_by_trace.setdefault(entry.trace_id, []).append(entry)

    seed_rows = [
        _evaluate_probe_split_seed(
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
            "required": len(required_keys),
            "used": len(feature_rows),
            "projection_dim": cfg.projection_dim,
            "feature_fields": list(cfg.feature_fields),
            "projected_dim": cfg.projection_dim * len(cfg.feature_fields),
            "include_entropy_feature": cfg.include_entropy_feature,
        },
        "probe": {
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


def write_hidden_probe_split_seed_sweep(
    summary: dict[str, Any],
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, str]:
    """Write hidden-probe split-seed robustness outputs."""

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = {"json": str(json_path)}
    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_probe_split_seed_csv(summary, csv_path)
        written["csv"] = str(csv_path)
    return written


def _evaluate_probe_split_seed(
    records: list[TraceRecord],
    entries_by_trace: dict[str, list[_FeatureEntry]],
    seed: int,
    train_ratio: float,
    calibration_ratio: float,
    config: HiddenProbeConfig,
    use_model_selection: bool,
    calibration_mode: str,
) -> dict[str, Any]:
    splits = split_by_question(
        records,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        seed=seed,
    )
    split_entries = {
        split_name: [
            entry
            for record in split_records
            for entry in entries_by_trace.get(record.trace_id, [])
        ]
        for split_name, split_records in splits.items()
    }
    model = _fit_probe(split_entries["train"], split_entries["calibration"], config, seed=seed)
    all_entries = [
        entry
        for split_name in ("train", "calibration", "test")
        for entry in split_entries[split_name]
    ]
    scores_by_key = {
        (entry.trace_id, entry.step_index): _predict_probability(model, entry.features)
        for entry in all_entries
    }
    rewritten = _rewrite_records(records, scores_by_key, config)
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
        "probe_prefix_metrics": {
            name: _entry_metrics(model, split_entries[name])
            for name in ("train", "calibration", "test")
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


def _write_probe_split_seed_csv(summary: dict[str, Any], path: Path) -> None:
    aggregate = summary.get("aggregate", {})
    fieldnames = [
        "seed",
        "train_correct",
        "train_incorrect",
        "calibration_correct",
        "calibration_incorrect",
        "test_correct",
        "test_incorrect",
        "probe_train_auroc",
        "probe_calibration_auroc",
        "probe_test_auroc",
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
                    "probe_train_auroc": _nested(row, "probe_prefix_metrics.train.metrics.auroc"),
                    "probe_calibration_auroc": _nested(row, "probe_prefix_metrics.calibration.metrics.auroc"),
                    "probe_test_auroc": _nested(row, "probe_prefix_metrics.test.metrics.auroc"),
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


def _validate_config(config: HiddenProbeConfig) -> None:
    if config.projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if not config.feature_fields:
        raise ValueError("feature_fields must not be empty")
    if any(not field for field in config.feature_fields):
        raise ValueError("feature_fields must not contain empty field names")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    if config.calibration_epochs < 0:
        raise ValueError("calibration_epochs must be non-negative")


def _load_projected_feature_rows(
    path: str | Path,
    required_keys: set[tuple[str, int]],
    labels_by_trace: dict[str, int],
    question_by_trace: dict[str, str],
    config: HiddenProbeConfig,
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
            features: list[float] = []
            for field_name in config.feature_fields:
                vector = row.get(field_name)
                if not isinstance(vector, list):
                    raise ValueError(
                        f"missing hidden feature field {field_name} "
                        f"for trace_id={trace_id} step_index={step_index}"
                    )
                features.extend(_project_hidden_vector(vector, config.projection_dim))
            if config.include_entropy_feature:
                entropy = 0.0 if row.get("entropy") is None else float(row["entropy"])
                features.append(entropy)
            entries.append(
                _FeatureEntry(
                    question_id=question_by_trace[trace_id],
                    trace_id=trace_id,
                    step_index=step_index,
                    label=labels_by_trace[trace_id],
                    features=features,
                )
            )
            seen.add(key)
    return entries


def _project_hidden_vector(vector: list[object], projection_dim: int) -> list[float]:
    projected = [0.0] * projection_dim
    for index, value in enumerate(vector):
        bucket = ((index * 2654435761) & 0xFFFFFFFF) % projection_dim
        sign = 1.0 if (((index * 2246822519) & 0x2) == 0) else -1.0
        projected[bucket] += sign * float(value)
    norm = math.sqrt(sum(value * value for value in projected))
    if norm:
        return [value / norm for value in projected]
    return projected


def _fit_probe(
    train_entries: list[_FeatureEntry],
    calibration_entries: list[_FeatureEntry],
    config: HiddenProbeConfig,
    seed: int,
) -> _ProbeModel:
    if not train_entries:
        raise ValueError("training split has no hidden feature rows")
    dim = len(train_entries[0].features)
    means, inv_stds = _fit_standardizer([entry.features for entry in train_entries])
    weights = [0.0] * dim
    bias = _initial_bias([entry.label for entry in train_entries])
    rng = random.Random(seed)
    indices = list(range(len(train_entries)))
    for epoch in range(config.epochs):
        rng.shuffle(indices)
        rate = config.learning_rate / math.sqrt(epoch + 1.0)
        for index in indices:
            entry = train_entries[index]
            features = _standardize(entry.features, means, inv_stds)
            logit = _dot(weights, features) + bias
            error = _sigmoid(logit) - entry.label
            for i, value in enumerate(features):
                weights[i] -= rate * (error * value + config.l2 * weights[i])
            bias -= rate * error

    calibration_logits = [
        _dot(weights, _standardize(entry.features, means, inv_stds)) + bias
        for entry in calibration_entries
    ]
    calibration_labels = [entry.label for entry in calibration_entries]
    calibration_a, calibration_b = _fit_platt_scaler(
        calibration_logits,
        calibration_labels,
        epochs=config.calibration_epochs,
        learning_rate=config.learning_rate,
        seed=seed,
    )
    return _ProbeModel(
        weights=weights,
        bias=bias,
        means=means,
        inv_stds=inv_stds,
        calibration_a=calibration_a,
        calibration_b=calibration_b,
    )


def _fit_standardizer(features: list[list[float]]) -> tuple[list[float], list[float]]:
    dim = len(features[0])
    means = [0.0] * dim
    for row in features:
        for index, value in enumerate(row):
            means[index] += value
    means = [value / len(features) for value in means]
    variances = [0.0] * dim
    for row in features:
        for index, value in enumerate(row):
            diff = value - means[index]
            variances[index] += diff * diff
    inv_stds = []
    for variance in variances:
        std = math.sqrt(variance / len(features))
        inv_stds.append(0.0 if std == 0.0 else 1.0 / std)
    return means, inv_stds


def _fit_platt_scaler(
    logits: list[float],
    labels: list[int],
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[float, float]:
    if not logits or len(set(labels)) < 2 or epochs == 0:
        return 1.0, 0.0
    a = 1.0
    b = _initial_bias(labels)
    indices = list(range(len(logits)))
    rng = random.Random(seed + 17)
    for epoch in range(epochs):
        rng.shuffle(indices)
        rate = learning_rate / math.sqrt(epoch + 1.0)
        for index in indices:
            logit = logits[index]
            error = _sigmoid(a * logit + b) - labels[index]
            a -= rate * error * logit
            b -= rate * error
    return a, b


def _rewrite_records(
    records: list[TraceRecord],
    scores_by_key: dict[tuple[str, int], float],
    config: HiddenProbeConfig,
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
        metadata["score_source"] = "hidden_probe"
        metadata["hidden_probe"] = {
            "projection_dim": config.projection_dim,
            "feature_fields": list(config.feature_fields),
            "include_entropy_feature": config.include_entropy_feature,
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


def _entry_metrics(model: _ProbeModel, entries: list[_FeatureEntry]) -> dict[str, object]:
    if not entries:
        return {"rows": 0, "metrics": None}
    labels = [entry.label for entry in entries]
    probabilities = [_predict_probability(model, entry.features) for entry in entries]
    return {
        "rows": len(entries),
        "metrics": evaluate_predictions(labels, probabilities),
    }


def _split_summary(records: list[TraceRecord]) -> dict[str, int]:
    return {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "correct": sum(1 for record in records if record.correct),
        "incorrect": sum(1 for record in records if not record.correct),
    }


def _predict_probability(model: _ProbeModel, features: list[float]) -> float:
    standardized = _standardize(features, model.means, model.inv_stds)
    raw_logit = _dot(model.weights, standardized) + model.bias
    return _sigmoid(model.calibration_a * raw_logit + model.calibration_b)


def _standardize(features: list[float], means: list[float], inv_stds: list[float]) -> list[float]:
    return [
        (value - means[index]) * inv_stds[index]
        for index, value in enumerate(features)
    ]


def _initial_bias(labels: list[int]) -> float:
    positives = sum(labels) + 0.5
    negatives = len(labels) - sum(labels) + 0.5
    return math.log(positives / negatives)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _sigmoid(value: float) -> float:
    if value >= 35.0:
        return 1.0 - 1e-15
    if value <= -35.0:
        return 1e-15
    return 1.0 / (1.0 + math.exp(-value))
