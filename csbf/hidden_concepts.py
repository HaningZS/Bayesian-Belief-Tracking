"""Rewrite concept codes with train-split hidden-state clusters."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from csbf.concepts import self_verification_concept_code, text_concept_code
from csbf.evaluation import evaluate_trace_methods
from csbf.experiment_diagnostics import BASELINE_METHODS, ONLINE_HMM_METHODS
from csbf.hidden_probe import _project_hidden_vector
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
    _split_counts,
    _subtract_or_none,
    _write_csv,
)


@dataclass(frozen=True)
class HiddenClusterConfig:
    """Configuration for deterministic hidden-vector k-means concepts."""

    cluster_count: int = 16
    projection_dim: int = 64
    iterations: int = 20
    feature_field: str = "hidden_last_token"
    include_entropy_feature: bool = False
    text_concept_mode: str | None = None


@dataclass(frozen=True)
class _ClusterEntry:
    question_id: str
    trace_id: str
    step_index: int
    features: list[float]


def build_hidden_cluster_records(
    records_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    seed: int = 0,
    config: HiddenClusterConfig | None = None,
) -> dict[str, Any]:
    """Fit hidden-vector clusters on train questions and rewrite concept codes."""

    cfg = config or HiddenClusterConfig()
    _validate_config(cfg)
    records = load_jsonl(records_path)
    splits = split_by_question(records, train_ratio=train_ratio, calibration_ratio=calibration_ratio, seed=seed)
    split_by_trace = {
        record.trace_id: split_name
        for split_name, split_records in splits.items()
        for record in split_records
    }
    question_by_trace = {record.trace_id: record.question_id for record in records}
    required_keys = {
        (record.trace_id, observation.step_index)
        for record in records
        for observation in record.observations
    }
    feature_rows = _load_projected_feature_rows(
        features_path,
        required_keys=required_keys,
        question_by_trace=question_by_trace,
        config=cfg,
    )
    missing = required_keys - {(entry.trace_id, entry.step_index) for entry in feature_rows}
    if missing:
        trace_id, step_index = sorted(missing)[0]
        raise ValueError(f"missing hidden feature for trace_id={trace_id} step_index={step_index}")

    train_entries = [entry for entry in feature_rows if split_by_trace[entry.trace_id] == "train"]
    centers = _fit_kmeans([entry.features for entry in train_entries], cfg.cluster_count, cfg.iterations)
    assignments = {
        (entry.trace_id, entry.step_index): _nearest_center(entry.features, centers)
        for entry in feature_rows
    }
    rewritten = _rewrite_records(records, assignments, cfg, effective_cluster_count=len(centers))
    save_jsonl(rewritten, output_path)

    summary = {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "split": {name: _split_summary(split_records) for name, split_records in splits.items()},
        "feature_rows": {
            "required": len(required_keys),
            "used": len(feature_rows),
            "projection_dim": cfg.projection_dim,
            "feature_field": cfg.feature_field,
            "include_entropy_feature": cfg.include_entropy_feature,
        },
        "clustering": {
            "requested_cluster_count": cfg.cluster_count,
            "effective_cluster_count": len(centers),
            "iterations": cfg.iterations,
            "trained_on_rows": len(train_entries),
        },
        "concept_join": _concept_join_summary(cfg),
        "cluster_usage": _cluster_usage(feature_rows, assignments, split_by_trace),
        "outputs": {"records": str(output_path)},
    }
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        summary["outputs"]["report"] = str(report)
        report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_hidden_cluster_split_seed_sweep(
    records_path: str | Path,
    features_path: str | Path,
    seeds: Iterable[int],
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    config: HiddenClusterConfig | None = None,
    use_model_selection: bool = False,
    calibration_mode: str = "all_steps",
) -> dict[str, Any]:
    """Refit hidden clusters per split seed before HMM evaluation."""

    cfg = config or HiddenClusterConfig()
    _validate_config(cfg)
    records = load_jsonl(records_path)
    question_by_trace = {record.trace_id: record.question_id for record in records}
    required_keys = {
        (record.trace_id, observation.step_index)
        for record in records
        for observation in record.observations
    }
    feature_rows = _load_projected_feature_rows(
        features_path,
        required_keys=required_keys,
        question_by_trace=question_by_trace,
        config=cfg,
    )
    missing = required_keys - {(entry.trace_id, entry.step_index) for entry in feature_rows}
    if missing:
        trace_id, step_index = sorted(missing)[0]
        raise ValueError(f"missing hidden feature for trace_id={trace_id} step_index={step_index}")

    entries_by_trace: dict[str, list[_ClusterEntry]] = {}
    for entry in feature_rows:
        entries_by_trace.setdefault(entry.trace_id, []).append(entry)

    seed_rows = [
        _evaluate_cluster_split_seed(
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
            "feature_field": cfg.feature_field,
            "include_entropy_feature": cfg.include_entropy_feature,
        },
        "clustering": {
            "requested_cluster_count": cfg.cluster_count,
            "iterations": cfg.iterations,
        },
        "concept_join": _concept_join_summary(cfg),
        "seed_count": len(seed_rows),
        "train_ratio": train_ratio,
        "calibration_ratio": calibration_ratio,
        "use_model_selection": bool(use_model_selection),
        "calibration_mode": calibration_mode,
        "valid_auroc_seed_count": sum(row["hmm_vs_baseline_auroc_gap"] is not None for row in seed_rows),
        "aggregate": _aggregate_seed_rows(seed_rows),
        "seeds": seed_rows,
    }


def write_hidden_cluster_split_seed_sweep(
    summary: dict[str, Any],
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, str]:
    """Write hidden-cluster split-seed robustness outputs."""

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


def _evaluate_cluster_split_seed(
    records: list[TraceRecord],
    entries_by_trace: dict[str, list[_ClusterEntry]],
    seed: int,
    train_ratio: float,
    calibration_ratio: float,
    config: HiddenClusterConfig,
    use_model_selection: bool,
    calibration_mode: str,
) -> dict[str, Any]:
    splits = split_by_question(
        records,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        seed=seed,
    )
    split_by_trace = {
        record.trace_id: split_name
        for split_name, split_records in splits.items()
        for record in split_records
    }
    split_entries = {
        split_name: [
            entry
            for record in split_records
            for entry in entries_by_trace.get(record.trace_id, [])
        ]
        for split_name, split_records in splits.items()
    }
    centers = _fit_kmeans(
        [entry.features for entry in split_entries["train"]],
        config.cluster_count,
        config.iterations,
    )
    all_entries = [
        entry
        for split_name in ("train", "calibration", "test")
        for entry in split_entries[split_name]
    ]
    assignments = {
        (entry.trace_id, entry.step_index): _nearest_center(entry.features, centers)
        for entry in all_entries
    }
    rewritten = _rewrite_records(records, assignments, config, effective_cluster_count=len(centers))
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
        "cluster_usage": _cluster_usage(all_entries, assignments, split_by_trace),
        "concept_join": _concept_join_summary(config),
        "effective_cluster_count": len(centers),
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


def _validate_config(config: HiddenClusterConfig) -> None:
    if config.cluster_count <= 0:
        raise ValueError("cluster_count must be positive")
    if config.projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if not config.feature_field:
        raise ValueError("feature_field must be non-empty")
    if config.text_concept_mode not in {None, "text_pattern", "self_verification"}:
        raise ValueError("text_concept_mode must be None, text_pattern, or self_verification")


def _load_projected_feature_rows(
    path: str | Path,
    required_keys: set[tuple[str, int]],
    question_by_trace: dict[str, str],
    config: HiddenClusterConfig,
) -> list[_ClusterEntry]:
    entries: list[_ClusterEntry] = []
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
            features = _project_hidden_vector(vector, config.projection_dim)
            if config.include_entropy_feature:
                entropy = 0.0 if row.get("entropy") is None else float(row["entropy"])
                features.append(entropy)
            entries.append(
                _ClusterEntry(
                    question_id=question_by_trace[trace_id],
                    trace_id=trace_id,
                    step_index=step_index,
                    features=features,
                )
            )
            seen.add(key)
    return entries


def _fit_kmeans(features: list[list[float]], cluster_count: int, iterations: int) -> list[list[float]]:
    if not features:
        raise ValueError("training split has no hidden feature rows")
    effective_count = min(cluster_count, len(features))
    centers = _initial_centers(features, effective_count)
    for _ in range(iterations):
        groups = [[] for _ in centers]
        for row in features:
            groups[_nearest_center(row, centers)].append(row)
        centers = [
            _mean_vector(group) if group else centers[index]
            for index, group in enumerate(groups)
        ]
    return centers


def _initial_centers(features: list[list[float]], cluster_count: int) -> list[list[float]]:
    ordered = sorted(features)
    if cluster_count == 1:
        return [list(ordered[0])]
    last_index = len(ordered) - 1
    indexes = [
        min(last_index, int(round(index * last_index / (cluster_count - 1))))
        for index in range(cluster_count)
    ]
    return [list(ordered[index]) for index in indexes]


def _nearest_center(features: list[float], centers: list[list[float]]) -> int:
    distances = [_squared_distance(features, center) for center in centers]
    return min(range(len(distances)), key=lambda index: distances[index])


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right, strict=True))


def _mean_vector(rows: list[list[float]]) -> list[float]:
    dim = len(rows[0])
    totals = [0.0] * dim
    for row in rows:
        for index, value in enumerate(row):
            totals[index] += value
    return [value / len(rows) for value in totals]


def _rewrite_records(
    records: list[TraceRecord],
    assignments: dict[tuple[str, int], int],
    config: HiddenClusterConfig,
    effective_cluster_count: int,
) -> list[TraceRecord]:
    rewritten: list[TraceRecord] = []
    for record in records:
        observations: list[Observation] = []
        for observation in record.observations:
            cluster = assignments[(record.trace_id, observation.step_index)]
            concept_code = _concept_code(cluster, observation, config)
            observations.append(
                Observation(
                    step_index=observation.step_index,
                    text=observation.text,
                    score=observation.score,
                    concept_code=concept_code,
                    entropy=observation.entropy,
                    score_delta=observation.score_delta,
                )
            )
        metadata = dict(record.metadata)
        metadata["concept_source"] = _concept_source(config)
        metadata["hidden_clusters"] = {
            "cluster_count": effective_cluster_count,
            "projection_dim": config.projection_dim,
            "include_entropy_feature": config.include_entropy_feature,
            "text_concept_mode": config.text_concept_mode,
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


def _concept_code(cluster: int, observation: Observation, config: HiddenClusterConfig) -> str:
    cluster_code = f"hidden_cluster_{cluster}"
    if config.text_concept_mode is None:
        return cluster_code
    if config.text_concept_mode == "text_pattern":
        return f"{cluster_code}|{text_concept_code(observation.text)}"
    if config.text_concept_mode == "self_verification":
        return f"{cluster_code}|{self_verification_concept_code(observation.text)}"
    raise ValueError(f"unsupported text_concept_mode: {config.text_concept_mode}")


def _concept_source(config: HiddenClusterConfig) -> str:
    if config.text_concept_mode is None:
        return "hidden_cluster"
    return f"hidden_cluster+{config.text_concept_mode}"


def _concept_join_summary(config: HiddenClusterConfig) -> dict[str, str | None]:
    return {
        "concept_source": _concept_source(config),
        "text_concept_mode": config.text_concept_mode,
    }


def _cluster_usage(
    entries: list[_ClusterEntry],
    assignments: dict[tuple[str, int], int],
    split_by_trace: dict[str, str],
) -> dict[str, dict[str, int]]:
    usage: dict[str, dict[str, int]] = {"all": {}}
    for entry in entries:
        cluster_key = f"hidden_cluster_{assignments[(entry.trace_id, entry.step_index)]}"
        usage["all"][cluster_key] = usage["all"].get(cluster_key, 0) + 1
        split_name = split_by_trace[entry.trace_id]
        usage.setdefault(split_name, {})
        usage[split_name][cluster_key] = usage[split_name].get(cluster_key, 0) + 1
    return usage


def _split_summary(records: list[TraceRecord]) -> dict[str, int]:
    return {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "correct": sum(1 for record in records if record.correct),
        "incorrect": sum(1 for record in records if not record.correct),
    }
