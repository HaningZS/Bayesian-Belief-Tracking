"""Rollout-based prefix value validation helpers.

This module keeps task export and aggregation dependency-free. Continuation
generation can be performed by any external model runner that writes the JSONL
result contract consumed here.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from csbf.schema import TraceRecord


DEFAULT_PREFIX_FRACTIONS = (0.25, 0.5, 0.75)


def build_prefix_rollout_tasks(
    records: Iterable[TraceRecord],
    *,
    prefix_fractions: Iterable[float] = DEFAULT_PREFIX_FRACTIONS,
    rollouts_per_prefix: int = 4,
    max_prefix_groups: int | None = None,
    seed: int = 0,
    prompt_dataset_type: str = "auto",
    prompt_format: str | None = None,
    prompt_style: str | None = None,
) -> list[dict[str, Any]]:
    """Create deterministic continuation tasks from existing trace prefixes."""

    fractions = [float(value) for value in prefix_fractions]
    if not fractions:
        raise ValueError("prefix_fractions must not be empty")
    if any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("prefix_fractions must be in (0, 1]")
    if rollouts_per_prefix <= 0:
        raise ValueError("rollouts_per_prefix must be positive")
    if max_prefix_groups is not None and max_prefix_groups <= 0:
        raise ValueError("max_prefix_groups must be positive when set")
    if prompt_dataset_type not in {"auto", "gsm8k", "math"}:
        raise ValueError("prompt_dataset_type must be 'auto', 'gsm8k', or 'math'")

    candidates: list[dict[str, Any]] = []
    for record in records:
        if not record.observations:
            continue
        n_observations = len(record.observations)
        seen_steps: set[int] = set()
        for requested_fraction in fractions:
            step_index = _step_index_for_fraction(n_observations, requested_fraction)
            if step_index in seen_steps:
                continue
            seen_steps.add(step_index)
            candidates.append(
                _prefix_group(
                    record,
                    step_index,
                    requested_fraction=requested_fraction,
                    prompt_dataset_type=prompt_dataset_type,
                    prompt_format=prompt_format,
                    prompt_style=prompt_style,
                )
            )

    if max_prefix_groups is not None and len(candidates) > max_prefix_groups:
        selected = set(random.Random(seed).sample(range(len(candidates)), max_prefix_groups))
        candidates = [group for index, group in enumerate(candidates) if index in selected]

    tasks: list[dict[str, Any]] = []
    for group in candidates:
        for rollout_index in range(rollouts_per_prefix):
            task_id = f"{group['task_group_id']}|rollout={rollout_index}"
            row = dict(group)
            row["task_id"] = task_id
            row["rollout_index"] = rollout_index
            row["rollouts_per_prefix"] = rollouts_per_prefix
            tasks.append(row)
    return tasks


def summarize_prefix_rollout_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize an exported task list."""

    groups = {str(task["task_group_id"]) for task in tasks}
    question_ids = {str(task.get("question_id", "")) for task in tasks}
    trace_ids = {str(task.get("trace_id", "")) for task in tasks}
    return {
        "task_count": len(tasks),
        "prefix_group_count": len(groups),
        "question_count": len(question_ids - {""}),
        "trace_count": len(trace_ids - {""}),
        "rollouts_per_prefix_min": min((int(task.get("rollouts_per_prefix", 0)) for task in tasks), default=0),
        "rollouts_per_prefix_max": max((int(task.get("rollouts_per_prefix", 0)) for task in tasks), default=0),
        "prefix_fractions": sorted(
            {
                float(task["actual_prefix_fraction"])
                for task in tasks
                if task.get("actual_prefix_fraction") is not None
            }
        ),
    }


def aggregate_prefix_rollouts(
    tasks: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    score_bins: int = 10,
    prediction_rows: list[dict[str, Any]] | None = None,
    prediction_specs: dict[str, str] | None = None,
    require_predictions: bool = False,
) -> dict[str, Any]:
    """Aggregate continuation correctness into empirical prefix success rates."""

    if score_bins <= 0:
        raise ValueError("score_bins must be positive")
    prediction_specs = dict(prediction_specs or {})
    if require_predictions and not prediction_specs:
        raise ValueError("require_predictions needs at least one prediction spec")
    for prediction_name in prediction_specs:
        _validate_prediction_name(prediction_name)
    predictions_by_key = _prediction_lookup(prediction_rows or [])

    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    grouped_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped_tasks[str(task["task_group_id"])].append(task)

    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmatched_result_count = 0
    matched_task_ids: set[str] = set()
    matched_result_count = 0
    for result in results:
        task_id = str(result.get("task_id", ""))
        task = tasks_by_id.get(task_id)
        if task is None:
            unmatched_result_count += 1
            continue
        matched_task_ids.add(task_id)
        matched_result_count += 1
        grouped_results[str(task["task_group_id"])].append(result)

    groups: list[dict[str, Any]] = []
    for task_group_id, group_tasks in grouped_tasks.items():
        group_results = grouped_results.get(task_group_id, [])
        if not group_results:
            continue
        template = group_tasks[0]
        rollout_success_rate = sum(1.0 for row in group_results if bool(row.get("correct"))) / len(group_results)
        source_score = _optional_float(template.get("source_score"))
        prediction_row = predictions_by_key.get(
            (str(template.get("trace_id", "")), int(template.get("step_index", 0)))
        )
        joined_predictions = {
            prediction_name: _prediction_value(prediction_row, source_field)
            for prediction_name, source_field in prediction_specs.items()
        }
        if require_predictions and any(value is None for value in joined_predictions.values()):
            continue
        groups.append(
            _drop_none_values(
                {
                    "task_group_id": task_group_id,
                    "question_id": template.get("question_id"),
                    "trace_id": template.get("trace_id"),
                    "step_index": int(template.get("step_index", 0)),
                    "actual_prefix_fraction": _optional_float(template.get("actual_prefix_fraction")),
                    "requested_prefix_fraction": _optional_float(template.get("requested_prefix_fraction")),
                    "source_score": source_score,
                    "source_correct": bool(template.get("source_correct")),
                    "rollout_count": len(group_results),
                    "rollout_success_count": sum(1 for row in group_results if bool(row.get("correct"))),
                    "rollout_success_rate": rollout_success_rate,
                    **joined_predictions,
                }
            )
        )

    scored_pairs = [
        (float(group["source_score"]), float(group["rollout_success_rate"]))
        for group in groups
        if group.get("source_score") is not None
    ]
    rollout_counts = [int(group["rollout_count"]) for group in groups]
    included_result_count = sum(rollout_counts)
    summary: dict[str, Any] = {
        "task_count": len(tasks),
        "result_count": included_result_count,
        "matched_result_count": matched_result_count,
        "matched_task_count": len(matched_task_ids),
        "duplicate_result_count": max(0, matched_result_count - len(matched_task_ids)),
        "unmatched_result_count": unmatched_result_count,
        "missing_result_count": max(0, len(tasks_by_id) - len(matched_task_ids)),
        "group_count": len(groups),
        "rollout_count_min": min(rollout_counts, default=0),
        "rollout_count_max": max(rollout_counts, default=0),
        "rollout_count_mean": _mean(rollout_counts),
        "mean_rollout_success_rate": _mean([group["rollout_success_rate"] for group in groups]),
        "source_correct_rate": _mean([1.0 if group["source_correct"] else 0.0 for group in groups]),
        "score_brier_against_rollout_success": _brier(scored_pairs),
        "score_pearson_with_rollout_success": _pearson(scored_pairs),
        "score_spearman_with_rollout_success": _spearman(scored_pairs),
        "score_calibration_bins": _score_bins(groups, score_bins),
        "require_predictions": bool(require_predictions),
        "groups": groups,
    }
    for prediction_name in prediction_specs:
        pairs = _prediction_pairs(groups, prediction_name)
        summary[f"{prediction_name}_coverage_count"] = len(pairs)
        summary[f"{prediction_name}_brier_against_rollout_success"] = _brier(pairs)
        summary[f"{prediction_name}_pearson_with_rollout_success"] = _pearson(pairs)
        summary[f"{prediction_name}_spearman_with_rollout_success"] = _spearman(pairs)
        summary[f"{prediction_name}_calibration_bins"] = _prediction_bins(
            groups,
            score_bins,
            value_field=prediction_name,
            lower_key="prediction_lower",
            upper_key="prediction_upper",
            mean_key="mean_prediction",
        )
    return summary


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
    return rows


def write_jsonl_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_prefix_rollout_summary(
    summary: dict[str, Any],
    *,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> None:
    if output_json is not None:
        target = Path(output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_csv is not None:
        _write_summary_csv(summary, Path(output_csv))
    if output_markdown is not None:
        _write_summary_markdown(summary, Path(output_markdown))


def _prefix_group(
    record: TraceRecord,
    step_index: int,
    *,
    requested_fraction: float,
    prompt_dataset_type: str,
    prompt_format: str | None,
    prompt_style: str | None,
) -> dict[str, Any]:
    observation = record.observations[step_index]
    metadata = dict(record.metadata)
    actual_fraction = (step_index + 1) / len(record.observations)
    resolved_dataset_type = (
        _infer_prompt_dataset_type(record) if prompt_dataset_type == "auto" else prompt_dataset_type
    )
    resolved_prompt_format = prompt_format or str(metadata.get("prompt_format", "raw"))
    resolved_prompt_style = prompt_style or str(metadata.get("prompt_style", "default"))
    task_group_id = f"{record.trace_id}|step={step_index}"
    return {
        "task_group_id": task_group_id,
        "question_id": record.question_id,
        "trace_id": record.trace_id,
        "step_index": step_index,
        "requested_prefix_fraction": requested_fraction,
        "actual_prefix_fraction": actual_fraction,
        "question": record.question,
        "gold_answer": record.gold_answer,
        "prefix_text": "\n".join(item.text for item in record.observations[: step_index + 1]),
        "prompt_dataset_type": resolved_dataset_type,
        "prompt_format": resolved_prompt_format,
        "prompt_style": resolved_prompt_style,
        "generation_prefix": str(metadata.get("generation_prefix", "")),
        "source_score": None if observation.score is None else float(observation.score),
        "source_correct": bool(record.correct),
        "source_final_answer": record.final_answer,
        "source_observation_count": len(record.observations),
        "source_dataset": metadata.get("dataset"),
        "source_model": metadata.get("model"),
    }


def _step_index_for_fraction(n_observations: int, fraction: float) -> int:
    return max(0, min(n_observations - 1, math.ceil(n_observations * fraction) - 1))


def _infer_prompt_dataset_type(record: TraceRecord) -> str:
    dataset = str(record.metadata.get("dataset", "")).lower()
    if "gsm" in dataset:
        return "gsm8k"
    return "math"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _prediction_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("trace_id") is None or row.get("step_index") is None:
            continue
        lookup[(str(row["trace_id"]), int(row["step_index"]))] = row
    return lookup


def _prediction_value(row: dict[str, Any] | None, field: str) -> float | None:
    if row is None or row.get(field) is None:
        return None
    return _optional_float(row[field])


def _prediction_pairs(groups: list[dict[str, Any]], prediction_name: str) -> list[tuple[float, float]]:
    return [
        (float(group[prediction_name]), float(group["rollout_success_rate"]))
        for group in groups
        if group.get(prediction_name) is not None
    ]


def _validate_prediction_name(name: str) -> None:
    if not name or any(not (char.isalnum() or char == "_") for char in name):
        raise ValueError("prediction names must be non-empty alphanumeric/underscore strings")


def _drop_none_values(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value is not None}


def _mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values]
    if not items:
        return None
    return sum(items) / len(items)


def _brier(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    return sum((score - success) ** 2 for score, success in pairs) / len(pairs)


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [left for left, _ in pairs]
    ys = [right for _, right in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return numerator / (denom_x * denom_y)


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = _average_ranks([left for left, _ in pairs])
    ys = _average_ranks([right for _, right in pairs])
    return _pearson(list(zip(xs, ys)))


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = average_rank
        cursor = end
    return ranks


def _score_bins(groups: list[dict[str, Any]], score_bins: int) -> list[dict[str, Any]]:
    return _prediction_bins(
        groups,
        score_bins,
        value_field="source_score",
        lower_key="score_lower",
        upper_key="score_upper",
        mean_key="mean_source_score",
    )


def _prediction_bins(
    groups: list[dict[str, Any]],
    score_bins: int,
    *,
    value_field: str,
    lower_key: str,
    upper_key: str,
    mean_key: str,
) -> list[dict[str, Any]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(score_bins)]
    for group in groups:
        value = group.get(value_field)
        if value is None:
            continue
        index = min(score_bins - 1, max(0, int(float(value) * score_bins)))
        buckets[index].append(group)
    rows: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        lower = index / score_bins
        upper = (index + 1) / score_bins
        mean_prediction = _mean([float(row[value_field]) for row in bucket])
        mean_success = _mean([float(row["rollout_success_rate"]) for row in bucket])
        rows.append(
            {
                "bin_index": index,
                lower_key: lower,
                upper_key: upper,
                "count": len(bucket),
                mean_key: mean_prediction,
                "mean_rollout_success_rate": mean_success,
                "calibration_gap": None
                if mean_prediction is None or mean_success is None
                else mean_prediction - mean_success,
            }
        )
    return rows


def _write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "task_count",
        "result_count",
        "unmatched_result_count",
        "missing_result_count",
        "group_count",
        "rollout_count_min",
        "rollout_count_max",
        "rollout_count_mean",
        "mean_rollout_success_rate",
        "source_correct_rate",
        "score_brier_against_rollout_success",
        "score_pearson_with_rollout_success",
        "score_spearman_with_rollout_success",
    ]
    keys.extend(
        key
        for key in sorted(summary)
        if key not in keys and _is_scalar_prediction_metric_key(key)
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in keys:
            writer.writerow({"metric": key, "value": summary.get(key)})


def _write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prefix Rollout Validation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Prefix groups | {summary.get('group_count')} |",
        f"| Continuation results | {summary.get('result_count')} |",
        f"| Mean rollout success | {_format_value(summary.get('mean_rollout_success_rate'))} |",
        f"| Score Brier vs rollout success | {_format_value(summary.get('score_brier_against_rollout_success'))} |",
        f"| Score Pearson vs rollout success | {_format_value(summary.get('score_pearson_with_rollout_success'))} |",
        f"| Score Spearman vs rollout success | {_format_value(summary.get('score_spearman_with_rollout_success'))} |",
    ]
    for prediction_name in _prediction_names_from_summary(summary):
        display_name = _display_prediction_name(prediction_name)
        lines.extend(
            [
                f"| {display_name} coverage | {_format_value(summary.get(f'{prediction_name}_coverage_count'))} |",
                f"| {display_name} Brier vs rollout success | {_format_value(summary.get(f'{prediction_name}_brier_against_rollout_success'))} |",
                f"| {display_name} Pearson vs rollout success | {_format_value(summary.get(f'{prediction_name}_pearson_with_rollout_success'))} |",
                f"| {display_name} Spearman vs rollout success | {_format_value(summary.get(f'{prediction_name}_spearman_with_rollout_success'))} |",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _is_scalar_prediction_metric_key(key: str) -> bool:
    return key.endswith(
        (
            "_coverage_count",
            "_brier_against_rollout_success",
            "_pearson_with_rollout_success",
            "_spearman_with_rollout_success",
        )
    )


def _prediction_names_from_summary(summary: dict[str, Any]) -> list[str]:
    suffix = "_brier_against_rollout_success"
    return sorted(
        key[: -len(suffix)]
        for key in summary
        if key.endswith(suffix) and not key.startswith("score_")
    )


def _display_prediction_name(name: str) -> str:
    return name.replace("_", " ").title()


def _format_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
