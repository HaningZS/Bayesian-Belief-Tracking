"""Export prefix-safe SBBT beliefs for rollout validation joins."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from csbf.evaluation import _base_rate, _fit_likelihoods, _codes, _paired_scores_codes, _scores
from csbf.filtering import BayesianReliabilityFilter, HMMConfig
from csbf.schema import TraceRecord
from csbf.split import split_by_question


SUPPORTED_PREDICTION_METHODS = {"hmm_score", "hmm_concept", "hmm_hybrid", "hmm_joint"}
SUPPORTED_SPLIT_FILTERS = {"train", "calibration", "test", "all"}


def build_sbbt_prefix_prediction_rows(
    records: list[TraceRecord],
    tasks: list[dict[str, Any]],
    *,
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    seed: int = 0,
    calibration_mode: str = "all_steps",
    method: str = "hmm_hybrid",
    split_filter: str = "test",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build SBBT belief rows keyed by ``trace_id`` and ``step_index``.

    The default exports only rollout prefixes whose source trace is in the test
    question split. This keeps rollout validation aligned with the usual
    train/calibration/test protocol.
    """

    if method not in SUPPORTED_PREDICTION_METHODS:
        raise ValueError(f"method must be one of {sorted(SUPPORTED_PREDICTION_METHODS)}")
    if split_filter not in SUPPORTED_SPLIT_FILTERS:
        raise ValueError(f"split_filter must be one of {sorted(SUPPORTED_SPLIT_FILTERS)}")

    splits = split_by_question(
        records,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        seed=seed,
    )
    calibration_records = splits["calibration"]
    config = HMMConfig(initial_on_track=_base_rate(calibration_records))
    score_likelihood, concept_likelihood, joint_likelihood = _fit_likelihoods(
        calibration_records,
        config,
        calibration_mode,
    )

    trace_split = {
        record.trace_id: split_name
        for split_name, split_records in splits.items()
        for record in split_records
    }
    selected_records = _selected_records(splits, split_filter)
    requested_steps = _requested_steps_by_trace(tasks)

    rows: list[dict[str, Any]] = []
    covered_keys: set[tuple[str, int]] = set()
    skipped_step_count = 0
    for record in selected_records:
        step_indices = sorted(requested_steps.get(record.trace_id, set()))
        if not step_indices:
            continue
        beliefs = _run_prediction_method(
            record,
            method=method,
            config=config,
            score_likelihood=score_likelihood,
            concept_likelihood=concept_likelihood,
            joint_likelihood=joint_likelihood,
        )
        for step_index in step_indices:
            if step_index < 0 or step_index >= len(beliefs):
                skipped_step_count += 1
                continue
            covered_keys.add((record.trace_id, step_index))
            rows.append(
                {
                    "trace_id": record.trace_id,
                    "question_id": record.question_id,
                    "step_index": step_index,
                    "split": trace_split[record.trace_id],
                    "prediction_method": method,
                    "sbbt_belief": beliefs[step_index],
                    "correct": bool(record.correct),
                    "observation_count": len(record.observations),
                }
            )

    covered_task_count = sum(
        1
        for task in tasks
        if (str(task.get("trace_id", "")), int(task.get("step_index", -1))) in covered_keys
    )
    summary = {
        "record_count": len(records),
        "task_count": len(tasks),
        "prefix_group_count": len(_task_keys(tasks)),
        "prediction_count": len(rows),
        "covered_task_count": covered_task_count,
        "skipped_task_count": len(tasks) - covered_task_count,
        "skipped_step_count": skipped_step_count,
        "split_filter": split_filter,
        "prediction_method": method,
        "calibration_mode": calibration_mode,
        "train_ratio": train_ratio,
        "calibration_ratio": calibration_ratio,
        "seed": seed,
        "split_trace_counts": {
            split_name: len(split_records)
            for split_name, split_records in splits.items()
        },
    }
    return rows, summary


def _selected_records(
    splits: dict[str, list[TraceRecord]],
    split_filter: str,
) -> list[TraceRecord]:
    if split_filter == "all":
        return splits["train"] + splits["calibration"] + splits["test"]
    return splits[split_filter]


def _requested_steps_by_trace(tasks: list[dict[str, Any]]) -> dict[str, set[int]]:
    steps: dict[str, set[int]] = defaultdict(set)
    for task in tasks:
        if task.get("trace_id") is None or task.get("step_index") is None:
            continue
        steps[str(task["trace_id"])].add(int(task["step_index"]))
    return steps


def _task_keys(tasks: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(task.get("trace_id", "")), int(task.get("step_index", -1)))
        for task in tasks
        if task.get("trace_id") is not None and task.get("step_index") is not None
    }


def _run_prediction_method(
    record: TraceRecord,
    *,
    method: str,
    config: HMMConfig,
    score_likelihood: Any,
    concept_likelihood: Any,
    joint_likelihood: Any,
) -> list[float]:
    if method == "hmm_score":
        return BayesianReliabilityFilter(
            config=config,
            score_likelihood=score_likelihood,
        ).run(scores=_scores(record))
    if method == "hmm_concept":
        return BayesianReliabilityFilter(
            config=config,
            concept_likelihood=concept_likelihood,
        ).run(concept_codes=_codes(record))
    if method == "hmm_joint":
        scores, codes = _paired_scores_codes(record)
        return BayesianReliabilityFilter(
            config=config,
            joint_likelihood=joint_likelihood,
        ).run(scores=scores, concept_codes=codes)
    scores, codes = _paired_scores_codes(record)
    return BayesianReliabilityFilter(
        config=config,
        score_likelihood=score_likelihood,
        concept_likelihood=concept_likelihood,
    ).run(scores=scores, concept_codes=codes)
