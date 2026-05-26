"""Trace JSONL quality diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from csbf.schema import TraceRecord


def summarize_trace_records(records: list[TraceRecord]) -> dict[str, Any]:
    """Summarize trace records and emit warnings for common pipeline blockers."""

    if not records:
        raise ValueError("records must not be empty")

    question_ids = {record.question_id for record in records}
    traces_by_question: dict[str, int] = defaultdict(int)
    labels = [1 if record.correct else 0 for record in records]
    observation_counts = [len(record.observations) for record in records]
    scores = [
        float(observation.score)
        for record in records
        for observation in record.observations
        if observation.score is not None
    ]
    concept_codes = [
        observation.concept_code
        for record in records
        for observation in record.observations
        if observation.concept_code is not None
    ]
    for record in records:
        traces_by_question[record.question_id] += 1

    warnings = _warnings(
        num_questions=len(question_ids),
        labels=labels,
        scores=scores,
        concept_codes=concept_codes,
        observation_counts=observation_counts,
    )

    return {
        "num_records": len(records),
        "num_questions": len(question_ids),
        "traces_per_question": {
            "min": min(traces_by_question.values()),
            "max": max(traces_by_question.values()),
        },
        "correctness": {
            "correct": sum(labels),
            "incorrect": len(labels) - sum(labels),
        },
        "observations_per_trace": {
            "min": min(observation_counts),
            "max": max(observation_counts),
            "mean": round(sum(observation_counts) / len(observation_counts), 12),
        },
        "score_summary": _score_summary(scores),
        "concept_code_summary": {
            "count": len(concept_codes),
            "unique_count": len(set(concept_codes)),
        },
        "warnings": warnings,
    }


def _score_summary(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {
            "count": 0,
            "unique_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "saturated_fraction": None,
        }
    saturated = [score for score in scores if score <= 0.0 or score >= 1.0]
    return {
        "count": len(scores),
        "unique_count": len(set(scores)),
        "min": min(scores),
        "max": max(scores),
        "mean": round(sum(scores) / len(scores), 12),
        "saturated_fraction": round(len(saturated) / len(scores), 12),
    }


def _warnings(
    num_questions: int,
    labels: list[int],
    scores: list[float],
    concept_codes: list[object],
    observation_counts: list[int],
) -> list[str]:
    warnings: list[str] = []
    if num_questions < 3:
        warnings.append("need at least 3 question ids for train/calibration/test splits")
    if len(set(labels)) < 2:
        warnings.append("only one correctness class is present")
    if 0 in observation_counts:
        warnings.append("some traces have no observations")
    if not scores:
        warnings.append("no score observations are present")
    elif len(set(scores)) == 1:
        warnings.append("all observation scores are identical")
    elif _dominant_value_fraction(scores) >= 0.8:
        warnings.append("observation scores are dominated by one value")
    if scores and len(scores) >= 10 and len(set(scores)) <= 3:
        warnings.append("observation scores have very few unique values")
    if scores and sum(1 for score in scores if score <= 0.0 or score >= 1.0) / len(scores) >= 0.65:
        warnings.append("observation scores are highly saturated at 0 or 1")
    if not concept_codes:
        warnings.append("no concept_code observations are present")
    elif len(set(concept_codes)) == 1:
        warnings.append("all concept codes are identical")
    return warnings


def _dominant_value_fraction(values: list[object]) -> float:
    counter = Counter(values)
    return counter.most_common(1)[0][1] / len(values)
