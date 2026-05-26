"""Prefix-safe termination / survival observation rewrites."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl


DEFAULT_HORIZON_OBSERVATIONS = 64


def termination_hazard_score(prefix_index: int, horizon_observations: int = DEFAULT_HORIZON_OBSERVATIONS) -> float:
    """Return an elapsed-reasoning shortness score for a prefix.

    The score is prefix-safe: at prefix index ``t`` it only uses the elapsed
    observation count ``t + 1`` and a fixed horizon, not the future trace length.
    """

    if prefix_index < 0:
        raise ValueError("prefix_index must be non-negative")
    horizon = _checked_horizon(horizon_observations)
    elapsed_fraction = min((int(prefix_index) + 1) / horizon, 1.0)
    return round(1.0 - elapsed_fraction, 12)


def termination_hazard_code(prefix_index: int, horizon_observations: int = DEFAULT_HORIZON_OBSERVATIONS) -> str:
    """Bucket elapsed prefix position as a discrete termination-hazard concept."""

    if prefix_index < 0:
        raise ValueError("prefix_index must be non-negative")
    horizon = _checked_horizon(horizon_observations)
    elapsed_fraction = min((int(prefix_index) + 1) / horizon, 1.0)
    if elapsed_fraction <= 0.25:
        bucket = "elapsed_le_25"
    elif elapsed_fraction <= 0.50:
        bucket = "elapsed_le_50"
    elif elapsed_fraction <= 0.75:
        bucket = "elapsed_le_75"
    else:
        bucket = "elapsed_gt_75"
    return f"term_hazard|{bucket}"


def build_termination_hazard_records(
    records_or_path: list[TraceRecord] | str | Path,
    *,
    output_path: str | Path,
    report_path: str | Path,
    horizon_observations: int = DEFAULT_HORIZON_OBSERVATIONS,
) -> dict[str, Any]:
    """Rewrite records so scores/concepts encode elapsed-reasoning survival."""

    horizon = _checked_horizon(horizon_observations)
    records = _load_records(records_or_path)
    rewritten = [_rewrite_record(record, horizon_observations=horizon) for record in records]
    save_jsonl(rewritten, output_path)

    scores = [
        float(observation.score)
        for record in rewritten
        for observation in record.observations
        if observation.score is not None
    ]
    concepts = Counter(
        str(observation.concept_code)
        for record in rewritten
        for observation in record.observations
        if observation.concept_code is not None
    )
    summary: dict[str, Any] = {
        "record_count": len(rewritten),
        "observation_count": len(scores),
        "horizon_observations": horizon,
        "score_summary": _score_summary(scores),
        "concept_usage": dict(sorted(concepts.items())),
        "outputs": {
            "records": str(output_path),
            "report": str(report_path),
        },
    }
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _rewrite_record(record: TraceRecord, *, horizon_observations: int) -> TraceRecord:
    observations: list[Observation] = []
    previous_score: float | None = None
    for index, observation in enumerate(record.observations):
        score = termination_hazard_score(index, horizon_observations=horizon_observations)
        delta = 0.0 if previous_score is None else round(score - previous_score, 12)
        observations.append(
            Observation(
                step_index=observation.step_index,
                text=observation.text,
                score=score,
                concept_code=termination_hazard_code(index, horizon_observations=horizon_observations),
                entropy=observation.entropy,
                score_delta=delta,
            )
        )
        previous_score = score

    metadata = dict(record.metadata)
    metadata["score_source"] = "termination_hazard"
    metadata["concept_source"] = "termination_hazard"
    metadata["termination_hazard"] = {
        "horizon_observations": horizon_observations,
        "score": "1 - min(elapsed_observations / horizon_observations, 1)",
        "prefix_safe": True,
    }
    return TraceRecord(
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


def _load_records(records_or_path: list[TraceRecord] | str | Path) -> list[TraceRecord]:
    if isinstance(records_or_path, str | Path):
        return load_jsonl(records_or_path)
    return list(records_or_path)


def _checked_horizon(horizon_observations: int) -> int:
    horizon = int(horizon_observations)
    if horizon <= 0:
        raise ValueError("horizon_observations must be positive")
    return horizon


def _score_summary(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(scores),
        "min": round(min(scores), 12),
        "max": round(max(scores), 12),
        "mean": round(sum(scores) / len(scores), 12),
    }
