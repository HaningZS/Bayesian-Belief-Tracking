"""Belief trajectory visualization data export."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from csbf.calibration import ConceptLikelihood, HistogramLikelihood
from csbf.filtering import BayesianReliabilityFilter, HMMConfig
from csbf.schema import TraceRecord


def export_belief_data(
    records: list[TraceRecord],
    score_likelihood: HistogramLikelihood | None = None,
    concept_likelihood: ConceptLikelihood | None = None,
    config: HMMConfig | None = None,
) -> list[dict[str, Any]]:
    """Export belief trajectories for all records as JSON-serializable dicts."""

    cfg = config or HMMConfig()
    filt = BayesianReliabilityFilter(
        config=cfg,
        score_likelihood=score_likelihood,
        concept_likelihood=concept_likelihood,
    )
    results: list[dict[str, Any]] = []
    for record in records:
        scores = [float(o.score) for o in record.observations if o.score is not None]
        codes = [o.concept_code for o in record.observations if o.concept_code is not None]

        beliefs = filt.run(
            scores=scores if scores else None,
            concept_codes=codes if codes else None,
        )
        results.append({
            "trace_id": record.trace_id,
            "question_id": record.question_id,
            "correct": record.correct,
            "num_steps": len(record.observations),
            "scores": scores,
            "beliefs": beliefs,
        })
    return results


def categorize_traces(
    belief_data: list[dict[str, Any]],
    recovery_threshold: float = 0.15,
) -> dict[str, list[str]]:
    """Categorize traces into correct/wrong/self-repair by belief pattern."""

    categories: dict[str, list[str]] = {
        "correct": [],
        "wrong": [],
        "self_repair": [],
    }
    for entry in belief_data:
        trace_id = entry["trace_id"]
        beliefs = entry["beliefs"]
        correct = entry["correct"]

        if not beliefs:
            continue

        if not correct:
            categories["wrong"].append(trace_id)
            continue

        if len(beliefs) >= 3:
            min_belief = min(beliefs)
            min_idx = beliefs.index(min_belief)
            if min_idx > 0 and min_idx < len(beliefs) - 1:
                drop = beliefs[0] - min_belief
                recovery = beliefs[-1] - min_belief
                if drop > recovery_threshold and recovery > recovery_threshold:
                    categories["self_repair"].append(trace_id)
                    continue

        categories["correct"].append(trace_id)

    return categories
