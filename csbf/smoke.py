"""No-GPU, no-API smoke pipeline over synthetic traces."""

from __future__ import annotations

from dataclasses import dataclass

from csbf.calibration import ConceptLikelihood, HistogramLikelihood
from csbf.filtering import BayesianReliabilityFilter, HMMConfig
from csbf.metrics import brier_score, roc_auc_score


@dataclass(frozen=True)
class SyntheticTrace:
    scores: list[float]
    concept_codes: list[int]
    final_correct: int


def run_offline_smoke() -> dict[str, object]:
    """Run a deterministic end-to-end smoke test without model/API dependencies."""

    traces = [
        SyntheticTrace(scores=[0.30, 0.72, 0.88], concept_codes=[2, 7, 7], final_correct=1),
        SyntheticTrace(scores=[0.76, 0.42, 0.18], concept_codes=[7, 3, 3], final_correct=0),
        SyntheticTrace(scores=[0.38, 0.58], concept_codes=[2, 7], final_correct=1),
    ]

    calibration_scores: list[float] = []
    calibration_codes: list[int] = []
    calibration_labels: list[int] = []
    for trace in traces:
        calibration_scores.extend(trace.scores)
        calibration_codes.extend(trace.concept_codes)
        calibration_labels.extend([trace.final_correct] * len(trace.scores))

    score_likelihood = HistogramLikelihood.fit(
        calibration_scores,
        calibration_labels,
        bins=5,
        smoothing=0.5,
    )
    concept_likelihood = ConceptLikelihood.fit(
        calibration_codes,
        calibration_labels,
        smoothing=0.5,
    )
    reliability_filter = BayesianReliabilityFilter(
        config=HMMConfig(p_error=0.10, p_recover=0.20, initial_on_track=0.5),
        score_likelihood=score_likelihood,
        concept_likelihood=concept_likelihood,
    )

    beliefs = [
        reliability_filter.run(scores=trace.scores, concept_codes=trace.concept_codes)
        for trace in traces
    ]
    final_beliefs = [trace_beliefs[-1] for trace_beliefs in beliefs]
    labels = [trace.final_correct for trace in traces]

    return {
        "num_traces": len(traces),
        "beliefs": beliefs,
        "metrics": {
            "brier": brier_score(labels, final_beliefs),
            "auc": roc_auc_score(labels, final_beliefs),
        },
    }
