from csbf.calibration import ON_TRACK, OFF_TRACK
from csbf.em import em_fit_score_likelihood
from csbf.filtering import HMMConfig
from csbf.schema import Observation, TraceRecord


def _make_record(question_id: str, correct: bool, scores: list[float]) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=f"{question_id}-t0",
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(step_index=i, text=f"step {i}", score=s, concept_code=0)
            for i, s in enumerate(scores)
        ],
    )


def test_em_fit_converges_on_separable_traces():
    """EM should estimate emissions that separate ON/OFF scores."""
    records = [
        _make_record("q1", True, [0.8, 0.9, 0.85]),
        _make_record("q2", True, [0.7, 0.8, 0.75]),
        _make_record("q3", False, [0.3, 0.2, 0.15]),
        _make_record("q4", False, [0.4, 0.3, 0.25]),
    ]
    config = HMMConfig(p_error=0.1, p_recover=0.1)
    likelihood = em_fit_score_likelihood(records, config, bins=5, max_iterations=10)

    assert likelihood.probability(0.85, ON_TRACK) > likelihood.probability(0.85, OFF_TRACK)
    assert likelihood.probability(0.15, OFF_TRACK) > likelihood.probability(0.15, ON_TRACK)


def test_em_fit_handles_single_class_gracefully():
    """EM should not crash when all traces are correct."""
    records = [
        _make_record("q1", True, [0.8, 0.9]),
        _make_record("q2", True, [0.7, 0.6]),
    ]
    config = HMMConfig(p_error=0.1, p_recover=0.1)
    likelihood = em_fit_score_likelihood(records, config, bins=5, max_iterations=5)

    assert likelihood.probability(0.5, ON_TRACK) > 0


def test_em_fit_converges_within_iterations():
    """With clean data and loose tolerance EM should converge before max_iterations."""
    records = [
        _make_record("q1", True, [0.9, 0.95]),
        _make_record("q2", False, [0.1, 0.05]),
    ]
    config = HMMConfig(p_error=0.05, p_recover=0.05)
    likelihood = em_fit_score_likelihood(
        records, config, bins=5, max_iterations=50, tolerance=1e-3,
    )

    assert likelihood.probability(0.9, ON_TRACK) > likelihood.probability(0.9, OFF_TRACK)
    assert likelihood.probability(0.1, OFF_TRACK) > likelihood.probability(0.1, ON_TRACK)


def test_em_fit_single_step_traces():
    """EM should handle traces with only one observation each."""
    records = [
        _make_record("q1", True, [0.9]),
        _make_record("q2", False, [0.2]),
    ]
    config = HMMConfig(p_error=0.1, p_recover=0.1)
    likelihood = em_fit_score_likelihood(records, config, bins=5, max_iterations=10)

    assert likelihood.probability(0.9, ON_TRACK) > 0
    assert likelihood.probability(0.2, OFF_TRACK) > 0
