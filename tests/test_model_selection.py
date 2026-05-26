from csbf.model_selection import select_hmm_config
from csbf.schema import Observation, TraceRecord


def _trace(trace_id: str, correct: bool, scores: list[float], codes: list[int]) -> TraceRecord:
    return TraceRecord(
        question_id=trace_id.split("-")[0],
        question="question",
        trace_id=trace_id,
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(step_index=index, text=f"step {index}", score=score, concept_code=code)
            for index, (score, code) in enumerate(zip(scores, codes))
        ],
    )


def test_select_hmm_config_returns_candidate_from_grid():
    calibration = [
        _trace("q1-t0", False, [0.2, 0.3], [1, 1]),
        _trace("q2-t0", True, [0.7, 0.8], [2, 2]),
    ]
    validation = [
        _trace("q3-t0", False, [0.3, 0.2], [1, 1]),
        _trace("q4-t0", True, [0.6, 0.9], [2, 2]),
    ]

    config = select_hmm_config(
        calibration,
        validation,
        p_error_grid=[0.05, 0.10],
        p_recover_grid=[0.10, 0.20],
    )

    assert config.p_error in {0.05, 0.10}
    assert config.p_recover in {0.10, 0.20}
    assert config.initial_on_track in {0.3, 0.5, 0.7}


def test_select_hmm_config_uses_separate_fit_and_validation_sets():
    fit_records = [
        _trace("q1-t0", False, [0.1, 0.2], [1, 1]),
        _trace("q2-t0", True, [0.8, 0.9], [2, 2]),
        _trace("q3-t0", False, [0.15, 0.25], [1, 1]),
    ]
    val_records = [
        _trace("q4-t0", True, [0.7, 0.85], [2, 2]),
        _trace("q5-t0", False, [0.2, 0.1], [1, 1]),
    ]

    config = select_hmm_config(
        fit_records,
        val_records,
        p_error_grid=[0.05, 0.10],
        p_recover_grid=[0.10, 0.20],
        calibration_mode="all_steps",
    )

    assert config.p_error in {0.05, 0.10}
    assert config.p_recover in {0.10, 0.20}

    config_final = select_hmm_config(
        fit_records,
        val_records,
        p_error_grid=[0.05, 0.10],
        p_recover_grid=[0.10, 0.20],
        calibration_mode="final_step",
    )

    assert config_final.p_error in {0.05, 0.10}
    assert config_final.p_recover in {0.10, 0.20}


def test_select_hmm_config_accepts_initial_belief_grid():
    calibration = [
        _trace("q1-t0", False, [0.2, 0.3], [1, 1]),
        _trace("q2-t0", True, [0.7, 0.8], [2, 2]),
    ]
    validation = [
        _trace("q3-t0", False, [0.3, 0.2], [1, 1]),
        _trace("q4-t0", True, [0.6, 0.9], [2, 2]),
    ]

    config = select_hmm_config(
        calibration,
        validation,
        p_error_grid=[0.05],
        p_recover_grid=[0.10],
        initial_on_track_grid=[0.25, 0.75],
    )

    assert config.initial_on_track in {0.25, 0.75}
