from pathlib import Path

from csbf.pipeline import run_preobserved_pipeline
from csbf.schema import Observation, TraceRecord, save_jsonl


def _record(question_id: str, trace_id: str, correct: bool, score: float, code: int) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=trace_id,
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(step_index=0, text="step 0", score=score, concept_code=code),
            Observation(step_index=1, text="step 1", score=score, concept_code=code),
        ],
    )


def test_run_preobserved_pipeline_uses_question_level_splits(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    save_jsonl(
        [
            _record("q1", "q1-t0", False, 0.2, 1),
            _record("q2", "q2-t0", True, 0.8, 2),
            _record("q3", "q3-t0", False, 0.3, 1),
            _record("q4", "q4-t0", True, 0.7, 2),
            _record("q5", "q5-t0", False, 0.25, 1),
            _record("q6", "q6-t0", True, 0.75, 2),
        ],
        path,
    )

    result = run_preobserved_pipeline(path, train_ratio=0.34, calibration_ratio=0.33, seed=0)

    assert result["num_records"] == 6
    assert result["split_sizes"]["calibration"] > 0
    assert result["diagnostics"]["num_questions"] == 6
    assert "hmm_hybrid" in result["methods"]
    assert "p50" in result["prefix_diagnostics"]
    assert "hmm_hybrid_smooth" in result["prefix_diagnostics"]["p50"]
    assert result["calibration_mode"] == "all_steps"
    assert result["selected_hmm_config"] is None
    assert 0.0 <= result["methods"]["hmm_hybrid"]["metrics"]["brier"] <= 1.0


def test_run_preobserved_pipeline_reports_selected_hmm_config(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    save_jsonl(
        [
            _record("q1", "q1-t0", False, 0.2, 1),
            _record("q2", "q2-t0", True, 0.8, 2),
            _record("q3", "q3-t0", False, 0.3, 1),
            _record("q4", "q4-t0", True, 0.7, 2),
            _record("q5", "q5-t0", False, 0.25, 1),
            _record("q6", "q6-t0", True, 0.75, 2),
            _record("q7", "q7-t0", False, 0.35, 1),
            _record("q8", "q8-t0", True, 0.85, 2),
        ],
        path,
    )

    result = run_preobserved_pipeline(
        path,
        train_ratio=0.5,
        calibration_ratio=0.25,
        seed=0,
        use_model_selection=True,
    )

    assert result["use_model_selection"] is True
    assert result["selected_hmm_config"] is not None
    assert set(result["selected_hmm_config"]) == {"p_error", "p_recover", "initial_on_track"}
