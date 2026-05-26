from csbf.diagnostics import summarize_trace_records
from csbf.schema import Observation, TraceRecord


def _record(question_id: str, trace_id: str, correct: bool, score: float) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=trace_id,
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(step_index=0, text="step 0", score=score, concept_code=1),
            Observation(step_index=1, text="step 1", score=score, concept_code=2),
        ],
    )


def test_summarize_trace_records_warns_on_smoke_file_too_small_for_pipeline():
    summary = summarize_trace_records([_record("q1", "q1-t0", True, 1.0)])

    assert summary["num_records"] == 1
    assert summary["num_questions"] == 1
    assert "need at least 3 question ids for train/calibration/test splits" in summary["warnings"]
    assert "only one correctness class is present" in summary["warnings"]
    assert "all observation scores are identical" in summary["warnings"]


def test_summarize_trace_records_reports_score_and_class_variation():
    summary = summarize_trace_records(
        [
            _record("q1", "q1-t0", True, 0.8),
            _record("q2", "q2-t0", False, 0.2),
            _record("q3", "q3-t0", True, 0.6),
        ]
    )

    assert summary["correctness"] == {"correct": 2, "incorrect": 1}
    assert summary["score_summary"]["unique_count"] == 3
    assert summary["warnings"] == []


def test_summarize_trace_records_warns_on_high_score_saturation():
    records = [
        _record("q1", "q1-t0", True, 1.0),
        _record("q2", "q2-t0", False, 1.0),
        _record("q3", "q3-t0", True, 0.9),
        _record("q4", "q4-t0", False, 1.0),
        _record("q5", "q5-t0", True, 0.9),
        _record("q6", "q6-t0", False, 1.0),
    ]

    summary = summarize_trace_records(records)

    assert "observation scores are highly saturated at 0 or 1" in summary["warnings"]
    assert "observation scores have very few unique values" in summary["warnings"]
