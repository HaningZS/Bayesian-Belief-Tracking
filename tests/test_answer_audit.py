from csbf.answer_audit import build_answer_audit_rows
from csbf.schema import Observation, TraceRecord


def test_build_answer_audit_rows_flags_label_risk_categories():
    records = [
        _record(
            trace_id="q1-t0",
            correct=False,
            final_answer="1/2",
            gold_answer="0.5",
            trace_text="work \\boxed{1/2}",
        ),
        _record(
            trace_id="q2-t0",
            correct=False,
            final_answer="",
            gold_answer="7",
            trace_text="unfinished",
        ),
        _record(
            trace_id="q3-t0",
            correct=True,
            final_answer="42",
            gold_answer="42",
            trace_text="answer is 42",
        ),
        _record(
            trace_id="q4-t0",
            correct=False,
            final_answer="9",
            gold_answer="10",
            trace_text="long work \\boxed{9}",
            metadata={"finish_reason": "length"},
        ),
    ]

    rows = build_answer_audit_rows(records, max_per_category=2, seed=0)

    by_id = {row.trace_id: row for row in rows}
    assert "incorrect_with_boxed_answer" in by_id["q1-t0"].categories
    assert "possible_equivalent_answer" in by_id["q1-t0"].categories
    assert "missing_final_answer" in by_id["q2-t0"].categories
    assert "correct_without_boxed_marker" in by_id["q3-t0"].categories
    assert "token_cap_hit" in by_id["q4-t0"].categories
    assert by_id["q1-t0"].audit_id == "q1:q1-t0"
    assert by_id["q1-t0"].question_id == "q1"
    assert by_id["q1-t0"].observation_count == 1


def test_build_answer_audit_rows_limits_per_category_but_deduplicates_records():
    records = [
        _record(trace_id="q1-t0", correct=False, final_answer="1", trace_text="\\boxed{1}"),
        _record(trace_id="q2-t0", correct=False, final_answer="2", trace_text="\\boxed{2}"),
        _record(trace_id="q3-t0", correct=False, final_answer="3", trace_text="\\boxed{3}"),
    ]

    rows = build_answer_audit_rows(records, max_per_category=2, seed=123)

    assert len(rows) == 2
    assert all("incorrect_with_boxed_answer" in row.categories for row in rows)


def _record(
    trace_id: str,
    correct: bool,
    final_answer: str,
    trace_text: str,
    gold_answer: str = "gold",
    metadata: dict[str, object] | None = None,
) -> TraceRecord:
    return TraceRecord(
        question_id=trace_id.split("-")[0],
        question=f"Question {trace_id}",
        trace_id=trace_id,
        trace_text=trace_text,
        final_answer=final_answer,
        gold_answer=gold_answer,
        correct=correct,
        observations=[Observation(step_index=0, score=0.5)],
        metadata=metadata or {},
    )
