from csbf.schema import TraceRecord
from csbf.split import split_by_question


def _record(question_id: str, trace_id: str) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=trace_id,
        trace_text="trace",
        final_answer="1",
        gold_answer="1",
        correct=True,
    )


def test_split_by_question_keeps_all_traces_for_a_question_together():
    records = [
        _record("q1", "q1-t0"),
        _record("q1", "q1-t1"),
        _record("q2", "q2-t0"),
        _record("q3", "q3-t0"),
        _record("q4", "q4-t0"),
    ]

    split = split_by_question(records, train_ratio=0.5, calibration_ratio=0.25, seed=3)

    split_names_by_question = {}
    for split_name, split_records in split.items():
        for record in split_records:
            split_names_by_question.setdefault(record.question_id, set()).add(split_name)

    assert all(len(split_names) == 1 for split_names in split_names_by_question.values())
    assert sorted(split.keys()) == ["calibration", "test", "train"]
