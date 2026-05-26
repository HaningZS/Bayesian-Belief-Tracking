from csbf.schema import Observation, TraceRecord
from csbf.split import split_by_question_stratified


def _record(question_id: str, trace_id: str, correct: bool) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=trace_id,
        trace_text="trace",
        final_answer="1",
        gold_answer="1",
        correct=correct,
        observations=[Observation(step_index=0, score=0.5, concept_code=1)],
    )


def test_stratified_split_keeps_questions_together():
    records = [
        _record("q1", "q1-t0", True),
        _record("q1", "q1-t1", True),
        _record("q2", "q2-t0", False),
        _record("q3", "q3-t0", True),
        _record("q4", "q4-t0", False),
        _record("q5", "q5-t0", True),
    ]
    splits = split_by_question_stratified(records, train_ratio=0.5, calibration_ratio=0.25, seed=0)

    split_names_by_question: dict[str, set[str]] = {}
    for split_name, split_records in splits.items():
        for record in split_records:
            split_names_by_question.setdefault(record.question_id, set()).add(split_name)

    assert all(len(names) == 1 for names in split_names_by_question.values())
    assert sorted(splits.keys()) == ["calibration", "test", "train"]


def test_stratified_split_produces_nonempty_partitions():
    records = [
        _record("q1", "q1-t0", True),
        _record("q2", "q2-t0", False),
        _record("q3", "q3-t0", True),
        _record("q4", "q4-t0", False),
        _record("q5", "q5-t0", True),
        _record("q6", "q6-t0", False),
    ]
    splits = split_by_question_stratified(records, train_ratio=0.5, calibration_ratio=0.25, seed=0)

    assert len(splits["train"]) > 0
    assert len(splits["calibration"]) > 0
    assert len(splits["test"]) > 0

    all_ids = set()
    for split_records in splits.values():
        for r in split_records:
            all_ids.add(r.trace_id)
    assert all_ids == {r.trace_id for r in records}


def test_stratified_split_rejects_fewer_than_three_questions():
    records = [_record("q1", "q1-t0", True), _record("q2", "q2-t0", False)]
    try:
        split_by_question_stratified(records, train_ratio=0.5, calibration_ratio=0.25)
    except ValueError as error:
        assert "three" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_stratified_split_rejects_empty_records():
    try:
        split_by_question_stratified([], train_ratio=0.5, calibration_ratio=0.25)
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("expected ValueError")
