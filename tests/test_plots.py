from csbf.plots import categorize_traces, export_belief_data
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
            Observation(step_index=i, text=f"step {i}", score=s, concept_code=c)
            for i, (s, c) in enumerate(zip(scores, codes))
        ],
    )


def test_export_belief_data_returns_correct_structure():
    records = [
        _trace("q1-t0", True, [0.8, 0.7, 0.9], [1, 1, 2]),
        _trace("q2-t0", False, [0.3, 0.2, 0.1], [1, 1, 1]),
    ]
    data = export_belief_data(records)

    assert len(data) == 2
    for entry in data:
        assert "trace_id" in entry
        assert "question_id" in entry
        assert "correct" in entry
        assert "num_steps" in entry
        assert "scores" in entry
        assert "beliefs" in entry
        assert isinstance(entry["beliefs"], list)
        assert len(entry["beliefs"]) == entry["num_steps"]

    assert data[0]["correct"] is True
    assert data[1]["correct"] is False


def test_categorize_traces_separates_correct_wrong_and_self_repair():
    belief_data = [
        {
            "trace_id": "correct-1",
            "beliefs": [0.5, 0.6, 0.7],
            "correct": True,
        },
        {
            "trace_id": "wrong-1",
            "beliefs": [0.5, 0.3, 0.2],
            "correct": False,
        },
        {
            "trace_id": "repair-1",
            "beliefs": [0.8, 0.3, 0.75],
            "correct": True,
        },
    ]
    categories = categorize_traces(belief_data, recovery_threshold=0.15)

    assert "correct-1" in categories["correct"]
    assert "wrong-1" in categories["wrong"]
    assert "repair-1" in categories["self_repair"]
