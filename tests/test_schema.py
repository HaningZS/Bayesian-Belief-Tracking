from pathlib import Path

from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl


def test_trace_record_round_trips_jsonl(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    records = [
        TraceRecord(
            question_id="q1",
            question="What is 2+2?",
            trace_id="q1-t0",
            trace_text="2+2=4. #### 4",
            final_answer="4",
            gold_answer="4",
            correct=True,
            observations=[
                Observation(step_index=0, text="2+2=4.", score=0.8, concept_code=7),
            ],
        )
    ]

    save_jsonl(records, path)
    loaded = load_jsonl(path)

    assert loaded == records


def test_trace_record_rejects_observations_without_signal():
    record = TraceRecord(
        question_id="q1",
        question="What is 2+2?",
        trace_id="q1-t0",
        trace_text="2+2=4",
        final_answer="4",
        gold_answer="4",
        correct=True,
        observations=[Observation(step_index=0, text="2+2=4")],
    )

    try:
        record.validate()
    except ValueError as error:
        assert "score or concept_code" in str(error)
    else:
        raise AssertionError("expected validation failure")


def test_observation_loads_without_new_fields():
    """Old JSONL without entropy/score_delta still loads correctly."""
    data = {"step_index": 0, "text": "step", "score": 0.5, "concept_code": 1}
    obs = Observation.from_dict(data)
    assert obs.entropy is None
    assert obs.score_delta is None


def test_observation_round_trips_new_fields():
    obs = Observation(step_index=0, text="step", score=0.5, concept_code=1, entropy=2.3, score_delta=-0.1)
    data = obs.to_dict()
    loaded = Observation.from_dict(data)
    assert loaded == obs
    assert loaded.entropy == 2.3
    assert loaded.score_delta == -0.1


def test_observation_rejects_negative_entropy():
    try:
        Observation(step_index=0, score=0.5, concept_code=1, entropy=-0.1).validate()
    except ValueError as error:
        assert "entropy" in str(error)
    else:
        raise AssertionError("expected validation failure")


def test_observation_rejects_out_of_range_score_delta():
    try:
        Observation(step_index=0, score=0.5, concept_code=1, score_delta=1.5).validate()
    except ValueError as error:
        assert "score_delta" in str(error)
    else:
        raise AssertionError("expected validation failure")
