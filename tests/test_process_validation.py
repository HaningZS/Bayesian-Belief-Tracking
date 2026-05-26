from csbf.process_validation import ProcessLabel, build_process_prefix_examples, evaluate_process_labels
from csbf.schema import Observation, TraceRecord


def test_build_process_prefix_examples_marks_prefixes_after_first_error():
    records = [
        _record(
            trace_id="t1",
            scores=[0.9, 0.7, 0.2],
            correct=False,
        ),
        _record(
            trace_id="t2",
            scores=[0.8, 0.85],
            correct=True,
        ),
    ]
    labels = {
        "t1": ProcessLabel(trace_id="t1", first_error_step=1),
        "t2": ProcessLabel(trace_id="t2", first_error_step=None),
    }

    examples = build_process_prefix_examples(records, labels)

    assert [(item.trace_id, item.step_index, item.error_started) for item in examples] == [
        ("t1", 0, 0),
        ("t1", 1, 1),
        ("t1", 2, 1),
        ("t2", 0, 0),
        ("t2", 1, 0),
    ]
    assert examples[0].predicted_error_probability == 0.1
    assert examples[2].predicted_error_probability == 0.8


def test_evaluate_process_labels_reports_coverage_and_metrics():
    records = [
        _record(trace_id="t1", scores=[0.9, 0.7, 0.2], correct=False),
        _record(trace_id="t2", scores=[0.8, 0.85], correct=True),
        _record(trace_id="missing", scores=[0.5], correct=False),
    ]
    labels = {
        "t1": ProcessLabel(trace_id="t1", first_error_step=1),
        "t2": ProcessLabel(trace_id="t2", first_error_step=None),
    }

    report = evaluate_process_labels(records, labels)

    assert report["trace_count"] == 3
    assert report["labeled_trace_count"] == 2
    assert report["prefix_count"] == 5
    assert report["positive_prefix_count"] == 2
    assert report["metrics"]["auroc"] == 1.0
    assert report["missing_label_trace_ids"] == ["missing"]


def _record(trace_id: str, scores: list[float], correct: bool) -> TraceRecord:
    return TraceRecord(
        question_id=trace_id,
        question=f"Question {trace_id}",
        trace_id=trace_id,
        trace_text="trace",
        final_answer="1",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(step_index=index, score=score)
            for index, score in enumerate(scores)
        ],
    )
