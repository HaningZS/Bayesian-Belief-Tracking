from csbf.prefix_beliefs import build_sbbt_prefix_prediction_rows
from csbf.prefix_rollouts import build_prefix_rollout_tasks
from csbf.schema import Observation, TraceRecord
from csbf.split import split_by_question


def _records() -> list[TraceRecord]:
    records: list[TraceRecord] = []
    for index in range(5):
        correct = index % 2 == 0
        score_base = 0.7 if correct else 0.3
        records.append(
            TraceRecord(
                question_id=f"q{index}",
                question=f"What is {index}?",
                trace_id=f"q{index}-trace",
                trace_text="step 0\nstep 1\nstep 2",
                final_answer=str(index),
                gold_answer=str(index),
                correct=correct,
                observations=[
                    Observation(step_index=0, text="step 0", score=score_base, concept_code="start"),
                    Observation(step_index=1, text="step 1", score=score_base + 0.05, concept_code="middle"),
                    Observation(step_index=2, text="step 2", score=score_base + 0.1, concept_code="final"),
                ],
            )
        )
    return records


def test_build_sbbt_prefix_prediction_rows_defaults_to_test_split_only():
    records = _records()
    tasks = build_prefix_rollout_tasks(records, prefix_fractions=[0.5], rollouts_per_prefix=1)
    split_records = split_by_question(records, train_ratio=0.4, calibration_ratio=0.2, seed=0)
    expected_test_trace_ids = {record.trace_id for record in split_records["test"]}

    rows, summary = build_sbbt_prefix_prediction_rows(
        records,
        tasks,
        train_ratio=0.4,
        calibration_ratio=0.2,
        seed=0,
        split_filter="test",
    )

    assert {row["trace_id"] for row in rows} == expected_test_trace_ids
    assert {row["split"] for row in rows} == {"test"}
    assert {row["step_index"] for row in rows} == {1}
    assert all(0.0 <= row["sbbt_belief"] <= 1.0 for row in rows)
    assert summary["prediction_count"] == len(rows)
    assert summary["skipped_task_count"] == len(tasks) - len(rows)
