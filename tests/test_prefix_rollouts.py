import math

from csbf.prefix_rollouts import (
    aggregate_prefix_rollouts,
    build_prefix_rollout_tasks,
)
from csbf.schema import Observation, TraceRecord


def _record() -> TraceRecord:
    return TraceRecord(
        question_id="q1",
        question="What is 40 + 2?",
        trace_id="q1-local-0",
        trace_text="first\nsecond\nthird\nfourth",
        final_answer="42",
        gold_answer="42",
        correct=True,
        observations=[
            Observation(step_index=0, text="first", score=0.2),
            Observation(step_index=1, text="second", score=0.6),
            Observation(step_index=2, text="third", score=0.9),
            Observation(step_index=3, text="fourth", score=0.8),
        ],
        metadata={"dataset": "math500", "prompt_format": "raw", "prompt_style": "concise"},
    )


def test_build_prefix_rollout_tasks_selects_deterministic_prefix_groups():
    tasks = build_prefix_rollout_tasks(
        [_record()],
        prefix_fractions=[0.25, 0.75],
        rollouts_per_prefix=2,
        seed=0,
    )

    assert [task["task_id"] for task in tasks] == [
        "q1-local-0|step=0|rollout=0",
        "q1-local-0|step=0|rollout=1",
        "q1-local-0|step=2|rollout=0",
        "q1-local-0|step=2|rollout=1",
    ]
    assert tasks[0]["prefix_text"] == "first"
    assert tasks[2]["prefix_text"] == "first\nsecond\nthird"
    assert tasks[2]["source_score"] == 0.9
    assert tasks[2]["actual_prefix_fraction"] == 0.75
    assert tasks[2]["requested_prefix_fraction"] == 0.75
    assert tasks[2]["prompt_dataset_type"] == "math"
    assert tasks[2]["prompt_format"] == "raw"
    assert tasks[2]["prompt_style"] == "concise"


def test_aggregate_prefix_rollouts_compares_scores_to_empirical_success():
    tasks = build_prefix_rollout_tasks(
        [_record()],
        prefix_fractions=[0.25, 0.75],
        rollouts_per_prefix=2,
    )
    results = [
        {"task_id": "q1-local-0|step=0|rollout=0", "correct": False},
        {"task_id": "q1-local-0|step=0|rollout=1", "correct": True},
        {"task_id": "q1-local-0|step=2|rollout=0", "correct": True},
        {"task_id": "q1-local-0|step=2|rollout=1", "correct": True},
    ]

    summary = aggregate_prefix_rollouts(tasks, results, score_bins=2)

    assert summary["group_count"] == 2
    assert summary["result_count"] == 4
    assert summary["rollout_count_min"] == 2
    assert summary["rollout_count_max"] == 2
    assert math.isclose(summary["score_brier_against_rollout_success"], 0.05)
    assert summary["score_pearson_with_rollout_success"] > 0.99
    assert summary["score_spearman_with_rollout_success"] > 0.99
    assert summary["groups"][0]["rollout_success_rate"] == 0.5
    assert summary["groups"][1]["rollout_success_rate"] == 1.0
    assert len(summary["score_calibration_bins"]) == 2


def test_aggregate_prefix_rollouts_joins_prediction_rows_by_trace_and_step():
    tasks = build_prefix_rollout_tasks(
        [_record()],
        prefix_fractions=[0.25, 0.75],
        rollouts_per_prefix=2,
    )
    results = [
        {"task_id": "q1-local-0|step=0|rollout=0", "correct": False},
        {"task_id": "q1-local-0|step=0|rollout=1", "correct": True},
        {"task_id": "q1-local-0|step=2|rollout=0", "correct": True},
        {"task_id": "q1-local-0|step=2|rollout=1", "correct": True},
    ]

    summary = aggregate_prefix_rollouts(
        tasks,
        results,
        score_bins=2,
        prediction_rows=[
            {"trace_id": "q1-local-0", "step_index": 0, "sbbt_belief": 0.4},
            {"trace_id": "q1-local-0", "step_index": 2, "sbbt_belief": 0.9},
        ],
        prediction_specs={"belief": "sbbt_belief"},
    )

    assert summary["belief_coverage_count"] == 2
    assert math.isclose(summary["belief_brier_against_rollout_success"], 0.01)
    assert summary["belief_pearson_with_rollout_success"] > 0.99
    assert summary["belief_spearman_with_rollout_success"] > 0.99
    assert summary["groups"][0]["belief"] == 0.4
    assert summary["groups"][1]["belief"] == 0.9
    assert len(summary["belief_calibration_bins"]) == 2


def test_aggregate_prefix_rollouts_can_restrict_to_prediction_covered_groups():
    tasks = build_prefix_rollout_tasks(
        [_record()],
        prefix_fractions=[0.25, 0.75],
        rollouts_per_prefix=2,
    )
    results = [
        {"task_id": "q1-local-0|step=0|rollout=0", "correct": False},
        {"task_id": "q1-local-0|step=0|rollout=1", "correct": True},
        {"task_id": "q1-local-0|step=2|rollout=0", "correct": True},
        {"task_id": "q1-local-0|step=2|rollout=1", "correct": True},
    ]

    summary = aggregate_prefix_rollouts(
        tasks,
        results,
        prediction_rows=[
            {"trace_id": "q1-local-0", "step_index": 2, "sbbt_belief": 0.9},
        ],
        prediction_specs={"belief": "sbbt_belief"},
        require_predictions=True,
    )

    assert summary["group_count"] == 1
    assert summary["result_count"] == 2
    assert summary["groups"][0]["step_index"] == 2
    assert summary["belief_coverage_count"] == 1
    assert math.isclose(summary["score_brier_against_rollout_success"], 0.01)


def test_aggregate_prefix_rollouts_does_not_report_negative_missing_for_duplicate_results():
    tasks = build_prefix_rollout_tasks(
        [_record()],
        prefix_fractions=[0.25],
        rollouts_per_prefix=1,
    )
    task_id = tasks[0]["task_id"]

    summary = aggregate_prefix_rollouts(
        tasks,
        [
            {"task_id": task_id, "correct": False},
            {"task_id": task_id, "correct": True},
        ],
    )

    assert summary["result_count"] == 2
    assert summary["matched_task_count"] == 1
    assert summary["duplicate_result_count"] == 1
    assert summary["missing_result_count"] == 0
