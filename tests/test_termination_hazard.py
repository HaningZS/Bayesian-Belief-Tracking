import json
from pathlib import Path

from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl
from csbf.termination_hazard import (
    build_termination_hazard_records,
    termination_hazard_code,
    termination_hazard_score,
)


def _record(scores: list[float]) -> TraceRecord:
    return TraceRecord(
        question_id="q1",
        question="question",
        trace_id="q1-t0",
        trace_text="trace",
        final_answer="1",
        gold_answer="1",
        correct=True,
        observations=[
            Observation(step_index=index, text=f"step {index}", score=score, concept_code="old")
            for index, score in enumerate(scores)
        ],
    )


def test_termination_hazard_score_is_prefix_safe_elapsed_shortness():
    assert termination_hazard_score(prefix_index=0, horizon_observations=4) == 0.75
    assert termination_hazard_score(prefix_index=1, horizon_observations=4) == 0.5
    assert termination_hazard_score(prefix_index=3, horizon_observations=4) == 0.0
    assert termination_hazard_score(prefix_index=9, horizon_observations=4) == 0.0


def test_termination_hazard_code_buckets_elapsed_prefix_position():
    assert termination_hazard_code(prefix_index=0, horizon_observations=4) == "term_hazard|elapsed_le_25"
    assert termination_hazard_code(prefix_index=1, horizon_observations=4) == "term_hazard|elapsed_le_50"
    assert termination_hazard_code(prefix_index=2, horizon_observations=4) == "term_hazard|elapsed_le_75"
    assert termination_hazard_code(prefix_index=3, horizon_observations=4) == "term_hazard|elapsed_gt_75"


def test_build_termination_hazard_records_rewrites_scores_codes_and_report(tmp_path: Path):
    records_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "records_termination_hazard.jsonl"
    report_path = tmp_path / "termination_hazard_report.json"
    save_jsonl([_record([0.2, 0.8, 0.9, 1.0])], records_path)

    summary = build_termination_hazard_records(
        records_path,
        output_path=output_path,
        report_path=report_path,
        horizon_observations=4,
    )

    rewritten = load_jsonl(output_path)
    observations = rewritten[0].observations
    assert [observation.score for observation in observations] == [0.75, 0.5, 0.25, 0.0]
    assert [observation.score_delta for observation in observations] == [0.0, -0.25, -0.25, -0.25]
    assert [observation.concept_code for observation in observations] == [
        "term_hazard|elapsed_le_25",
        "term_hazard|elapsed_le_50",
        "term_hazard|elapsed_le_75",
        "term_hazard|elapsed_gt_75",
    ]
    assert rewritten[0].metadata["score_source"] == "termination_hazard"
    assert rewritten[0].metadata["concept_source"] == "termination_hazard"
    assert rewritten[0].metadata["termination_hazard"]["horizon_observations"] == 4
    assert summary["score_summary"]["min"] == 0.0
    assert summary["score_summary"]["max"] == 0.75
    assert json.loads(report_path.read_text(encoding="utf-8"))["outputs"]["records"] == str(output_path)
