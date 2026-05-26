import json
from pathlib import Path

from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl
from csbf.score_dynamics import build_score_dynamics_records, dynamic_score_code


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


def test_dynamic_score_code_marks_prefix_recovery():
    codes, deltas = dynamic_score_code(
        [0.20, 0.35, 0.30, 0.55],
        ema_alpha=0.4,
        volatility_window=2,
        delta_threshold=0.05,
        residual_threshold=0.05,
        volatility_threshold=0.08,
        recovery_threshold=0.20,
    )

    assert deltas == [0.0, 0.15, -0.05, 0.25]
    assert codes[0] == "score_dyn|flat|low_vol|near_ema|stable"
    assert codes[-1].endswith("|recovery")
    assert "up" in codes[-1]
    assert "high_vol" in codes[-1]


def test_build_score_dynamics_records_rewrites_codes_and_report(tmp_path: Path):
    records_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "records_score_dynamics.jsonl"
    report_path = tmp_path / "score_dynamics_report.json"
    save_jsonl([_record([0.20, 0.35, 0.30, 0.55])], records_path)

    summary = build_score_dynamics_records(
        records_path,
        output_path=output_path,
        report_path=report_path,
        ema_alpha=0.4,
        volatility_window=2,
        recovery_threshold=0.20,
    )

    rewritten = load_jsonl(output_path)
    observations = rewritten[0].observations
    assert [observation.score_delta for observation in observations] == [0.0, 0.15, -0.05, 0.25]
    assert rewritten[0].metadata["concept_source"] == "score_dynamics"
    assert rewritten[0].metadata["score_dynamics"]["ema_alpha"] == 0.4
    assert observations[-1].concept_code.endswith("|recovery")
    assert summary["concept_usage"][observations[-1].concept_code] == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["outputs"]["records"] == str(output_path)
