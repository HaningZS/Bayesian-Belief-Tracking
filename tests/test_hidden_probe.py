import json
from pathlib import Path

import pytest

from csbf.hidden_probe import (
    HiddenProbeConfig,
    build_hidden_probe_records,
    run_hidden_probe_split_seed_sweep,
    write_hidden_probe_split_seed_sweep,
)
from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl


def _record(question_id: str, correct: bool) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=f"{question_id}-t0",
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(step_index=0, text="step 0", score=0.5, concept_code="old"),
            Observation(step_index=1, text="step 1", score=0.5, concept_code="old"),
        ],
    )


def _write_features(path: Path, records: list[TraceRecord]) -> None:
    rows = []
    for record in records:
        sign = 1.0 if record.correct else -1.0
        for observation in record.observations:
            rows.append(
                {
                    "trace_id": record.trace_id,
                    "question_id": record.question_id,
                    "step_index": observation.step_index,
                    "hidden_last_token": [3.0 * sign, 1.5 * sign, -0.5 * sign],
                    "hidden_mean_pool": [1.0 * sign, 0.5 * sign],
                    "entropy": 0.1 if record.correct else 1.0,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    records = [_record(str(index), index % 2 == 0) for index in range(12)]
    records_path = tmp_path / "records.jsonl"
    features_path = tmp_path / "features.jsonl"
    save_jsonl(records, records_path)
    _write_features(features_path, records)
    return records_path, features_path


def test_build_hidden_probe_records_rewrites_scores_without_test_leakage(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    output_path = tmp_path / "records_hidden_probe.jsonl"
    report_path = tmp_path / "hidden_probe_report.json"

    summary = build_hidden_probe_records(
        records_path,
        features_path,
        output_path=output_path,
        report_path=report_path,
        train_ratio=0.5,
        calibration_ratio=0.25,
        seed=0,
        config=HiddenProbeConfig(
            projection_dim=8,
            epochs=25,
            learning_rate=0.2,
            calibration_epochs=80,
            include_entropy_feature=False,
        ),
    )

    rewritten = load_jsonl(output_path)
    scores = {record.trace_id: record.observations[-1].score for record in rewritten}
    correct_scores = [scores[record.trace_id] for record in rewritten if record.correct]
    incorrect_scores = [scores[record.trace_id] for record in rewritten if not record.correct]
    assert min(correct_scores) > max(incorrect_scores)
    assert all(0.0 <= float(score) <= 1.0 for score in scores.values())
    assert all(record.metadata["score_source"] == "hidden_probe" for record in rewritten)
    assert summary["split"]["train"]["questions"] == 6
    assert summary["feature_rows"]["used"] == 24
    assert report_path.exists()


def test_build_hidden_probe_records_can_use_pooled_feature_fields(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    output_path = tmp_path / "records_hidden_pool_probe.jsonl"
    report_path = tmp_path / "hidden_pool_probe_report.json"

    summary = build_hidden_probe_records(
        records_path,
        features_path,
        output_path=output_path,
        report_path=report_path,
        train_ratio=0.5,
        calibration_ratio=0.25,
        seed=0,
        config=HiddenProbeConfig(
            projection_dim=4,
            epochs=20,
            learning_rate=0.2,
            calibration_epochs=40,
            feature_fields=("hidden_mean_pool",),
        ),
    )

    rewritten = load_jsonl(output_path)
    scores = {record.trace_id: record.observations[-1].score for record in rewritten}
    correct_scores = [scores[record.trace_id] for record in rewritten if record.correct]
    incorrect_scores = [scores[record.trace_id] for record in rewritten if not record.correct]
    assert min(correct_scores) > max(incorrect_scores)
    assert summary["feature_rows"]["feature_fields"] == ["hidden_mean_pool"]
    assert rewritten[0].metadata["hidden_probe"]["feature_fields"] == ["hidden_mean_pool"]


def test_build_hidden_probe_records_rejects_missing_hidden_rows(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    rows = features_path.read_text(encoding="utf-8").splitlines()
    features_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing hidden feature"):
        build_hidden_probe_records(
            records_path,
            features_path,
            output_path=tmp_path / "out.jsonl",
            report_path=tmp_path / "report.json",
            config=HiddenProbeConfig(projection_dim=4, epochs=1),
        )


def test_build_hidden_probe_records_rejects_missing_configured_feature_field(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)

    with pytest.raises(ValueError, match="missing hidden feature field hidden_token_pool"):
        build_hidden_probe_records(
            records_path,
            features_path,
            output_path=tmp_path / "out.jsonl",
            report_path=tmp_path / "report.json",
            config=HiddenProbeConfig(projection_dim=4, epochs=1, feature_fields=("hidden_token_pool",)),
        )


def test_run_hidden_probe_split_seed_sweep_retrains_probe_per_seed(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)

    summary = run_hidden_probe_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(3),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=HiddenProbeConfig(
            projection_dim=8,
            epochs=15,
            learning_rate=0.2,
            calibration_epochs=40,
            include_entropy_feature=False,
        ),
    )

    assert summary["seed_count"] == 3
    assert summary["aggregate"]["mean_hmm_score_brier"] is not None
    assert all(row["probe_prefix_metrics"]["test"]["rows"] > 0 for row in summary["seeds"])
    assert all("hmm_score" in row["method_metrics"] for row in summary["seeds"])


def test_write_hidden_probe_split_seed_sweep_outputs_json_and_csv(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    summary = run_hidden_probe_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(2),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=HiddenProbeConfig(projection_dim=4, epochs=5, calibration_epochs=10),
    )
    written = write_hidden_probe_split_seed_sweep(
        summary,
        tmp_path / "probe_split.json",
        tmp_path / "probe_split.csv",
    )

    assert Path(written["json"]).exists()
    assert Path(written["csv"]).exists()
    assert "mean_hmm_score_brier" in Path(written["csv"]).read_text(encoding="utf-8")
