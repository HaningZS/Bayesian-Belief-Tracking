import json
import math
from pathlib import Path

import pytest

from csbf.latent_trajectory import (
    LatentTrajectoryConfig,
    build_latent_trajectory_records,
    compute_trajectory_metrics,
    run_latent_trajectory_split_seed_sweep,
    write_latent_trajectory_split_seed_sweep,
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
            Observation(step_index=2, text="step 2", score=0.5, concept_code="old"),
        ],
    )


def _write_features(path: Path, records: list[TraceRecord]) -> None:
    rows = []
    for record in records:
        vectors = (
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
            if record.correct
            else [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        )
        for observation, vector in zip(record.observations, vectors, strict=True):
            rows.append(
                {
                    "trace_id": record.trace_id,
                    "question_id": record.question_id,
                    "step_index": observation.step_index,
                    "hidden_last_token": vector,
                    "hidden_layer_1_last_token": [value * 2.0 for value in vector],
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


def test_compute_trajectory_metrics_measures_net_cumulative_and_alignment():
    metrics = compute_trajectory_metrics([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    assert metrics["net_change"] == pytest.approx(math.sqrt(2.0) / 3.0)
    assert metrics["cumulative_change"] == pytest.approx(2.0)
    assert metrics["aligned_change"] == pytest.approx(math.sqrt(0.5))


def test_build_latent_trajectory_records_rewrites_scores_with_train_only_scaler(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    output_path = tmp_path / "records_lt.jsonl"
    report_path = tmp_path / "lt_report.json"

    summary = build_latent_trajectory_records(
        records_path,
        features_path,
        output_path=output_path,
        report_path=report_path,
        train_ratio=0.5,
        calibration_ratio=0.25,
        seed=0,
        config=LatentTrajectoryConfig(projection_dim=2, metric="composite"),
    )

    rewritten = load_jsonl(output_path)
    final_scores = {record.trace_id: record.observations[-1].score for record in rewritten}
    correct_scores = [final_scores[record.trace_id] for record in rewritten if record.correct]
    incorrect_scores = [final_scores[record.trace_id] for record in rewritten if not record.correct]
    assert min(correct_scores) > max(incorrect_scores)
    assert all(0.0 <= float(score) <= 1.0 for score in final_scores.values())
    assert all(record.metadata["score_source"] == "latent_trajectory" for record in rewritten)
    assert summary["feature_rows"]["used"] == 36
    assert summary["feature_rows"]["feature_field"] == "hidden_last_token"
    assert summary["transform"]["metric"] == "composite"
    assert report_path.exists()


def test_build_latent_trajectory_records_accepts_layerwise_feature_field(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    output_path = tmp_path / "records_lt_layer.jsonl"

    summary = build_latent_trajectory_records(
        records_path,
        features_path,
        output_path=output_path,
        train_ratio=0.5,
        calibration_ratio=0.25,
        seed=0,
        config=LatentTrajectoryConfig(
            projection_dim=2,
            metric="aligned_change",
            feature_field="hidden_layer_1_last_token",
        ),
    )

    rewritten = load_jsonl(output_path)
    assert summary["feature_rows"]["feature_field"] == "hidden_layer_1_last_token"
    assert all(record.metadata["latent_trajectory"]["feature_field"] == "hidden_layer_1_last_token" for record in rewritten)


def test_build_latent_trajectory_records_rejects_missing_hidden_rows(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    rows = features_path.read_text(encoding="utf-8").splitlines()
    features_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing hidden feature"):
        build_latent_trajectory_records(
            records_path,
            features_path,
            output_path=tmp_path / "out.jsonl",
            report_path=tmp_path / "report.json",
            config=LatentTrajectoryConfig(projection_dim=2),
        )


def test_run_latent_trajectory_split_seed_sweep_evaluates_rewritten_scores(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)

    summary = run_latent_trajectory_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(3),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=LatentTrajectoryConfig(projection_dim=2, metric="composite"),
    )

    assert summary["seed_count"] == 3
    assert summary["aggregate"]["mean_hmm_score_brier"] is not None
    assert all(row["lt_final_metrics"]["test"]["metrics"]["auroc"] == 1.0 for row in summary["seeds"])
    assert all("hmm_score" in row["method_metrics"] for row in summary["seeds"])


def test_write_latent_trajectory_split_seed_sweep_outputs_json_and_csv(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    summary = run_latent_trajectory_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(2),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=LatentTrajectoryConfig(projection_dim=2, metric="composite"),
    )
    written = write_latent_trajectory_split_seed_sweep(
        summary,
        tmp_path / "lt_split.json",
        tmp_path / "lt_split.csv",
    )

    assert Path(written["json"]).exists()
    assert Path(written["csv"]).exists()
    assert "mean_hmm_score_brier" in Path(written["csv"]).read_text(encoding="utf-8")
