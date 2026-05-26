import json
from pathlib import Path

import pytest

from csbf.hidden_concepts import (
    HiddenClusterConfig,
    build_hidden_cluster_records,
    run_hidden_cluster_split_seed_sweep,
    write_hidden_cluster_split_seed_sweep,
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
            Observation(step_index=0, text="step 0", score=0.8 if correct else 0.2, concept_code="old"),
            Observation(step_index=1, text="step 1", score=0.7 if correct else 0.3, concept_code="old"),
        ],
    )


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    records = [_record(str(index), index % 2 == 0) for index in range(12)]
    records_path = tmp_path / "records.jsonl"
    features_path = tmp_path / "features.jsonl"
    save_jsonl(records, records_path)
    rows = []
    for record in records:
        sign = 1.0 if record.correct else -1.0
        for observation in record.observations:
            rows.append(
                {
                    "trace_id": record.trace_id,
                    "question_id": record.question_id,
                    "step_index": observation.step_index,
                    "hidden_last_token": [sign, sign * 2.0, -sign],
                }
            )
    features_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return records_path, features_path


def test_build_hidden_cluster_records_rewrites_concept_codes_from_train_clusters(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    output_path = tmp_path / "records_hidden_clusters.jsonl"
    report_path = tmp_path / "hidden_clusters.json"

    summary = build_hidden_cluster_records(
        records_path,
        features_path,
        output_path=output_path,
        report_path=report_path,
        train_ratio=0.5,
        calibration_ratio=0.25,
        seed=0,
        config=HiddenClusterConfig(cluster_count=2, projection_dim=4, iterations=8),
    )

    rewritten = load_jsonl(output_path)
    codes = {
        observation.concept_code
        for record in rewritten
        for observation in record.observations
    }
    assert all(str(code).startswith("hidden_cluster_") for code in codes)
    assert len(codes) <= 2
    assert summary["feature_rows"]["used"] == 24
    assert report_path.exists()


def test_build_hidden_cluster_records_can_join_self_verification_concepts(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    output_path = tmp_path / "records_hidden_clusters_self.jsonl"

    summary = build_hidden_cluster_records(
        records_path,
        features_path,
        output_path=output_path,
        train_ratio=0.5,
        calibration_ratio=0.25,
        seed=0,
        config=HiddenClusterConfig(
            cluster_count=2,
            projection_dim=4,
            iterations=4,
            text_concept_mode="self_verification",
        ),
    )

    rewritten = load_jsonl(output_path)
    codes = [
        str(observation.concept_code)
        for record in rewritten
        for observation in record.observations
    ]
    assert all(code.startswith("hidden_cluster_") for code in codes)
    assert all("|sv_" in code for code in codes)
    assert summary["concept_join"]["text_concept_mode"] == "self_verification"


def test_build_hidden_cluster_records_rejects_missing_hidden_rows(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    rows = features_path.read_text(encoding="utf-8").splitlines()
    features_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing hidden feature"):
        build_hidden_cluster_records(
            records_path,
            features_path,
            output_path=tmp_path / "out.jsonl",
            config=HiddenClusterConfig(cluster_count=2, projection_dim=4, iterations=1),
        )


def test_run_hidden_cluster_split_seed_sweep_refits_clusters_per_seed(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)

    summary = run_hidden_cluster_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(3),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=HiddenClusterConfig(cluster_count=2, projection_dim=4, iterations=4),
    )

    assert summary["seed_count"] == 3
    assert summary["aggregate"]["mean_best_online_hmm_auroc"] is not None
    assert all(row["cluster_usage"]["train"] for row in summary["seeds"])
    assert all("hmm_concept" in row["method_metrics"] for row in summary["seeds"])


def test_run_hidden_cluster_split_seed_sweep_supports_joint_self_verification(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)

    summary = run_hidden_cluster_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(2),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=HiddenClusterConfig(
            cluster_count=2,
            projection_dim=4,
            iterations=2,
            text_concept_mode="self_verification",
        ),
    )

    assert summary["concept_join"]["text_concept_mode"] == "self_verification"
    assert all(row["concept_join"]["text_concept_mode"] == "self_verification" for row in summary["seeds"])
    assert all("hmm_concept" in row["method_metrics"] for row in summary["seeds"])


def test_write_hidden_cluster_split_seed_sweep_outputs_json_and_csv(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    summary = run_hidden_cluster_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(2),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=HiddenClusterConfig(cluster_count=2, projection_dim=4, iterations=2),
    )
    written = write_hidden_cluster_split_seed_sweep(
        summary,
        tmp_path / "cluster_split.json",
        tmp_path / "cluster_split.csv",
    )

    assert Path(written["json"]).exists()
    assert Path(written["csv"]).exists()
    assert "hmm_score_brier" in Path(written["csv"]).read_text(encoding="utf-8")
