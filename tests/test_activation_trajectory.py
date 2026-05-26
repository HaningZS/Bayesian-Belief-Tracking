import json
from pathlib import Path

from csbf.activation_trajectory import (
    ActivationTrajectoryConfig,
    run_activation_trajectory_split_seed_sweep,
    write_activation_trajectory_split_seed_sweep,
)
from csbf.schema import Observation, TraceRecord, save_jsonl


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


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    records = [_record(str(index), index % 2 == 0) for index in range(12)]
    records_path = tmp_path / "records.jsonl"
    features_path = tmp_path / "features.jsonl"
    save_jsonl(records, records_path)
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
                }
            )
    features_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return records_path, features_path


def test_activation_trajectory_wrapper_marks_diagnostic_family(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)

    summary = run_activation_trajectory_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(2),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=ActivationTrajectoryConfig(projection_dim=2, metric="aligned_change"),
    )

    assert summary["diagnostic_family"] == "activation_trajectory"
    assert summary["base_method"] == "latent_trajectory"
    assert summary["transform"]["metric"] == "aligned_change"
    assert summary["aggregate"]["mean_hmm_score_brier"] is not None


def test_write_activation_trajectory_split_seed_sweep_outputs_json_and_csv(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    summary = run_activation_trajectory_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(2),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=ActivationTrajectoryConfig(projection_dim=2),
    )
    written = write_activation_trajectory_split_seed_sweep(
        summary,
        tmp_path / "activation_split.json",
        tmp_path / "activation_split.csv",
    )

    assert Path(written["json"]).exists()
    assert Path(written["csv"]).exists()
    assert "activation_trajectory" in Path(written["json"]).read_text(encoding="utf-8")
