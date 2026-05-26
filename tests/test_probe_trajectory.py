import json
from pathlib import Path

from csbf.probe_trajectory import (
    ProbeTrajectoryConfig,
    build_probe_trajectory_records,
    run_probe_trajectory_split_seed_sweep,
    write_probe_trajectory_split_seed_sweep,
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
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
            if record.correct
            else [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
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
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    records = [_record(str(index), index % 2 == 0) for index in range(12)]
    records_path = tmp_path / "records.jsonl"
    features_path = tmp_path / "features.jsonl"
    save_jsonl(records, records_path)
    _write_features(features_path, records)
    return records_path, features_path


def _config(metric: str = "logit_delta") -> ProbeTrajectoryConfig:
    return ProbeTrajectoryConfig(
        projection_dim=8,
        epochs=30,
        learning_rate=0.2,
        calibration_epochs=0,
        metric=metric,
    )


def test_build_probe_trajectory_records_rewrites_scores_from_probe_logit_dynamics(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    output_path = tmp_path / "records_probe_traj.jsonl"
    report_path = tmp_path / "probe_traj_report.json"

    summary = build_probe_trajectory_records(
        records_path,
        features_path,
        output_path=output_path,
        report_path=report_path,
        train_ratio=0.5,
        calibration_ratio=0.25,
        seed=0,
        config=_config("probe_logit"),
    )

    rewritten = load_jsonl(output_path)
    final_scores = {record.trace_id: record.observations[-1].score for record in rewritten}
    correct_scores = [final_scores[record.trace_id] for record in rewritten if record.correct]
    incorrect_scores = [final_scores[record.trace_id] for record in rewritten if not record.correct]
    assert min(correct_scores) > max(incorrect_scores)
    assert all(0.0 <= float(score) <= 1.0 for score in final_scores.values())
    assert all(record.metadata["score_source"] == "probe_trajectory" for record in rewritten)
    assert summary["probe_trajectory"]["metric"] == "probe_logit"
    assert report_path.exists()


def test_run_probe_trajectory_split_seed_sweep_evaluates_rewritten_scores(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)

    summary = run_probe_trajectory_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(3),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=_config("probe_logit"),
    )

    assert summary["seed_count"] == 3
    assert summary["aggregate"]["mean_hmm_score_brier"] is not None
    assert all(row["probe_trajectory_final_metrics"]["test"]["metrics"]["auroc"] == 1.0 for row in summary["seeds"])
    assert all("hmm_score" in row["method_metrics"] for row in summary["seeds"])


def test_write_probe_trajectory_split_seed_sweep_outputs_json_and_csv(tmp_path: Path):
    records_path, features_path = _fixture_paths(tmp_path)
    summary = run_probe_trajectory_split_seed_sweep(
        records_path,
        features_path,
        seeds=range(2),
        train_ratio=0.5,
        calibration_ratio=0.25,
        config=_config("probe_logit"),
    )
    written = write_probe_trajectory_split_seed_sweep(
        summary,
        tmp_path / "probe_traj_split.json",
        tmp_path / "probe_traj_split.csv",
    )

    assert Path(written["json"]).exists()
    assert Path(written["csv"]).exists()
    assert "mean_hmm_score_brier" in Path(written["csv"]).read_text(encoding="utf-8")
