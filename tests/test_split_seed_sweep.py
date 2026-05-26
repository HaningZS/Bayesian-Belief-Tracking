from pathlib import Path

from csbf.schema import Observation, TraceRecord, save_jsonl
from csbf.split_seed_sweep import run_split_seed_sweep, write_split_seed_sweep


def _record(question_id: str, trace_index: int, correct: bool, score: float, code: int) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=f"{question_id}-t{trace_index}",
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(step_index=0, text="step 0", score=score, concept_code=code),
            Observation(step_index=1, text="step 1", score=score, concept_code=code),
            Observation(step_index=2, text="step 2", score=score, concept_code=code),
        ],
    )


def _records() -> list[TraceRecord]:
    records: list[TraceRecord] = []
    for qid, correct, score, code in [
        ("q1", True, 0.9, 1),
        ("q2", True, 0.8, 1),
        ("q3", True, 0.7, 1),
        ("q4", False, 0.3, 2),
        ("q5", False, 0.2, 2),
        ("q6", False, 0.1, 2),
    ]:
        for trace_index in range(2):
            records.append(_record(qid, trace_index, correct, score, code))
    return records


def test_run_split_seed_sweep_summarizes_method_stability():
    summary = run_split_seed_sweep(
        _records(),
        seeds=[0, 1, 2],
        train_ratio=0.5,
        calibration_ratio=0.25,
        use_model_selection=True,
    )

    assert summary["seed_count"] == 3
    assert summary["valid_auroc_seed_count"] >= 1
    assert summary["aggregate"]["mean_hmm_vs_baseline_auroc_gap"] is not None
    assert summary["aggregate"]["mean_calibrated_last_step_brier"] is not None
    assert summary["aggregate"]["hmm_score_brier_minus_calibrated_last_step"] is not None
    assert summary["aggregate"]["median_hmm_score_brier_minus_calibrated_last_step"] is not None
    assert summary["aggregate"]["negative_hmm_score_brier_minus_calibrated_last_step_fraction"] is not None
    assert summary["aggregate"]["mean_hmm_score_ece"] is not None
    assert summary["aggregate"]["hmm_score_ece_minus_ema"] is not None
    assert summary["aggregate"]["hmm_score_ece_minus_calibrated_last_step"] is not None
    assert summary["aggregate"]["mean_hmm_hybrid_auprc"] is not None
    assert summary["aggregate"]["hmm_hybrid_auprc_minus_ema"] is not None
    assert summary["aggregate"]["positive_hmm_vs_baseline_auroc_gap_fraction"] is not None
    assert summary["aggregate"]["mean_hmm_hybrid_auroc_minus_ema"] is not None
    assert len(summary["seeds"]) == 3
    assert {"train", "calibration", "test"} <= set(summary["seeds"][0]["split"])
    valid_seed = next(row for row in summary["seeds"] if row["best_baseline_method"] is not None)
    assert valid_seed["best_baseline_method"] in {
        "calibrated_last_step",
        "ema",
        "last_step",
        "moving_average",
        "mean_score",
        "temporal_metric",
    }
    assert valid_seed["best_online_hmm_method"] in {
        "hmm_score",
        "hmm_concept",
        "hmm_hybrid",
        "hmm_joint",
        "hmm_nonstationary",
    }


def test_run_split_seed_sweep_counts_prefix_feature_classifier_as_baseline():
    records: list[TraceRecord] = []
    for index, correct in enumerate([True, False, True, False, True, False, True, False]):
        code = "high" if correct else "low"
        for trace_index in range(2):
            records.append(_record(f"q{index}", trace_index, correct, 0.5, code))

    summary = run_split_seed_sweep(
        records,
        seeds=[0],
        train_ratio=0.5,
        calibration_ratio=0.25,
    )

    seed_row = summary["seeds"][0]
    assert seed_row["best_baseline_method"] == "prefix_feature_classifier"
    assert seed_row["method_metrics"]["prefix_feature_classifier"]["auroc"] == 1.0


def test_write_split_seed_sweep_writes_json_and_csv(tmp_path: Path):
    traces_path = tmp_path / "traces.jsonl"
    save_jsonl(_records(), traces_path)

    summary = run_split_seed_sweep(traces_path, seeds=[0, 1], train_ratio=0.5, calibration_ratio=0.25)
    written = write_split_seed_sweep(summary, tmp_path / "sweep.json", tmp_path / "sweep.csv")

    assert Path(written["json"]).exists()
    assert Path(written["csv"]).exists()
    assert "mean_hmm_vs_baseline_auroc_gap" in Path(written["json"]).read_text(encoding="utf-8")
    csv_text = Path(written["csv"]).read_text(encoding="utf-8")
    assert "best_online_hmm_method" in csv_text
    assert "calibrated_last_step_brier" in csv_text
    assert "hmm_score_brier_minus_calibrated_last_step" in csv_text
    assert "hmm_score_ece_minus_ema" in csv_text
    assert "hmm_hybrid_auprc_minus_ema" in csv_text
    assert "hmm_hybrid_auroc_minus_ema" in csv_text


def test_run_split_seed_sweep_defaults_to_no_model_selection():
    summary = run_split_seed_sweep(
        _records(),
        seeds=[0],
        train_ratio=0.5,
        calibration_ratio=0.25,
    )

    assert summary["use_model_selection"] is False
    assert summary["seeds"][0]["selected_hmm_config"] is None
