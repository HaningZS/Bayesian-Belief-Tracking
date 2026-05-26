import json
from pathlib import Path

from csbf.schema import Observation, TraceRecord, save_jsonl
from csbf import utility_table as utility_module
from csbf.utility_table import (
    build_utility_split_seed_sweep,
    build_utility_table,
    write_utility_split_seed_sweep,
    write_utility_table,
)


def _record(question_id: str, trace_index: int, correct: bool, scores: list[float]) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=f"{question_id}-t{trace_index}",
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(
                step_index=index,
                text=f"step {index}",
                score=score,
                concept_code=1 if correct else 2,
            )
            for index, score in enumerate(scores)
        ],
    )


def _records() -> list[TraceRecord]:
    rows: list[TraceRecord] = []
    settings = [
        ("q0", True, [0.9, 0.9, 0.8]),
        ("q1", False, [0.8, 0.4, 0.2]),
        ("q2", True, [0.7, 0.8, 0.9]),
        ("q3", False, [0.6, 0.3, 0.1]),
        ("q4", True, [0.8, 0.8, 0.7]),
        ("q5", False, [0.5, 0.2, 0.2]),
        ("q6", True, [0.6, 0.7, 0.8]),
        ("q7", False, [0.7, 0.5, 0.3]),
        ("q8", True, [0.9, 0.7, 0.8]),
        ("q9", False, [0.4, 0.3, 0.2]),
        ("q10", True, [0.8, 0.9, 0.9]),
        ("q11", False, [0.7, 0.6, 0.2]),
    ]
    for question_id, correct, scores in settings:
        for trace_index in range(2):
            rows.append(_record(question_id, trace_index, correct, scores))
    return rows


def test_build_utility_table_reports_threshold_rows():
    summary = build_utility_table(
        _records(),
        seed=1,
        train_ratio=0.5,
        calibration_ratio=0.25,
        methods=("score_prefix", "ema", "moving_average", "hmm_hybrid"),
        false_positive_rates=(0.25,),
        recalls=(0.5,),
    )

    assert summary["rows"]
    assert {row["selection_rule"] for row in summary["rows"]} == {"max_fpr", "min_recall"}
    assert {row["method"] for row in summary["rows"]} == {
        "score_prefix",
        "ema",
        "moving_average",
        "hmm_hybrid",
    }
    assert all(0.0 <= row["threshold"] <= 1.0 for row in summary["rows"])
    assert all(0.0 <= row["false_positive_rate"] <= 1.0 for row in summary["rows"])
    assert all(0.0 <= row["recall"] <= 1.0 for row in summary["rows"])
    assert all(0.0 <= row["mean_compute_saved"] <= 1.0 for row in summary["rows"])


def test_threshold_metrics_by_threshold_matches_individual_metrics():
    record_risks = [
        {"trace_id": "a", "question_id": "q1", "correct": True, "risks": [0.1, 0.4, 0.2]},
        {"trace_id": "b", "question_id": "q2", "correct": False, "risks": [0.2, 0.8, 0.6]},
        {"trace_id": "c", "question_id": "q3", "correct": False, "risks": [0.5, 0.3, 0.7]},
        {"trace_id": "d", "question_id": "q4", "correct": True, "risks": [0.0, 0.2, 0.3]},
    ]
    thresholds = utility_module._candidate_thresholds(record_risks)

    batched = utility_module._threshold_metrics_by_threshold(record_risks, thresholds)

    assert set(batched) == set(thresholds)
    for threshold in thresholds:
        assert batched[threshold] == utility_module._threshold_metrics(record_risks, threshold)


def test_write_utility_table_writes_json_csv_and_markdown(tmp_path: Path):
    records_path = tmp_path / "records.jsonl"
    save_jsonl(_records(), records_path)
    summary = build_utility_table(
        records_path,
        seed=1,
        train_ratio=0.5,
        calibration_ratio=0.25,
        methods=("score_prefix", "ema"),
        false_positive_rates=(0.25,),
        recalls=(0.5,),
    )

    written = write_utility_table(
        summary,
        output_json=tmp_path / "utility.json",
        output_csv=tmp_path / "utility.csv",
        output_markdown=tmp_path / "utility.md",
    )

    payload = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert payload["rows"][0]["selection_rule"] in {"max_fpr", "min_recall"}
    assert "mean_compute_saved" in Path(written["csv"]).read_text(encoding="utf-8")
    markdown = Path(written["markdown"]).read_text(encoding="utf-8")
    assert "Utility Table" in markdown
    assert "score_prefix" in markdown


def test_build_utility_split_seed_sweep_aggregates_operating_points():
    summary = build_utility_split_seed_sweep(
        _records(),
        seeds=[0, 1, 2],
        train_ratio=0.5,
        calibration_ratio=0.25,
        methods=("score_prefix", "ema"),
        false_positive_rates=(0.25,),
        recalls=(0.5,),
    )

    assert summary["seed_count"] == 3
    assert summary["valid_seed_count"] == 3
    assert len(summary["seeds"]) == 3
    assert {"train", "calibration", "test"} <= set(summary["seeds"][0]["split"])
    assert summary["aggregate_rows"]
    key_rows = {
        (row["method"], row["selection_rule"], row["target_false_positive_rate"], row["target_recall"])
        for row in summary["aggregate_rows"]
    }
    assert ("score_prefix", "max_fpr", 0.25, None) in key_rows
    assert ("ema", "min_recall", None, 0.5) in key_rows
    for row in summary["aggregate_rows"]:
        assert 0.0 <= row["mean_false_positive_rate"] <= 1.0
        assert 0.0 <= row["mean_recall"] <= 1.0
        assert 0.0 <= row["mean_compute_saved"] <= 1.0
        assert row["valid_seed_count"] == 3
    max_fpr_row = next(
        row
        for row in summary["aggregate_rows"]
        if row["method"] == "score_prefix" and row["selection_rule"] == "max_fpr"
    )
    assert 0.0 <= max_fpr_row["target_hit_fraction"] <= 1.0


def test_write_utility_split_seed_sweep_writes_json_and_csv(tmp_path: Path):
    summary = build_utility_split_seed_sweep(
        _records(),
        seeds=[0, 1],
        train_ratio=0.5,
        calibration_ratio=0.25,
        methods=("score_prefix",),
        false_positive_rates=(0.25,),
        recalls=(0.5,),
    )

    written = write_utility_split_seed_sweep(
        summary,
        output_json=tmp_path / "utility_sweep.json",
        output_csv=tmp_path / "utility_sweep.csv",
    )

    payload = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert payload["seed_count"] == 2
    csv_text = Path(written["csv"]).read_text(encoding="utf-8")
    assert "target_hit_fraction" in csv_text
    assert "mean_incorrect_compute_saved" in csv_text
