import importlib.util
import json
import sys
from pathlib import Path

from csbf.prefix_rollouts import build_prefix_rollout_tasks, write_jsonl_rows
from csbf.schema import Observation, TraceRecord, save_jsonl


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_prefix_belief_predictions.py"
    spec = importlib.util.spec_from_file_location("export_prefix_belief_predictions_cli", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records() -> list[TraceRecord]:
    rows: list[TraceRecord] = []
    for index in range(5):
        correct = index % 2 == 0
        score = 0.75 if correct else 0.25
        rows.append(
            TraceRecord(
                question_id=f"q{index}",
                question=f"What is {index}?",
                trace_id=f"q{index}-trace",
                trace_text="first\nsecond\nthird",
                final_answer=str(index),
                gold_answer=str(index),
                correct=correct,
                observations=[
                    Observation(step_index=0, text="first", score=score, concept_code="a"),
                    Observation(step_index=1, text="second", score=score, concept_code="b"),
                    Observation(step_index=2, text="third", score=score, concept_code="c"),
                ],
            )
        )
    return rows


def test_export_prefix_belief_predictions_cli_writes_predictions_and_summary(tmp_path, monkeypatch):
    cli = _load_cli_module()
    records = _records()
    tasks = build_prefix_rollout_tasks(records, prefix_fractions=[0.5], rollouts_per_prefix=1)
    records_path = tmp_path / "records.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    output_jsonl = tmp_path / "predictions.jsonl"
    summary_json = tmp_path / "summary.json"
    save_jsonl(records, records_path)
    write_jsonl_rows(tasks, tasks_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_prefix_belief_predictions",
            str(records_path),
            str(tasks_path),
            "--train-ratio",
            "0.4",
            "--calibration-ratio",
            "0.2",
            "--seed",
            "0",
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
        ],
    )

    cli.main()

    prediction_rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert prediction_rows
    assert all("sbbt_belief" in row for row in prediction_rows)
    assert all(row["prediction_method"] == "hmm_hybrid" for row in prediction_rows)
    assert summary["prediction_count"] == len(prediction_rows)
