import json
import importlib.util
import sys
from pathlib import Path

from csbf.schema import Observation, TraceRecord, save_jsonl
from csbf.split import split_by_question


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_prefix_rollout_tasks.py"
    spec = importlib.util.spec_from_file_location("export_prefix_rollout_tasks_cli", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_prefix_rollout_tasks_cli_writes_jsonl_and_summary(tmp_path, monkeypatch, capsys):
    cli = _load_cli_module()

    records_path = tmp_path / "records.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    summary_path = tmp_path / "summary.json"
    save_jsonl(
        [
            TraceRecord(
                question_id="q1",
                question="What is 40 + 2?",
                trace_id="q1-local-0",
                trace_text="a\nb\nc",
                final_answer="42",
                gold_answer="42",
                correct=True,
                observations=[
                    Observation(step_index=0, text="a", score=0.25),
                    Observation(step_index=1, text="b", score=0.5),
                    Observation(step_index=2, text="c", score=0.75),
                ],
                metadata={"dataset": "math_jsonl"},
            )
        ],
        records_path,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_prefix_rollout_tasks",
            str(records_path),
            "--output-jsonl",
            str(tasks_path),
            "--summary-json",
            str(summary_path),
            "--prefix-fraction",
            "0.34",
            "--prefix-fraction",
            "1.0",
            "--rollouts-per-prefix",
            "2",
        ],
    )

    cli.main()

    stdout = json.loads(capsys.readouterr().out)
    task_rows = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert stdout["task_count"] == 4
    assert summary["prefix_group_count"] == 2
    assert task_rows[0]["task_group_id"] == "q1-local-0|step=1"
    assert task_rows[0]["prompt_dataset_type"] == "math"
    assert task_rows[-1]["step_index"] == 2


def test_export_prefix_rollout_tasks_cli_can_filter_to_test_split(tmp_path, monkeypatch):
    cli = _load_cli_module()
    records = [
        TraceRecord(
            question_id=f"q{index}",
            question=f"What is {index}?",
            trace_id=f"q{index}-trace",
            trace_text="a\nb\nc",
            final_answer=str(index),
            gold_answer=str(index),
            correct=index % 2 == 0,
            observations=[
                Observation(step_index=0, text="a", score=0.25),
                Observation(step_index=1, text="b", score=0.5),
                Observation(step_index=2, text="c", score=0.75),
            ],
        )
        for index in range(5)
    ]
    expected_test_trace_ids = {
        record.trace_id
        for record in split_by_question(records, train_ratio=0.4, calibration_ratio=0.2, seed=0)["test"]
    }
    records_path = tmp_path / "records.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    save_jsonl(records, records_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_prefix_rollout_tasks",
            str(records_path),
            "--output-jsonl",
            str(tasks_path),
            "--prefix-fraction",
            "0.5",
            "--rollouts-per-prefix",
            "1",
            "--train-ratio",
            "0.4",
            "--calibration-ratio",
            "0.2",
            "--seed",
            "0",
            "--split-filter",
            "test",
        ],
    )

    cli.main()

    task_rows = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()]
    assert {row["trace_id"] for row in task_rows} == expected_test_trace_ids
