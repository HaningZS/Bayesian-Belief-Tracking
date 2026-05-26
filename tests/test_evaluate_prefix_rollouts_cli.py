import json
import importlib.util
import sys
from pathlib import Path


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_prefix_rollouts.py"
    spec = importlib.util.spec_from_file_location("evaluate_prefix_rollouts_cli", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_prefix_rollouts_cli_writes_json_csv_and_markdown(tmp_path, monkeypatch):
    cli = _load_cli_module()
    tasks_path = tmp_path / "tasks.jsonl"
    results_path = tmp_path / "results.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"
    output_md = tmp_path / "summary.md"

    task_rows = [
        {
            "task_id": "t0-r0",
            "task_group_id": "t0",
            "trace_id": "trace",
            "question_id": "q",
            "step_index": 0,
            "rollout_index": 0,
            "source_score": 0.25,
            "actual_prefix_fraction": 0.5,
            "source_correct": False,
        },
        {
            "task_id": "t0-r1",
            "task_group_id": "t0",
            "trace_id": "trace",
            "question_id": "q",
            "step_index": 0,
            "rollout_index": 1,
            "source_score": 0.25,
            "actual_prefix_fraction": 0.5,
            "source_correct": False,
        },
    ]
    result_rows = [
        {"task_id": "t0-r0", "correct": False},
        {"task_id": "t0-r1", "correct": True},
    ]
    prediction_rows = [
        {"trace_id": "trace", "step_index": 0, "sbbt_belief": 0.45},
    ]
    tasks_path.write_text("\n".join(json.dumps(row) for row in task_rows) + "\n", encoding="utf-8")
    results_path.write_text("\n".join(json.dumps(row) for row in result_rows) + "\n", encoding="utf-8")
    predictions_path.write_text("\n".join(json.dumps(row) for row in prediction_rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_prefix_rollouts",
            str(tasks_path),
            str(results_path),
            "--prediction-jsonl",
            str(predictions_path),
            "--prediction-field",
            "sbbt_belief",
            "--prediction-name",
            "belief",
            "--require-prediction",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-markdown",
            str(output_md),
        ],
    )

    cli.main()

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["group_count"] == 1
    assert summary["groups"][0]["rollout_success_rate"] == 0.5
    assert summary["groups"][0]["belief"] == 0.45
    assert summary["belief_coverage_count"] == 1
    assert "belief_brier_against_rollout_success" in output_csv.read_text(encoding="utf-8")
    assert "score_brier_against_rollout_success" in output_csv.read_text(encoding="utf-8")
    markdown = output_md.read_text(encoding="utf-8")
    assert "Prefix Rollout Validation" in markdown
    assert "Belief Brier vs rollout success" in markdown
