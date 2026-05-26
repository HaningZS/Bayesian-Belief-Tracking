import importlib.util
import json
import sys
from pathlib import Path


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_process_label_validation.py"
    spec = importlib.util.spec_from_file_location("run_process_label_validation_cli", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_process_label_validation_writes_summary_and_prefix_csv(monkeypatch, tmp_path):
    cli = _load_cli_module()
    records_path = tmp_path / "records.jsonl"
    labels_path = tmp_path / "process_labels.jsonl"
    records_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "Q1",
                        "trace_id": "t1",
                        "trace_text": "trace",
                        "final_answer": "0",
                        "gold_answer": "1",
                        "correct": False,
                        "observations": [
                            {"step_index": 0, "score": 0.9},
                            {"step_index": 1, "score": 0.2},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "question_id": "q2",
                        "question": "Q2",
                        "trace_id": "t2",
                        "trace_text": "trace",
                        "final_answer": "1",
                        "gold_answer": "1",
                        "correct": True,
                        "observations": [{"step_index": 0, "score": 0.8}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        "\n".join(
            [
                json.dumps({"trace_id": "t1", "first_error_step": 1}),
                json.dumps({"trace_id": "t2", "first_error_step": None}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "prefix.csv"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_process_label_validation",
            str(records_path),
            str(labels_path),
            "--output-json",
            str(output_json),
            "--output-prefix-csv",
            str(output_csv),
        ],
    )

    cli.main()

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["labeled_trace_count"] == 2
    assert report["positive_prefix_count"] == 1
    csv_text = output_csv.read_text(encoding="utf-8")
    assert "predicted_error_probability" in csv_text
    assert "t1" in csv_text
