import importlib.util
import json
import sys
from pathlib import Path


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sample_answer_label_audit.py"
    spec = importlib.util.spec_from_file_location("sample_answer_label_audit_cli", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sample_answer_label_audit_writes_jsonl_csv_and_summary(monkeypatch, tmp_path):
    cli = _load_cli_module()
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "Q1",
                        "trace_id": "q1-t0",
                        "trace_text": "work \\boxed{1/2}",
                        "final_answer": "1/2",
                        "gold_answer": "0.5",
                        "correct": False,
                        "observations": [{"step_index": 0, "score": 0.2}],
                    }
                ),
                json.dumps(
                    {
                        "question_id": "q2",
                        "question": "Q2",
                        "trace_id": "q2-t0",
                        "trace_text": "unfinished",
                        "final_answer": "",
                        "gold_answer": "7",
                        "correct": False,
                        "observations": [{"step_index": 0, "score": 0.4}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "audit.jsonl"
    output_csv = tmp_path / "audit.csv"
    summary_json = tmp_path / "summary.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "sample_answer_label_audit",
            str(records_path),
            "--max-per-category",
            "2",
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
        ],
    )

    cli.main()

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["audit_id"]
    assert "categories" in rows[0]
    assert "incorrect_with_boxed_answer" in output_csv.read_text(encoding="utf-8")
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["input_records"] == 2
    assert summary["audit_rows"] == 2
    assert summary["category_counts"]["possible_equivalent_answer"] == 1
