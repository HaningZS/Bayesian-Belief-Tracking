import importlib.util
import json
from pathlib import Path

from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rewrite_text_concepts.py"
    spec = importlib.util.spec_from_file_location("rewrite_text_concepts_cli", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rewrite_text_concepts_cli_writes_records_and_report(monkeypatch, tmp_path):
    cli = _load_cli_module()
    records = [
        TraceRecord(
            question_id="q1",
            question="question",
            trace_id="q1-t0",
            trace_text="trace",
            final_answer="1",
            gold_answer="1",
            correct=True,
            observations=[
                Observation(step_index=0, text="Let me verify the value.", score=0.7, concept_code="old"),
            ],
        )
    ]
    records_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "records_sv.jsonl"
    report_path = tmp_path / "report.json"
    save_jsonl(records, records_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rewrite_text_concepts",
            str(records_path),
            "--mode",
            "self_verification",
            "--output-records",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    cli.main()

    rewritten = load_jsonl(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rewritten[0].observations[0].concept_code == "sv_verification"
    assert report["concept_usage"]["sv_verification"] == 1
