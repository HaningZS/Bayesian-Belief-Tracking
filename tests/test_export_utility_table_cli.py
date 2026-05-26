import importlib.util
from pathlib import Path

from csbf.schema import Observation, TraceRecord, save_jsonl


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_utility_table.py"
    spec = importlib.util.spec_from_file_location("export_utility_table_cli", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(question_id: str, trace_index: int, correct: bool, score: float) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=f"{question_id}-t{trace_index}",
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(step_index=0, text="step 0", score=score, concept_code=1 if correct else 2),
            Observation(step_index=1, text="step 1", score=score, concept_code=1 if correct else 2),
            Observation(step_index=2, text="step 2", score=score, concept_code=1 if correct else 2),
        ],
    )


def test_export_utility_table_cli_writes_requested_outputs(monkeypatch, tmp_path):
    cli = _load_cli_module()
    records = []
    for index, correct in enumerate([True, False, True, False, True, False, True, False]):
        for trace_index in range(2):
            records.append(_record(f"q{index}", trace_index, correct, 0.8 if correct else 0.2))
    records_path = tmp_path / "records.jsonl"
    save_jsonl(records, records_path)
    output_json = tmp_path / "utility.json"
    output_csv = tmp_path / "utility.csv"
    output_markdown = tmp_path / "utility.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_utility_table",
            str(records_path),
            "--seed",
            "1",
            "--train-ratio",
            "0.5",
            "--calibration-ratio",
            "0.25",
            "--methods",
            "score_prefix,ema",
            "--false-positive-rates",
            "0.25",
            "--recalls",
            "0.5",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-markdown",
            str(output_markdown),
        ],
    )

    cli.main()

    assert output_json.exists()
    assert output_csv.exists()
    assert output_markdown.exists()
    markdown = output_markdown.read_text(encoding="utf-8")
    assert "Utility Table" in markdown
    assert "score_prefix" in markdown
