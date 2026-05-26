import json
from pathlib import Path

from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl
from csbf.text_concept_records import build_text_concept_records


def _records() -> list[TraceRecord]:
    return [
        TraceRecord(
            question_id="q1",
            question="question",
            trace_id="q1-t0",
            trace_text="trace",
            final_answer="1",
            gold_answer="1",
            correct=True,
            observations=[
                Observation(step_index=0, text="Let me double-check the result.", score=0.6, concept_code="old"),
                Observation(step_index=1, text="Therefore the answer is 1.", score=0.8, concept_code="old"),
            ],
        ),
        TraceRecord(
            question_id="q2",
            question="question",
            trace_id="q2-t0",
            trace_text="trace",
            final_answer="2",
            gold_answer="1",
            correct=False,
            observations=[
                Observation(step_index=0, text="Wait, this is wrong.", score=0.4, concept_code="old"),
                Observation(step_index=1, text="Alternatively, try another way.", score=0.3, concept_code="old"),
            ],
        ),
    ]


def test_build_text_concept_records_rewrites_self_verification_codes(tmp_path: Path):
    records_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "records_sv.jsonl"
    report_path = tmp_path / "report.json"
    save_jsonl(_records(), records_path)

    summary = build_text_concept_records(
        records_path,
        output_path=output_path,
        report_path=report_path,
        mode="self_verification",
    )

    rewritten = load_jsonl(output_path)
    codes = [observation.concept_code for record in rewritten for observation in record.observations]
    assert codes == ["sv_verification", "sv_none", "sv_correction", "sv_alternative"]
    assert all(record.metadata["concept_source"] == "self_verification_text" for record in rewritten)
    assert summary["concept_usage"]["sv_verification"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["outputs"]["records"] == str(output_path)
