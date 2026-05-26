import importlib.util
import json
from pathlib import Path

from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_latent_trajectory_baseline.py"
    spec = importlib.util.spec_from_file_location("run_latent_trajectory_baseline_cli", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(question_id: str, correct: bool) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=f"{question_id}-t0",
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(step_index=0, text="step 0", score=0.5, concept_code=1),
            Observation(step_index=1, text="step 1", score=0.5, concept_code=1),
            Observation(step_index=2, text="step 2", score=0.5, concept_code=1),
        ],
    )


def test_run_latent_trajectory_baseline_cli_writes_records_and_report(monkeypatch, tmp_path):
    cli = _load_cli_module()
    records = [_record(str(index), index % 2 == 0) for index in range(12)]
    records_path = tmp_path / "records.jsonl"
    features_path = tmp_path / "features.jsonl"
    output_path = tmp_path / "records_lt.jsonl"
    report_path = tmp_path / "lt_report.json"
    save_jsonl(records, records_path)
    rows = []
    for record in records:
        vectors = (
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
            if record.correct
            else [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        )
        for observation, vector in zip(record.observations, vectors, strict=True):
            rows.append(
                {
                    "trace_id": record.trace_id,
                    "question_id": record.question_id,
                    "step_index": observation.step_index,
                    "hidden_last_token": vector,
                }
            )
    features_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_latent_trajectory_baseline",
            str(records_path),
            str(features_path),
            "--output-records",
            str(output_path),
            "--report",
            str(report_path),
            "--train-ratio",
            "0.5",
            "--calibration-ratio",
            "0.25",
            "--projection-dim",
            "2",
            "--metric",
            "composite",
        ],
    )

    cli.main()

    rewritten = load_jsonl(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(rewritten) == len(records)
    assert report["outputs"]["records"] == str(output_path)
    assert report["feature_rows"]["used"] == 36
