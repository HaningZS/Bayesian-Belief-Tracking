import importlib.util
import json
from pathlib import Path

from csbf.schema import Observation, TraceRecord, save_jsonl


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_hidden_probe_split_seed_sweep.py"
    spec = importlib.util.spec_from_file_location("run_hidden_probe_split_seed_sweep_cli", path)
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
        ],
    )


def test_run_hidden_probe_split_seed_sweep_cli_writes_outputs(monkeypatch, tmp_path):
    cli = _load_cli_module()
    records = [_record(str(index), index % 2 == 0) for index in range(12)]
    records_path = tmp_path / "records.jsonl"
    features_path = tmp_path / "features.jsonl"
    output_json = tmp_path / "probe_split.json"
    output_csv = tmp_path / "probe_split.csv"
    save_jsonl(records, records_path)
    rows = []
    for record in records:
        sign = 1.0 if record.correct else -1.0
        for observation in record.observations:
            rows.append(
                {
                    "trace_id": record.trace_id,
                    "question_id": record.question_id,
                    "step_index": observation.step_index,
                    "hidden_last_token": [2.0 * sign, -1.0 * sign],
                    "hidden_mean_pool": [1.5 * sign, -0.5 * sign],
                }
            )
    features_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hidden_probe_split_seed_sweep",
            str(records_path),
            str(features_path),
            "--seed-count",
            "2",
            "--train-ratio",
            "0.5",
            "--calibration-ratio",
            "0.25",
            "--projection-dim",
            "4",
            "--epochs",
            "5",
            "--calibration-epochs",
            "10",
            "--feature-field",
            "hidden_mean_pool",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
    )

    cli.main()

    assert output_json.exists()
    assert output_csv.exists()
    assert "mean_hmm_score_brier" in output_csv.read_text(encoding="utf-8")
