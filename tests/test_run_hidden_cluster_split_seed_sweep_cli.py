import importlib.util
import json
from pathlib import Path

from csbf.schema import Observation, TraceRecord, save_jsonl


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_hidden_cluster_split_seed_sweep.py"
    spec = importlib.util.spec_from_file_location("run_hidden_cluster_split_seed_sweep_cli", path)
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
            Observation(step_index=0, text="step 0", score=0.8 if correct else 0.2, concept_code="old"),
            Observation(step_index=1, text="step 1", score=0.7 if correct else 0.3, concept_code="old"),
        ],
    )


def test_run_hidden_cluster_split_seed_sweep_cli_writes_outputs(monkeypatch, tmp_path):
    cli = _load_cli_module()
    records = [_record(str(index), index % 2 == 0) for index in range(12)]
    records_path = tmp_path / "records.jsonl"
    features_path = tmp_path / "features.jsonl"
    output_json = tmp_path / "cluster_split.json"
    output_csv = tmp_path / "cluster_split.csv"
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
                    "hidden_last_token": [sign, sign * 2.0, -sign],
                    "hidden_layer_8_last_token": [sign * 3.0, -sign],
                }
            )
    features_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hidden_cluster_split_seed_sweep",
            str(records_path),
            str(features_path),
            "--seed-count",
            "2",
            "--train-ratio",
            "0.5",
            "--calibration-ratio",
            "0.25",
            "--cluster-count",
            "2",
            "--projection-dim",
            "4",
            "--feature-field",
            "hidden_layer_8_last_token",
            "--iterations",
            "2",
            "--combine-text-concept",
            "self_verification",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
    )

    cli.main()

    assert output_json.exists()
    assert output_csv.exists()
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["concept_join"]["text_concept_mode"] == "self_verification"
    assert summary["feature_rows"]["feature_field"] == "hidden_layer_8_last_token"
    assert "hmm_score_brier" in output_csv.read_text(encoding="utf-8")
