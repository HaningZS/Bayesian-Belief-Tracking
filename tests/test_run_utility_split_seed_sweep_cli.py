import importlib.util
from pathlib import Path

from csbf.schema import Observation, TraceRecord, save_jsonl


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_utility_split_seed_sweep.py"
    spec = importlib.util.spec_from_file_location("run_utility_split_seed_sweep_cli", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(question_id: str, correct: bool, scores: list[float]) -> TraceRecord:
    return TraceRecord(
        question_id=question_id,
        question="question",
        trace_id=f"{question_id}-t0",
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(
                step_index=index,
                text=f"step {index}",
                score=score,
                concept_code=1 if correct else 2,
            )
            for index, score in enumerate(scores)
        ],
    )


def test_run_utility_split_seed_sweep_cli_writes_outputs(monkeypatch, tmp_path):
    cli = _load_cli_module()
    records_path = tmp_path / "records.jsonl"
    save_jsonl(
        [
            _record("q1", True, [0.9, 0.9, 0.8]),
            _record("q2", True, [0.8, 0.7, 0.8]),
            _record("q3", True, [0.7, 0.8, 0.9]),
            _record("q4", False, [0.5, 0.3, 0.2]),
            _record("q5", False, [0.6, 0.4, 0.2]),
            _record("q6", False, [0.4, 0.2, 0.1]),
        ],
        records_path,
    )
    output_json = tmp_path / "utility_sweep.json"
    output_csv = tmp_path / "utility_sweep.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_utility_split_seed_sweep",
            str(records_path),
            "--seed-count",
            "2",
            "--methods",
            "score_prefix,ema",
            "--false-positive-rates",
            "0.25",
            "--recalls",
            "0.5",
            "--train-ratio",
            "0.5",
            "--calibration-ratio",
            "0.25",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
    )

    cli.main()

    assert output_json.exists()
    assert output_csv.exists()
    assert "aggregate_rows" in output_json.read_text(encoding="utf-8")
    assert "target_hit_fraction" in output_csv.read_text(encoding="utf-8")
