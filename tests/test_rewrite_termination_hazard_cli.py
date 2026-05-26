import json
import subprocess
import sys
from pathlib import Path

from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl


def test_rewrite_termination_hazard_cli_writes_records_and_report(tmp_path: Path):
    records_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "records_termination_hazard.jsonl"
    report_path = tmp_path / "termination_hazard_report.json"
    save_jsonl(
        [
            TraceRecord(
                question_id="q1",
                question="question",
                trace_id="q1-t0",
                trace_text="trace",
                final_answer="1",
                gold_answer="1",
                correct=True,
                observations=[
                    Observation(step_index=0, text="step 0", score=0.2, concept_code="old"),
                    Observation(step_index=1, text="step 1", score=0.9, concept_code="old"),
                ],
            )
        ],
        records_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rewrite_termination_hazard",
            str(records_path),
            "--horizon-observations",
            "2",
            "--output-records",
            str(output_path),
            "--report",
            str(report_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    rewritten = load_jsonl(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stdout_summary = json.loads(completed.stdout)
    assert [observation.score for observation in rewritten[0].observations] == [0.5, 0.0]
    assert rewritten[0].metadata["termination_hazard"]["horizon_observations"] == 2
    assert report["score_summary"]["max"] == 0.5
    assert stdout_summary["outputs"]["report"] == str(report_path)
