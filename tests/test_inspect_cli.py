import json
import subprocess
import sys
from pathlib import Path

from csbf.schema import Observation, TraceRecord, save_jsonl


def test_inspect_trace_jsonl_cli_outputs_warnings(tmp_path: Path):
    path = tmp_path / "one_trace.jsonl"
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
                observations=[Observation(step_index=0, text="step", score=1.0, concept_code=1)],
            )
        ],
        path,
    )

    completed = subprocess.run(
        [sys.executable, "-m", "scripts.inspect_trace_jsonl", str(path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["num_records"] == 1
    assert summary["warnings"]
