import json
import subprocess
import sys
from pathlib import Path


def test_preobserved_pipeline_cli_runs_fixture():
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_preobserved_pipeline",
            "data/fixtures/preobserved_traces.jsonl",
            "--train-ratio",
            "0.34",
            "--calibration-ratio",
            "0.33",
            "--seed",
            "0",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["num_records"] == 6
    assert "hmm_hybrid" in result["methods"]


def test_preobserved_pipeline_cli_accepts_model_selection_and_calibration_mode():
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_preobserved_pipeline",
            "data/fixtures/preobserved_traces.jsonl",
            "--train-ratio",
            "0.34",
            "--calibration-ratio",
            "0.33",
            "--use-model-selection",
            "--calibration-mode",
            "final_step",
            "--seed",
            "0",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["num_records"] == 6
    assert "hmm_hybrid" in result["methods"]
