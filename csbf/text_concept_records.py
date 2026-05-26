"""Rewrite trace concept codes from observation text."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from csbf.concepts import self_verification_concept_code, text_concept_code
from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl

_MODES = ("text_pattern", "self_verification")


def build_text_concept_records(
    records_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    mode: str = "text_pattern",
) -> dict[str, Any]:
    """Rewrite `concept_code` values using deterministic text markers."""

    checked_mode = _validate_mode(mode)
    records = load_jsonl(records_path)
    rewritten: list[TraceRecord] = []
    usage: Counter[str] = Counter()
    coder = _coder_for_mode(checked_mode)
    for record in records:
        observations: list[Observation] = []
        for observation in record.observations:
            code = coder(observation.text)
            usage[code] += 1
            observations.append(
                Observation(
                    step_index=observation.step_index,
                    text=observation.text,
                    score=observation.score,
                    concept_code=code,
                    entropy=observation.entropy,
                    score_delta=observation.score_delta,
                )
            )
        metadata = dict(record.metadata)
        metadata["concept_source"] = _concept_source(checked_mode)
        rewritten.append(
            TraceRecord(
                question_id=record.question_id,
                question=record.question,
                trace_id=record.trace_id,
                trace_text=record.trace_text,
                final_answer=record.final_answer,
                gold_answer=record.gold_answer,
                correct=record.correct,
                observations=observations,
                metadata=metadata,
            )
        )
    save_jsonl(rewritten, output_path)
    summary: dict[str, Any] = {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "mode": checked_mode,
        "concept_source": _concept_source(checked_mode),
        "concept_usage": dict(sorted(usage.items())),
        "outputs": {"records": str(output_path)},
    }
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        summary["outputs"]["report"] = str(report)
        report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _validate_mode(mode: str) -> str:
    value = str(mode).strip()
    if value not in _MODES:
        raise ValueError(f"mode must be one of {', '.join(_MODES)}")
    return value


def _coder_for_mode(mode: str):
    if mode == "text_pattern":
        return text_concept_code
    if mode == "self_verification":
        return self_verification_concept_code
    raise ValueError(f"unsupported mode: {mode}")


def _concept_source(mode: str) -> str:
    if mode == "text_pattern":
        return "text_pattern"
    if mode == "self_verification":
        return "self_verification_text"
    raise ValueError(f"unsupported mode: {mode}")
