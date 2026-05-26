"""Validation helpers for external process/error-onset labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from csbf.evaluation import evaluate_predictions
from csbf.schema import TraceRecord


@dataclass(frozen=True)
class ProcessLabel:
    """External process label for one trace."""

    trace_id: str
    first_error_step: int | None = None

    def validate(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id is required")
        if self.first_error_step is not None and self.first_error_step < 0:
            raise ValueError("first_error_step must be non-negative or null")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessLabel":
        raw_step = data.get("first_error_step", data.get("error_step"))
        label = cls(
            trace_id=str(data["trace_id"]),
            first_error_step=None if raw_step is None else int(raw_step),
        )
        label.validate()
        return label


@dataclass(frozen=True)
class ProcessPrefixExample:
    """One prefix-level example for process-label validation."""

    trace_id: str
    question_id: str
    step_index: int
    error_started: int
    predicted_error_probability: float
    observation_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "question_id": self.question_id,
            "step_index": self.step_index,
            "error_started": self.error_started,
            "predicted_error_probability": self.predicted_error_probability,
            "observation_score": self.observation_score,
        }


def load_process_labels(path: str | Path) -> dict[str, ProcessLabel]:
    """Load process labels from JSONL keyed by trace_id."""

    labels: dict[str, ProcessLabel] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                label = ProcessLabel.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid process label at line {line_number}: {error}") from error
            labels[label.trace_id] = label
    return labels


def build_process_prefix_examples(
    records: list[TraceRecord],
    labels: dict[str, ProcessLabel],
) -> list[ProcessPrefixExample]:
    """Align external first-error labels to trace prefix observations."""

    examples: list[ProcessPrefixExample] = []
    for record in records:
        label = labels.get(record.trace_id)
        if label is None:
            continue
        for observation in record.observations:
            if observation.score is None:
                continue
            error_started = int(label.first_error_step is not None and observation.step_index >= label.first_error_step)
            score = float(observation.score)
            examples.append(
                ProcessPrefixExample(
                    trace_id=record.trace_id,
                    question_id=record.question_id,
                    step_index=observation.step_index,
                    error_started=error_started,
                    predicted_error_probability=round(1.0 - score, 12),
                    observation_score=score,
                )
            )
    return examples


def evaluate_process_labels(
    records: list[TraceRecord],
    labels: dict[str, ProcessLabel],
) -> dict[str, Any]:
    """Evaluate prefix scores against external process/error-onset labels."""

    examples = build_process_prefix_examples(records, labels)
    y_true = [example.error_started for example in examples]
    y_pred = [example.predicted_error_probability for example in examples]
    labeled_trace_ids = {record.trace_id for record in records if record.trace_id in labels}
    missing = sorted(record.trace_id for record in records if record.trace_id not in labels)
    return {
        "trace_count": len(records),
        "labeled_trace_count": len(labeled_trace_ids),
        "prefix_count": len(examples),
        "positive_prefix_count": sum(y_true),
        "negative_prefix_count": len(y_true) - sum(y_true),
        "missing_label_trace_ids": missing,
        "metrics": evaluate_predictions(y_true, y_pred) if examples else {},
    }
