"""Data schema for pre-observed reasoning traces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Observation:
    """One prefix observation at a reasoning step or chunk."""

    step_index: int
    text: str = ""
    score: float | None = None
    concept_code: int | str | None = None
    entropy: float | None = None
    score_delta: float | None = None

    def validate(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.score is not None and not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be in [0, 1]")
        if self.entropy is not None and float(self.entropy) < 0.0:
            raise ValueError("entropy must be non-negative")
        if self.score_delta is not None and not -1.0 <= float(self.score_delta) <= 1.0:
            raise ValueError("score_delta must be in [-1, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {
            "step_index": self.step_index,
            "text": self.text,
        }
        if self.score is not None:
            data["score"] = float(self.score)
        if self.concept_code is not None:
            data["concept_code"] = self.concept_code
        if self.entropy is not None:
            data["entropy"] = float(self.entropy)
        if self.score_delta is not None:
            data["score_delta"] = float(self.score_delta)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        observation = cls(
            step_index=int(data["step_index"]),
            text=str(data.get("text", "")),
            score=None if data.get("score") is None else float(data["score"]),
            concept_code=data.get("concept_code"),
            entropy=None if data.get("entropy") is None else float(data["entropy"]),
            score_delta=None if data.get("score_delta") is None else float(data["score_delta"]),
        )
        observation.validate()
        return observation


@dataclass(frozen=True)
class TraceRecord:
    """A generated reasoning trace and its prefix observations."""

    question_id: str
    question: str
    trace_id: str
    trace_text: str
    final_answer: str
    gold_answer: str
    correct: bool
    observations: list[Observation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for field_name in ("question_id", "question", "trace_id", "trace_text"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} is required")
        for observation in self.observations:
            observation.validate()
            if observation.score is None and observation.concept_code is None:
                raise ValueError("each observation must include score or concept_code")
        step_indexes = [observation.step_index for observation in self.observations]
        if step_indexes != sorted(step_indexes):
            raise ValueError("observations must be sorted by step_index")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {
            "question_id": self.question_id,
            "question": self.question,
            "trace_id": self.trace_id,
            "trace_text": self.trace_text,
            "final_answer": self.final_answer,
            "gold_answer": self.gold_answer,
            "correct": bool(self.correct),
            "observations": [observation.to_dict() for observation in self.observations],
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceRecord":
        record = cls(
            question_id=str(data["question_id"]),
            question=str(data["question"]),
            trace_id=str(data["trace_id"]),
            trace_text=str(data["trace_text"]),
            final_answer=str(data.get("final_answer", "")),
            gold_answer=str(data.get("gold_answer", "")),
            correct=bool(data["correct"]),
            observations=[Observation.from_dict(item) for item in data.get("observations", [])],
            metadata=dict(data.get("metadata", {})),
        )
        record.validate()
        return record


def load_jsonl(path: str | Path) -> list[TraceRecord]:
    """Load trace records from JSONL."""

    records: list[TraceRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(TraceRecord.from_dict(json.loads(stripped)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSONL record at line {line_number}: {error}") from error
    return records


def save_jsonl(records: list[TraceRecord], path: str | Path) -> None:
    """Save trace records to JSONL."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
