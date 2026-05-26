"""JSONL loading helpers for externally prepared math benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from csbf.answers import extract_final_answer
from csbf.math_dataset import MATHExample


_ID_FIELDS = ("question_id", "id", "uid", "problem_id")
_QUESTION_FIELDS = ("question", "problem", "prompt", "input")
_ANSWER_FIELDS = ("gold_answer", "answer", "target", "final_answer", "solution")
_LEVEL_FIELDS = ("level", "difficulty", "category", "source")


def load_math_jsonl_examples(
    path: str | Path,
    *,
    limit: int | None = None,
    offset: int = 0,
    levels: list[str] | None = None,
    id_field: str | None = None,
    question_field: str | None = None,
    answer_field: str | None = None,
    level_field: str | None = None,
) -> list[MATHExample]:
    """Load math-style examples from a local JSONL file.

    This is intentionally permissive so prepared GSM8K, AIME, Omni-MATH,
    OlymMATH, or RIMO slices can use a shared local runner path.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    path = Path(path)
    allowed_levels = {_normalize_level(level) for level in levels or []}
    examples: list[MATHExample] = []
    matched = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_index + 1} must be an object.")
            level = _normalize_level(_field_value(row, level_field, _LEVEL_FIELDS, required=False))
            if allowed_levels and level not in allowed_levels:
                continue
            if matched < offset:
                matched += 1
                continue
            matched += 1
            if limit is not None and len(examples) >= limit:
                break
            question_id = _field_value(row, id_field, _ID_FIELDS, required=False)
            question = _field_value(row, question_field, _QUESTION_FIELDS, required=True, label="question")
            answer = _field_value(row, answer_field, _ANSWER_FIELDS, required=True, label="answer")
            examples.append(
                MATHExample(
                    question_id=str(question_id) if question_id is not None else f"{path.stem}-{line_index}",
                    question=str(question),
                    gold_answer=extract_final_answer(str(answer)),
                    level=level,
                )
            )
    return examples


def _field_value(
    row: dict[str, Any],
    explicit_field: str | None,
    fallback_fields: tuple[str, ...],
    *,
    required: bool,
    label: str | None = None,
) -> Any:
    fields = (explicit_field,) if explicit_field else fallback_fields
    for field in fields:
        if field and field in row and row[field] is not None:
            return row[field]
    if required:
        display = label or "field"
        accepted = ", ".join(field for field in fields if field)
        raise ValueError(f"Missing {display} field; expected one of: {accepted}.")
    return None


def _normalize_level(level: Any) -> str | None:
    if level is None:
        return None
    text = str(level).strip()
    if not text:
        return None
    if text.lower().startswith("level "):
        suffix = text.split(None, 1)[1].strip()
        return f"Level {suffix}"
    if text.isdigit():
        return f"Level {text}"
    return text
