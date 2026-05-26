"""HuggingFace MATH-500 loading helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from csbf.answers import extract_final_answer


@dataclass(frozen=True)
class MATHExample:
    """One MATH-500 problem with extracted gold final answer."""

    question_id: str
    question: str
    gold_answer: str
    level: str | None = None


def load_math500_examples(
    limit: int | None = None,
    offset: int = 0,
    levels: list[str] | None = None,
    loader: Callable[..., Any] | None = None,
) -> list[MATHExample]:
    """Load MATH-500 examples from HuggingFace datasets.

    The dataset only has a ``test`` split.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    load_dataset = loader or _import_load_dataset()
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    rows = dataset["test"]
    allowed_levels = {_normalize_level(level) for level in levels or []}
    examples: list[MATHExample] = []
    matched = 0
    for index, row in enumerate(rows):
        level = _normalize_level(row.get("level")) if isinstance(row, dict) else None
        if allowed_levels and level not in allowed_levels:
            continue
        if matched < offset:
            matched += 1
            continue
        matched += 1
        if limit is not None and len(examples) >= limit:
            break
        examples.append(
            MATHExample(
                question_id=f"math500-{index}",
                question=str(row["problem"]),
                gold_answer=extract_final_answer(str(row["answer"])),
                level=level,
            )
        )
    return examples


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


def _import_load_dataset() -> Callable[..., Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The optional 'datasets' package is required for MATH-500 loading. "
            "Install with: python -m pip install '.[data]'"
        ) from error
    return load_dataset
