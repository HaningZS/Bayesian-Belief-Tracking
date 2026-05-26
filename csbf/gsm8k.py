"""HuggingFace GSM8K loading helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from csbf.answers import extract_final_answer


@dataclass(frozen=True)
class GSM8KExample:
    """One GSM8K question with extracted gold final answer."""

    question_id: str
    question: str
    gold_answer: str


def load_gsm8k_examples(
    split: str = "test",
    limit: int | None = None,
    offset: int = 0,
    loader: Callable[[str, str], Any] | None = None,
) -> list[GSM8KExample]:
    """Load GSM8K examples from HuggingFace datasets."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    load_dataset = loader or _import_load_dataset()
    dataset = load_dataset("openai/gsm8k", "main")
    rows = dataset[split]
    examples: list[GSM8KExample] = []
    for index, row in enumerate(rows):
        if index < offset:
            continue
        if limit is not None and len(examples) >= limit:
            break
        examples.append(
            GSM8KExample(
                question_id=f"gsm8k-{split}-{index}",
                question=str(row["question"]),
                gold_answer=extract_final_answer(str(row["answer"])),
            )
        )
    return examples


def _import_load_dataset() -> Callable[[str, str], Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The optional 'datasets' package is required for GSM8K loading. "
            "Install with: python -m pip install '.[data]'"
        ) from error
    return load_dataset
