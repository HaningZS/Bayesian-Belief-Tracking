"""Answer-label audit sampling for generated trace records."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from csbf.answers import is_correct_answer, symbolic_answer_equal
from csbf.schema import TraceRecord


DEFAULT_CATEGORIES = (
    "possible_equivalent_answer",
    "incorrect_with_boxed_answer",
    "missing_final_answer",
    "token_cap_hit",
    "correct_without_boxed_marker",
)


@dataclass(frozen=True)
class AnswerAuditRow:
    """One trace selected for manual answer-label audit."""

    audit_id: str
    question_id: str
    trace_id: str
    categories: list[str]
    correct: bool
    final_answer: str
    gold_answer: str
    observation_count: int
    question_excerpt: str
    trace_excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "question_id": self.question_id,
            "trace_id": self.trace_id,
            "categories": list(self.categories),
            "correct": self.correct,
            "final_answer": self.final_answer,
            "gold_answer": self.gold_answer,
            "observation_count": self.observation_count,
            "question_excerpt": self.question_excerpt,
            "trace_excerpt": self.trace_excerpt,
        }


def build_answer_audit_rows(
    records: list[TraceRecord],
    max_per_category: int = 25,
    seed: int = 0,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    excerpt_chars: int = 500,
) -> list[AnswerAuditRow]:
    """Select records likely to expose answer-label or extraction problems."""

    if max_per_category < 1:
        raise ValueError("max_per_category must be at least 1")
    selected: dict[str, TraceRecord] = {}
    rng = random.Random(seed)
    for category in categories:
        matches = [record for record in records if category in classify_answer_audit_record(record)]
        rng.shuffle(matches)
        for record in matches[:max_per_category]:
            selected.setdefault(_audit_id(record), record)

    return [
        _build_row(record, excerpt_chars=excerpt_chars)
        for record in sorted(selected.values(), key=lambda item: (_category_sort_key(item), item.question_id, item.trace_id))
    ]


def classify_answer_audit_record(record: TraceRecord) -> list[str]:
    """Return audit-risk categories for one trace record."""

    categories: list[str] = []
    has_final = bool(record.final_answer.strip())
    has_boxed = "\\boxed" in record.trace_text
    if not record.correct and has_final and has_boxed:
        categories.append("incorrect_with_boxed_answer")
    if not has_final:
        categories.append("missing_final_answer")
    if _token_cap_hit(record):
        categories.append("token_cap_hit")
    if record.correct and not has_boxed:
        categories.append("correct_without_boxed_marker")
    if not record.correct and has_final and record.gold_answer.strip():
        try:
            if symbolic_answer_equal(record.final_answer, record.gold_answer) or is_correct_answer(
                record.final_answer,
                record.gold_answer,
                checker="symbolic",
            ):
                categories.append("possible_equivalent_answer")
        except ValueError:
            pass
    return [category for category in DEFAULT_CATEGORIES if category in categories]


def summarize_answer_audit_rows(rows: list[AnswerAuditRow], input_count: int) -> dict[str, Any]:
    """Summarize selected audit rows."""

    category_counts = {category: 0 for category in DEFAULT_CATEGORIES}
    for row in rows:
        for category in row.categories:
            category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "input_records": input_count,
        "audit_rows": len(rows),
        "category_counts": category_counts,
    }


def _build_row(record: TraceRecord, excerpt_chars: int) -> AnswerAuditRow:
    return AnswerAuditRow(
        audit_id=_audit_id(record),
        question_id=record.question_id,
        trace_id=record.trace_id,
        categories=classify_answer_audit_record(record),
        correct=record.correct,
        final_answer=record.final_answer,
        gold_answer=record.gold_answer,
        observation_count=len(record.observations),
        question_excerpt=_excerpt(record.question, excerpt_chars),
        trace_excerpt=_excerpt(record.trace_text, excerpt_chars),
    )


def _audit_id(record: TraceRecord) -> str:
    return f"{record.question_id}:{record.trace_id}"


def _category_sort_key(record: TraceRecord) -> int:
    categories = classify_answer_audit_record(record)
    if not categories:
        return len(DEFAULT_CATEGORIES)
    return min(DEFAULT_CATEGORIES.index(category) for category in categories if category in DEFAULT_CATEGORIES)


def _token_cap_hit(record: TraceRecord) -> bool:
    metadata = record.metadata
    return bool(
        metadata.get("hit_token_cap")
        or metadata.get("token_cap_hit")
        or metadata.get("max_new_tokens_hit")
        or metadata.get("finish_reason") == "length"
    )


def _excerpt(text: str, max_chars: int) -> str:
    if max_chars < 1:
        return ""
    value = " ".join(text.split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."
