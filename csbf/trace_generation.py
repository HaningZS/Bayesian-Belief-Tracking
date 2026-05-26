"""Build trace records from GSM8K or MATH questions using a chat client."""

from __future__ import annotations

import sys
from typing import Any, Protocol, Union

from csbf.answers import is_correct_answer
from csbf.concepts import stable_concept_code
from csbf.gsm8k import GSM8KExample
from csbf.math_dataset import MATHExample
from csbf.schema import Observation, TraceRecord
from csbf.segmentation import fixed_token_chunks, rule_based_steps

DatasetExample = Union[GSM8KExample, MATHExample]

_MESSAGE_BUILDERS: dict[str, Any] = {
    "gsm8k": "_messages_for_gsm8k",
    "math": "_messages_for_math",
}

_SOURCE_LABELS: dict[str, str] = {
    "gsm8k": "deepseek_gsm8k",
    "math": "deepseek_math",
}


class JsonChatClient(Protocol):
    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        ...


def build_trace_records(
    examples: list[DatasetExample],
    client: JsonChatClient,
    dataset_type: str = "gsm8k",
    traces_per_question: int = 4,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.6,
    max_tokens: int = 1024,
    progress: bool = False,
) -> list[TraceRecord]:
    """Generate trace records for GSM8K or MATH examples using a JSON chat client."""

    if traces_per_question <= 0:
        raise ValueError("traces_per_question must be positive")
    if dataset_type not in _MESSAGE_BUILDERS:
        raise ValueError(f"unknown dataset_type: {dataset_type!r}")

    message_fn = globals()[_MESSAGE_BUILDERS[dataset_type]]
    source = _SOURCE_LABELS[dataset_type]

    records: list[TraceRecord] = []
    for q_idx, example in enumerate(examples):
        if progress:
            print(
                f"[{q_idx + 1}/{len(examples)}] {example.question_id}",
                file=sys.stderr,
            )
        for trace_index in range(traces_per_question):
            response = client.chat_json(
                messages=message_fn(example.question),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            records.append(_record_from_response(example, trace_index, response, source=source))
    return records


def build_gsm8k_trace_records(
    examples: list[GSM8KExample],
    client: JsonChatClient,
    traces_per_question: int = 4,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.6,
    max_tokens: int = 1024,
) -> list[TraceRecord]:
    """Generate trace records for GSM8K examples.  Backward-compatible wrapper."""

    return build_trace_records(
        examples,
        client=client,
        dataset_type="gsm8k",
        traces_per_question=traces_per_question,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _messages_for_gsm8k(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You solve grade-school math problems. Return only valid JSON with keys "
                "'steps' and 'final_answer'. 'steps' must be a list of objects with "
                "'text' and 'score'. The score is your prefix-local confidence that the "
                "solution remains on track, between 0 and 1. Do not include markdown."
            ),
        },
        {
            "role": "user",
            "content": f"Solve this GSM8K problem step by step:\n{question}",
        },
    ]


def _messages_for_math(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You solve math competition problems. Return only valid JSON with keys "
                "'steps' and 'final_answer'. 'steps' must be a list of objects with "
                "'text' and 'score'. Put your final answer in \\boxed{} format. "
                "The score is your prefix-local confidence between 0 and 1."
            ),
        },
        {"role": "user", "content": f"Solve this problem step by step:\n{question}"},
    ]


def _record_from_response(
    example: DatasetExample, trace_index: int, response: dict[str, Any], source: str = "deepseek_gsm8k",
) -> TraceRecord:
    final_answer = str(response.get("final_answer", ""))
    steps = response.get("steps", [])
    if not isinstance(steps, list) or not steps:
        trace_text = str(response.get("trace_text", ""))
        steps = [{"text": text, "score": None} for text in _fallback_steps(trace_text)]

    observations: list[Observation] = []
    trace_parts: list[str] = []
    for step_index, raw_step in enumerate(steps):
        step = raw_step if isinstance(raw_step, dict) else {"text": str(raw_step)}
        text = str(step.get("text", "")).strip()
        if not text:
            continue
        score = _coerce_score(step.get("score"))
        trace_parts.append(text)
        observations.append(
            Observation(
                step_index=step_index,
                text=text,
                score=score,
                concept_code=stable_concept_code(text),
            )
        )

    trace_text = "\n".join(trace_parts)
    if not observations:
        fallback_text = final_answer or "No reasoning returned."
        observations.append(
            Observation(
                step_index=0,
                text=fallback_text,
                score=0.5,
                concept_code=stable_concept_code(fallback_text),
            )
        )
        trace_text = fallback_text

    return TraceRecord(
        question_id=example.question_id,
        question=example.question,
        trace_id=f"{example.question_id}-t{trace_index}",
        trace_text=trace_text,
        final_answer=final_answer,
        gold_answer=example.gold_answer,
        correct=is_correct_answer(final_answer, example.gold_answer),
        observations=observations,
        metadata={"source": source},
    )


def _fallback_steps(trace_text: str) -> list[str]:
    steps = rule_based_steps(trace_text)
    if len(steps) <= 1:
        chunks = fixed_token_chunks(trace_text, chunk_size=64)
        return chunks or [trace_text]
    return steps


def _coerce_score(value: Any) -> float:
    if value is None:
        return 0.5
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(max(score, 0.0), 1.0)
