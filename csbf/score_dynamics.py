"""Prefix-only score trajectory concept rewrites."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from csbf.schema import Observation, TraceRecord, load_jsonl, save_jsonl


def build_score_dynamics_records(
    records_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    *,
    ema_alpha: float = 0.4,
    volatility_window: int = 3,
    delta_threshold: float = 0.05,
    residual_threshold: float = 0.05,
    volatility_threshold: float = 0.08,
    recovery_threshold: float = 0.15,
) -> dict[str, Any]:
    """Rewrite concept codes from prefix-only score trajectory dynamics."""

    config = ScoreDynamicsConfig(
        ema_alpha=ema_alpha,
        volatility_window=volatility_window,
        delta_threshold=delta_threshold,
        residual_threshold=residual_threshold,
        volatility_threshold=volatility_threshold,
        recovery_threshold=recovery_threshold,
    )
    records = load_jsonl(records_path)
    rewritten: list[TraceRecord] = []
    usage: Counter[str] = Counter()
    missing_score_count = 0
    for record in records:
        new_record = _rewrite_record(record, config)
        rewritten.append(new_record)
        for observation in new_record.observations:
            code = observation.concept_code
            if code == "score_dyn|missing_score":
                missing_score_count += 1
            usage[str(code)] += 1
    save_jsonl(rewritten, output_path)
    summary: dict[str, Any] = {
        "records": len(records),
        "questions": len({record.question_id for record in records}),
        "concept_source": "score_dynamics",
        "concept_usage": dict(sorted(usage.items())),
        "missing_score_observations": missing_score_count,
        "config": config.to_dict(),
        "outputs": {"records": str(output_path)},
    }
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        summary["outputs"]["report"] = str(report)
        report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def dynamic_score_code(
    scores: Sequence[float],
    *,
    ema_alpha: float = 0.4,
    volatility_window: int = 3,
    delta_threshold: float = 0.05,
    residual_threshold: float = 0.05,
    volatility_threshold: float = 0.08,
    recovery_threshold: float = 0.15,
) -> tuple[list[str], list[float]]:
    """Return dynamic concept codes and score deltas for a score sequence."""

    config = ScoreDynamicsConfig(
        ema_alpha=ema_alpha,
        volatility_window=volatility_window,
        delta_threshold=delta_threshold,
        residual_threshold=residual_threshold,
        volatility_threshold=volatility_threshold,
        recovery_threshold=recovery_threshold,
    )
    values = [float(score) for score in scores]
    if not values:
        return [], []
    codes: list[str] = []
    deltas: list[float] = []
    ema = values[0]
    delta_history: list[float] = []
    running_min = values[0]
    running_max = values[0]
    for index, score in enumerate(values):
        if index == 0:
            delta = 0.0
            residual = 0.0
            volatility = 0.0
        else:
            previous = values[index - 1]
            delta = _round(score - previous)
            residual = _round(score - ema)
            delta_history.append(delta)
            recent = delta_history[-config.volatility_window :]
            volatility = sum(abs(value) for value in recent) / len(recent)

        trend = _trend_bucket(delta, config.delta_threshold)
        volatility_bucket = "high_vol" if volatility >= config.volatility_threshold else "low_vol"
        residual_bucket = _residual_bucket(residual, config.residual_threshold)
        phase = _phase_bucket(
            score=score,
            delta=delta,
            running_min=running_min,
            running_max=running_max,
            delta_threshold=config.delta_threshold,
            recovery_threshold=config.recovery_threshold,
        )
        codes.append(f"score_dyn|{trend}|{volatility_bucket}|{residual_bucket}|{phase}")
        deltas.append(_round(delta))

        if index > 0:
            ema = config.ema_alpha * score + (1.0 - config.ema_alpha) * ema
            running_min = min(running_min, score)
            running_max = max(running_max, score)
    return codes, deltas


class ScoreDynamicsConfig:
    """Small value object for score-dynamics thresholds."""

    def __init__(
        self,
        *,
        ema_alpha: float = 0.4,
        volatility_window: int = 3,
        delta_threshold: float = 0.05,
        residual_threshold: float = 0.05,
        volatility_threshold: float = 0.08,
        recovery_threshold: float = 0.15,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if volatility_window <= 0:
            raise ValueError("volatility_window must be positive")
        for name, value in (
            ("delta_threshold", delta_threshold),
            ("residual_threshold", residual_threshold),
            ("volatility_threshold", volatility_threshold),
            ("recovery_threshold", recovery_threshold),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.ema_alpha = float(ema_alpha)
        self.volatility_window = int(volatility_window)
        self.delta_threshold = float(delta_threshold)
        self.residual_threshold = float(residual_threshold)
        self.volatility_threshold = float(volatility_threshold)
        self.recovery_threshold = float(recovery_threshold)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ema_alpha": self.ema_alpha,
            "volatility_window": self.volatility_window,
            "delta_threshold": self.delta_threshold,
            "residual_threshold": self.residual_threshold,
            "volatility_threshold": self.volatility_threshold,
            "recovery_threshold": self.recovery_threshold,
        }


def _rewrite_record(record: TraceRecord, config: ScoreDynamicsConfig) -> TraceRecord:
    scored_observations = [observation for observation in record.observations if observation.score is not None]
    codes, deltas = dynamic_score_code(
        [float(observation.score) for observation in scored_observations],
        ema_alpha=config.ema_alpha,
        volatility_window=config.volatility_window,
        delta_threshold=config.delta_threshold,
        residual_threshold=config.residual_threshold,
        volatility_threshold=config.volatility_threshold,
        recovery_threshold=config.recovery_threshold,
    )
    code_by_step = {observation.step_index: code for observation, code in zip(scored_observations, codes, strict=True)}
    delta_by_step = {
        observation.step_index: delta for observation, delta in zip(scored_observations, deltas, strict=True)
    }
    observations: list[Observation] = []
    for observation in record.observations:
        if observation.score is None:
            concept_code = "score_dyn|missing_score"
            score_delta = observation.score_delta
        else:
            concept_code = code_by_step[observation.step_index]
            score_delta = delta_by_step[observation.step_index]
        observations.append(
            Observation(
                step_index=observation.step_index,
                text=observation.text,
                score=observation.score,
                concept_code=concept_code,
                entropy=observation.entropy,
                score_delta=score_delta,
            )
        )
    metadata = dict(record.metadata)
    metadata["concept_source"] = "score_dynamics"
    metadata["score_dynamics"] = config.to_dict()
    return TraceRecord(
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


def _trend_bucket(delta: float, threshold: float) -> str:
    if delta >= threshold:
        return "up"
    if delta <= -threshold:
        return "down"
    return "flat"


def _residual_bucket(residual: float, threshold: float) -> str:
    if residual >= threshold:
        return "above_ema"
    if residual <= -threshold:
        return "below_ema"
    return "near_ema"


def _phase_bucket(
    *,
    score: float,
    delta: float,
    running_min: float,
    running_max: float,
    delta_threshold: float,
    recovery_threshold: float,
) -> str:
    if delta >= delta_threshold and score - running_min >= recovery_threshold:
        return "recovery"
    if delta <= -delta_threshold and running_max - score >= recovery_threshold:
        return "drawdown"
    return "stable"


def _round(value: float) -> float:
    return round(float(value), 12)
