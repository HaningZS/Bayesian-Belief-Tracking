"""Online score aggregation baselines."""

from __future__ import annotations

from collections.abc import Sequence


def moving_average(scores: Sequence[float], window: int) -> list[float]:
    """Compute a prefix-only moving average over reliability scores."""

    if window <= 0:
        raise ValueError("window must be positive")

    averaged: list[float] = []
    for end in range(1, len(scores) + 1):
        start = max(0, end - window)
        averaged.append(round(sum(scores[start:end]) / (end - start), 12))
    return averaged


def temporal_score_metric(scores: Sequence[float]) -> float:
    """Simplified Tracing-style temporal metric from score trajectory.

    Combines the final score with the mean score delta (trajectory trend).
    A declining trajectory produces a lower prediction than a stable one.
    """
    if not scores:
        return 0.5
    if len(scores) == 1:
        return float(scores[0])
    deltas = [float(scores[i]) - float(scores[i - 1]) for i in range(1, len(scores))]
    mean_delta = sum(deltas) / len(deltas)
    final = float(scores[-1])
    combined = 0.5 * final + 0.5 * max(0.0, min(1.0, 0.5 + mean_delta))
    return round(min(max(combined, 0.0), 1.0), 12)


def exponential_moving_average(scores: Sequence[float], alpha: float) -> list[float]:
    """Compute an online exponential moving average."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not scores:
        return []

    averaged = [round(float(scores[0]), 12)]
    for score in scores[1:]:
        next_value = alpha * float(score) + (1.0 - alpha) * averaged[-1]
        averaged.append(round(next_value, 12))
    return averaged
