"""Likelihood calibration for prefix observations.

The binary state names are historical shorthand. In paper-facing claims,
ON_TRACK means a prefix is more likely to come from a trace that will
eventually answer correctly, not an oracle label for step-level correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


OFF_TRACK = 0
ON_TRACK = 1
UNKNOWN_CODE = "__UNKNOWN__"


def _check_equal_lengths(first: Sequence[object], second: Sequence[object]) -> None:
    if len(first) != len(second):
        raise ValueError("inputs must have equal lengths")


def _check_state(state: int) -> None:
    if state not in (OFF_TRACK, ON_TRACK):
        raise ValueError("state must be 0 for OFF_TRACK or 1 for ON_TRACK")


def _check_labels(labels: Sequence[int]) -> None:
    bad_labels = [label for label in labels if label not in (OFF_TRACK, ON_TRACK)]
    if bad_labels:
        raise ValueError("labels must be binary: 0 for OFF_TRACK, 1 for ON_TRACK")


@dataclass(frozen=True)
class HistogramLikelihood:
    """Histogram estimate of P(score_bin | reliability_state)."""

    bins: int
    counts_by_state: dict[int, list[float]]

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        labels: Sequence[int],
        bins: int = 10,
        smoothing: float = 1.0,
    ) -> "HistogramLikelihood":
        if bins <= 0:
            raise ValueError("bins must be positive")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        _check_equal_lengths(scores, labels)
        _check_labels(labels)

        counts = {
            OFF_TRACK: [float(smoothing)] * bins,
            ON_TRACK: [float(smoothing)] * bins,
        }
        for score, label in zip(scores, labels):
            counts[int(label)][_bin_index(float(score), bins)] += 1.0
        return cls(bins=bins, counts_by_state=counts)

    def probability(self, score: float, state: int) -> float:
        _check_state(state)
        counts = self.counts_by_state[state]
        return counts[_bin_index(float(score), self.bins)] / sum(counts)


@dataclass(frozen=True)
class ConceptLikelihood:
    """Categorical estimate of P(concept_code | reliability_state)."""

    counts_by_state: dict[int, dict[object, float]]
    vocabulary: frozenset[object]

    @classmethod
    def fit(
        cls,
        concept_codes: Sequence[object],
        labels: Sequence[int],
        smoothing: float = 1.0,
    ) -> "ConceptLikelihood":
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        _check_equal_lengths(concept_codes, labels)
        _check_labels(labels)

        vocabulary = frozenset(concept_codes)
        base_counts = {code: float(smoothing) for code in vocabulary}
        base_counts[UNKNOWN_CODE] = float(smoothing)
        counts = {
            OFF_TRACK: dict(base_counts),
            ON_TRACK: dict(base_counts),
        }
        for code, label in zip(concept_codes, labels):
            counts[int(label)][code] += 1.0
        return cls(counts_by_state=counts, vocabulary=vocabulary)

    def probability(self, concept_code: object, state: int) -> float:
        _check_state(state)
        if concept_code not in self.vocabulary:
            return 1.0
        code = concept_code
        counts = self.counts_by_state[state]
        return counts[code] / sum(counts.values())


@dataclass(frozen=True)
class JointLikelihood:
    """Joint histogram of P(score_bin, concept_code | state)."""

    score_bins: int
    joint_counts: dict[int, dict[tuple[int, object], float]]
    vocabulary: frozenset[object]

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        concept_codes: Sequence[object],
        labels: Sequence[int],
        score_bins: int = 5,
        smoothing: float = 1.0,
    ) -> "JointLikelihood":
        if score_bins <= 0:
            raise ValueError("score_bins must be positive")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        _check_equal_lengths(scores, concept_codes)
        _check_equal_lengths(scores, labels)
        _check_labels(labels)

        vocabulary = frozenset(concept_codes)
        joint_counts: dict[int, dict[tuple[int, object], float]] = {}
        for state in (OFF_TRACK, ON_TRACK):
            joint_counts[state] = {}
            for bin_idx in range(score_bins):
                for code in vocabulary:
                    joint_counts[state][(bin_idx, code)] = float(smoothing)
                joint_counts[state][(bin_idx, UNKNOWN_CODE)] = float(smoothing)

        for score, code, label in zip(scores, concept_codes, labels):
            bin_idx = _bin_index(float(score), score_bins)
            joint_counts[int(label)][(bin_idx, code)] += 1.0

        return cls(score_bins=score_bins, joint_counts=joint_counts, vocabulary=vocabulary)

    def probability(self, score: float, concept_code: object, state: int) -> float:
        _check_state(state)
        bin_idx = _bin_index(float(score), self.score_bins)
        code = concept_code if concept_code in self.vocabulary else UNKNOWN_CODE
        counts = self.joint_counts[state]
        key = (bin_idx, code)
        return counts.get(key, 0.0) / sum(counts.values()) if counts else 1.0


@dataclass(frozen=True)
class QuantileLikelihood:
    """Quantile-based histogram estimate of P(score_range | reliability_state)."""

    bin_edges: tuple[float, ...]
    counts_by_state: dict[int, list[float]]

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        labels: Sequence[int],
        bins: int = 10,
        smoothing: float = 1.0,
    ) -> "QuantileLikelihood":
        if bins <= 0:
            raise ValueError("bins must be positive")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        _check_equal_lengths(scores, labels)
        _check_labels(labels)

        sorted_scores = sorted(set(scores))
        n = len(sorted_scores)
        edges: list[float] = [0.0]
        if n > 0:
            for i in range(1, bins):
                idx = min(int(i * n / bins), n - 1)
                edges.append(sorted_scores[idx])
        edges.append(1.0)
        edges = sorted(set(edges))
        actual_bins = len(edges) - 1

        counts = {
            OFF_TRACK: [float(smoothing)] * actual_bins,
            ON_TRACK: [float(smoothing)] * actual_bins,
        }
        for score, label in zip(scores, labels):
            b = _quantile_bin_index(float(score), edges)
            counts[int(label)][b] += 1.0
        return cls(bin_edges=tuple(edges), counts_by_state=counts)

    def probability(self, score: float, state: int) -> float:
        _check_state(state)
        counts = self.counts_by_state[state]
        b = _quantile_bin_index(float(score), self.bin_edges)
        return counts[b] / sum(counts)


def _bin_index(score: float, bins: int) -> int:
    clipped = min(max(score, 0.0), 1.0)
    return min(int(clipped * bins), bins - 1)


def _quantile_bin_index(score: float, edges: Sequence[float]) -> int:
    clipped = min(max(score, 0.0), 1.0)
    for i in range(len(edges) - 1):
        if clipped < edges[i + 1]:
            return i
    return len(edges) - 2
