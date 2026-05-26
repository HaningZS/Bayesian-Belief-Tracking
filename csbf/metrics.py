"""Small binary-classification metrics for offline smoke tests."""

from __future__ import annotations

from collections.abc import Sequence


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    _check_inputs(labels, probabilities)
    value = sum((float(probability) - int(label)) ** 2 for label, probability in zip(labels, probabilities))
    return round(value / len(labels), 12)


def expected_calibration_error(labels: Sequence[int], probabilities: Sequence[float], bins: int = 10) -> float:
    _check_inputs(labels, probabilities)
    if bins <= 0:
        raise ValueError("bins must be positive")

    total = len(labels)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        in_bin = [
            (int(label), float(probability))
            for label, probability in zip(labels, probabilities)
            if lower <= float(probability) < upper or (bin_index == bins - 1 and float(probability) == 1.0)
        ]
        if not in_bin:
            continue
        observed = sum(label for label, _ in in_bin) / len(in_bin)
        confidence = sum(probability for _, probability in in_bin) / len(in_bin)
        ece += (len(in_bin) / total) * abs(observed - confidence)
    return round(ece, 12)


def roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    _check_inputs(labels, scores)
    pairs = [(float(scores[i]), int(labels[i])) for i in range(len(labels))]
    n_pos = sum(1 for _, l in pairs if l == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc_score requires at least one positive and one negative label")

    pairs.sort(key=lambda x: x[0])
    i = 0
    rank_sum = 0.0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        midrank = (i + j + 1) / 2
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += midrank
        i = j

    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return round(auc, 12)


def average_precision_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute average precision for binary labels."""

    _check_inputs(labels, scores)
    positives = sum(1 for label in labels if int(label) == 1)
    if positives == 0:
        raise ValueError("average_precision_score requires at least one positive label")

    ranked = sorted(
        ((float(score), int(label)) for label, score in zip(labels, scores)),
        key=lambda item: item[0],
        reverse=True,
    )
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return round(precision_sum / positives, 12)


def _check_inputs(labels: Sequence[int], probabilities: Sequence[float]) -> None:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have equal lengths")
    if not labels:
        raise ValueError("at least one example is required")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary: 0 or 1")
    if any(not 0.0 <= float(probability) <= 1.0 for probability in probabilities):
        raise ValueError("probabilities must be in [0, 1]")
