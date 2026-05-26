"""Baum-Welch EM for HMM emission calibration."""

from __future__ import annotations

from csbf.calibration import HistogramLikelihood, ON_TRACK, OFF_TRACK
from csbf.filtering import HMMConfig
from csbf.schema import TraceRecord


def em_fit_score_likelihood(
    records: list[TraceRecord],
    config: HMMConfig,
    bins: int = 10,
    smoothing: float = 0.5,
    max_iterations: int = 20,
    tolerance: float = 1e-4,
) -> HistogramLikelihood:
    """Estimate score emission likelihoods using Baum-Welch EM.

    Instead of propagating the final-answer label to every step, this
    treats per-step reliability as latent and iterates:

      E-step  – forward-backward gives per-step state posteriors gamma.
      M-step  – re-estimate histogram bin counts weighted by gamma.

    Initialization uses the naive all-step label propagation (warm start).
    """
    trace_scores: list[list[float]] = []
    init_scores: list[float] = []
    init_labels: list[int] = []

    for record in records:
        scores = [
            float(obs.score)
            for obs in record.observations
            if obs.score is not None
        ]
        if not scores:
            continue
        trace_scores.append(scores)
        label = 1 if record.correct else 0
        for s in scores:
            init_scores.append(s)
            init_labels.append(label)

    if not trace_scores:
        raise ValueError("no score observations found in records")

    likelihood = HistogramLikelihood.fit(
        scores=init_scores, labels=init_labels, bins=bins, smoothing=smoothing,
    )

    for _ in range(max_iterations):
        old_probs = _bin_probabilities(likelihood, bins)

        all_scores: list[float] = []
        all_gammas: list[float] = []
        for scores in trace_scores:
            gammas = _forward_backward(scores, likelihood, config)
            all_scores.extend(scores)
            all_gammas.extend(gammas)

        counts: dict[int, list[float]] = {
            OFF_TRACK: [smoothing] * bins,
            ON_TRACK: [smoothing] * bins,
        }
        for score, gamma_on in zip(all_scores, all_gammas):
            idx = _bin_index(score, bins)
            counts[ON_TRACK][idx] += gamma_on
            counts[OFF_TRACK][idx] += 1.0 - gamma_on

        new_likelihood = HistogramLikelihood(bins=bins, counts_by_state=counts)
        new_probs = _bin_probabilities(new_likelihood, bins)

        max_change = max(
            abs(new_probs[state][i] - old_probs[state][i])
            for state in (ON_TRACK, OFF_TRACK)
            for i in range(bins)
        )
        likelihood = new_likelihood
        if max_change < tolerance:
            break

    return likelihood


# ------------------------------------------------------------------
# Standalone forward-backward (no dependency on filter.smooth())
# ------------------------------------------------------------------

def _forward_backward(
    scores: list[float],
    likelihood: HistogramLikelihood,
    config: HMMConfig,
) -> list[float]:
    """Return gamma(t) = P(ON_TRACK at t | full observation sequence)."""
    length = len(scores)
    if length == 0:
        return []

    alpha, scaling = _forward(scores, likelihood, config)
    beta = _backward(scores, likelihood, config, scaling)

    gammas: list[float] = []
    for t in range(length):
        a_on, a_off = alpha[t]
        b_on, b_off = beta[t]
        g_on = a_on * b_on
        g_off = a_off * b_off
        total = g_on + g_off
        gammas.append(g_on / total if total > 0 else 0.5)
    return gammas


def _forward(
    scores: list[float],
    likelihood: HistogramLikelihood,
    config: HMMConfig,
) -> tuple[list[tuple[float, float]], list[float]]:
    """Scaled forward pass.  Returns (alpha_hat, scaling_factors)."""
    alpha: list[tuple[float, float]] = []
    scaling: list[float] = []

    prior_on = config.initial_on_track
    lik_on = likelihood.probability(scores[0], ON_TRACK)
    lik_off = likelihood.probability(scores[0], OFF_TRACK)
    a_on = prior_on * lik_on
    a_off = (1.0 - prior_on) * lik_off
    c = a_on + a_off or 1.0
    alpha.append((a_on / c, a_off / c))
    scaling.append(c)

    for t in range(1, len(scores)):
        prev_on, prev_off = alpha[t - 1]
        pred_on = prev_on * (1.0 - config.p_error) + prev_off * config.p_recover
        pred_off = prev_on * config.p_error + prev_off * (1.0 - config.p_recover)

        lik_on = likelihood.probability(scores[t], ON_TRACK)
        lik_off = likelihood.probability(scores[t], OFF_TRACK)
        a_on = pred_on * lik_on
        a_off = pred_off * lik_off
        c = a_on + a_off or 1.0
        alpha.append((a_on / c, a_off / c))
        scaling.append(c)

    return alpha, scaling


def _backward(
    scores: list[float],
    likelihood: HistogramLikelihood,
    config: HMMConfig,
    scaling: list[float],
) -> list[tuple[float, float]]:
    """Scaled backward pass."""
    length = len(scores)
    beta: list[tuple[float, float]] = [(0.0, 0.0)] * length
    beta[length - 1] = (1.0, 1.0)

    for t in range(length - 2, -1, -1):
        b_on_next, b_off_next = beta[t + 1]
        lik_on_next = likelihood.probability(scores[t + 1], ON_TRACK)
        lik_off_next = likelihood.probability(scores[t + 1], OFF_TRACK)

        b_on = (
            (1.0 - config.p_error) * lik_on_next * b_on_next
            + config.p_error * lik_off_next * b_off_next
        )
        b_off = (
            config.p_recover * lik_on_next * b_on_next
            + (1.0 - config.p_recover) * lik_off_next * b_off_next
        )

        c = scaling[t + 1]
        if c > 0:
            b_on /= c
            b_off /= c

        beta[t] = (b_on, b_off)

    return beta


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _bin_index(score: float, bins: int) -> int:
    clipped = min(max(score, 0.0), 1.0)
    return min(int(clipped * bins), bins - 1)


def _bin_probabilities(
    likelihood: HistogramLikelihood, bins: int,
) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    for state in (ON_TRACK, OFF_TRACK):
        counts = likelihood.counts_by_state[state]
        total = sum(counts)
        result[state] = (
            [c / total for c in counts] if total > 0 else [1.0 / bins] * bins
        )
    return result
