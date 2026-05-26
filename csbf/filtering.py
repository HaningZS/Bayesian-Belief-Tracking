"""Two-state HMM-style Bayesian reliability filtering.

The online posterior is best interpreted as a prefix-conditioned belief in
eventual trace success. Viterbi and smoothing remain offline diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from csbf.calibration import (
    OFF_TRACK,
    ON_TRACK,
    ConceptLikelihood,
    HistogramLikelihood,
    JointLikelihood,
)

_LOG_FLOOR = 1e-300


@dataclass(frozen=True)
class HMMConfig:
    """Transition and initial-belief parameters for binary reliability states."""

    p_error: float = 0.05
    p_recover: float = 0.10
    initial_on_track: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (
            ("p_error", self.p_error),
            ("p_recover", self.p_recover),
            ("initial_on_track", self.initial_on_track),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class NonStationaryHMMConfig:
    """Position-dependent transition parameters for ON/OFF reliability states."""

    p_error_base: float = 0.03
    p_error_slope: float = 0.005
    p_recover_base: float = 0.15
    p_recover_decay: float = 0.02
    initial_on_track: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (
            ("p_error_base", self.p_error_base),
            ("p_recover_base", self.p_recover_base),
            ("initial_on_track", self.initial_on_track),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.p_error_slope < 0:
            raise ValueError("p_error_slope must be non-negative")
        if self.p_recover_decay < 0:
            raise ValueError("p_recover_decay must be non-negative")


@dataclass(frozen=True)
class BayesianReliabilityFilter:
    """Recursive belief tracker for prefix-only score and concept observations."""

    config: HMMConfig | NonStationaryHMMConfig
    score_likelihood: HistogramLikelihood | None = None
    concept_likelihood: ConceptLikelihood | None = None
    joint_likelihood: JointLikelihood | None = None

    # --- helpers ---

    def _get_transitions(self, step_index: int) -> tuple[float, float]:
        """Return (p_error, p_recover) for the given step."""
        if isinstance(self.config, NonStationaryHMMConfig):
            p_error = min(
                self.config.p_error_base + self.config.p_error_slope * step_index,
                0.5,
            )
            p_recover = max(
                self.config.p_recover_base - self.config.p_recover_decay * step_index,
                0.01,
            )
            return p_error, p_recover
        return self.config.p_error, self.config.p_recover

    def _observation_likelihoods(
        self, score: float | None, concept_code: object | None,
    ) -> tuple[float, float]:
        """Return (likelihood_on, likelihood_off) for one observation pair."""
        if self.joint_likelihood is not None:
            if score is None or concept_code is None:
                return 1.0, 1.0
            return (
                self.joint_likelihood.probability(float(score), concept_code, ON_TRACK),
                self.joint_likelihood.probability(float(score), concept_code, OFF_TRACK),
            )

        likelihood_on = 1.0
        likelihood_off = 1.0
        if score is not None and self.score_likelihood is not None:
            likelihood_on *= self.score_likelihood.probability(score, ON_TRACK)
            likelihood_off *= self.score_likelihood.probability(score, OFF_TRACK)
        if concept_code is not None and self.concept_likelihood is not None:
            likelihood_on *= self.concept_likelihood.probability(concept_code, ON_TRACK)
            likelihood_off *= self.concept_likelihood.probability(concept_code, OFF_TRACK)
        return likelihood_on, likelihood_off

    def _parse_observations(
        self,
        scores: Sequence[float] | None,
        concept_codes: Sequence[object] | None,
    ) -> int:
        """Validate observation lengths and return *n_steps*."""
        if scores is None and concept_codes is None:
            return 0
        n_steps = len(scores) if scores is not None else len(concept_codes or [])
        if scores is not None and len(scores) != n_steps:
            raise ValueError("scores length mismatch")
        if concept_codes is not None and len(concept_codes) != n_steps:
            raise ValueError("concept_codes length mismatch")
        return n_steps

    # --- public interface ---

    def run(
        self,
        scores: Sequence[float] | None = None,
        concept_codes: Sequence[object] | None = None,
    ) -> list[float]:
        """Forward filter: online belief in eventual success at each prefix."""
        n_steps = self._parse_observations(scores, concept_codes)
        if n_steps == 0:
            return []

        belief = self.config.initial_on_track
        beliefs: list[float] = []
        for index in range(n_steps):
            score = None if scores is None else scores[index]
            concept_code = None if concept_codes is None else concept_codes[index]
            belief = self.step(
                belief, score=score, concept_code=concept_code, step_index=index,
            )
            beliefs.append(round(belief, 12))
        return beliefs

    def step(
        self,
        previous_on_track: float,
        score: float | None = None,
        concept_code: object | None = None,
        step_index: int = 0,
    ) -> float:
        """Single predict-then-update cycle."""
        if not 0.0 <= previous_on_track <= 1.0:
            raise ValueError("previous_on_track must be in [0, 1]")

        p_error, p_recover = self._get_transitions(step_index)

        predicted_on = (
            previous_on_track * (1.0 - p_error)
            + (1.0 - previous_on_track) * p_recover
        )
        predicted_off = 1.0 - predicted_on

        likelihood_on, likelihood_off = self._observation_likelihoods(
            score, concept_code,
        )

        on_mass = likelihood_on * predicted_on
        off_mass = likelihood_off * predicted_off
        normalizer = on_mass + off_mass
        if normalizer == 0.0:
            return predicted_on
        return on_mass / normalizer

    def decode(
        self,
        scores: Sequence[float] | None = None,
        concept_codes: Sequence[object] | None = None,
    ) -> list[int]:
        """Viterbi: return most likely state sequence (0=OFF_TRACK, 1=ON_TRACK)."""
        n_steps = self._parse_observations(scores, concept_codes)
        if n_steps == 0:
            return []

        prev = [
            math.log(max(1.0 - self.config.initial_on_track, _LOG_FLOOR)),
            math.log(max(self.config.initial_on_track, _LOG_FLOOR)),
        ]
        backpointers: list[tuple[int, int]] = []

        for t in range(n_steps):
            score = None if scores is None else scores[t]
            concept_code = None if concept_codes is None else concept_codes[t]
            emit_on, emit_off = self._observation_likelihoods(score, concept_code)
            ll_on = math.log(max(emit_on, _LOG_FLOOR))
            ll_off = math.log(max(emit_off, _LOG_FLOOR))
            p_error, p_recover = self._get_transitions(t)

            log_stay_on = math.log(max(1.0 - p_error, _LOG_FLOOR))
            log_error = math.log(max(p_error, _LOG_FLOOR))
            log_recover = math.log(max(p_recover, _LOG_FLOOR))
            log_stay_off = math.log(max(1.0 - p_recover, _LOG_FLOOR))

            from_on_to_on = prev[ON_TRACK] + log_stay_on
            from_off_to_on = prev[OFF_TRACK] + log_recover
            if from_on_to_on >= from_off_to_on:
                best_to_on = from_on_to_on + ll_on
                bp_on = ON_TRACK
            else:
                best_to_on = from_off_to_on + ll_on
                bp_on = OFF_TRACK

            from_on_to_off = prev[ON_TRACK] + log_error
            from_off_to_off = prev[OFF_TRACK] + log_stay_off
            if from_on_to_off >= from_off_to_off:
                best_to_off = from_on_to_off + ll_off
                bp_off = ON_TRACK
            else:
                best_to_off = from_off_to_off + ll_off
                bp_off = OFF_TRACK

            backpointers.append((bp_off, bp_on))
            prev = [best_to_off, best_to_on]

        path = [ON_TRACK if prev[ON_TRACK] >= prev[OFF_TRACK] else OFF_TRACK]
        for t in range(n_steps - 1, 0, -1):
            path.append(backpointers[t][path[-1]])
        path.reverse()
        return path

    def smooth(
        self,
        scores: Sequence[float] | None = None,
        concept_codes: Sequence[object] | None = None,
    ) -> list[float]:
        """Forward-backward: smoothed P(ON_TRACK) at each step (offline only)."""
        n_steps = self._parse_observations(scores, concept_codes)
        if n_steps == 0:
            return []

        # --- forward pass (matches *run* but stores both state masses) ---
        alphas: list[tuple[float, float]] = []
        alpha_on: float = self.config.initial_on_track
        alpha_off: float = 1.0 - alpha_on

        for t in range(n_steps):
            score = None if scores is None else scores[t]
            concept_code = None if concept_codes is None else concept_codes[t]
            emit_on, emit_off = self._observation_likelihoods(score, concept_code)
            p_error, p_recover = self._get_transitions(t)

            pred_on = alpha_on * (1.0 - p_error) + alpha_off * p_recover
            pred_off = alpha_on * p_error + alpha_off * (1.0 - p_recover)

            alpha_on = pred_on * emit_on
            alpha_off = pred_off * emit_off

            norm = alpha_on + alpha_off
            if norm > 0:
                alpha_on /= norm
                alpha_off /= norm

            alphas.append((alpha_off, alpha_on))

        # --- backward pass ---
        betas: list[tuple[float, float]] = [(1.0, 1.0)] * n_steps
        beta_on: float = 1.0
        beta_off: float = 1.0

        for t in range(n_steps - 2, -1, -1):
            score_next = None if scores is None else scores[t + 1]
            concept_next = None if concept_codes is None else concept_codes[t + 1]
            emit_on, emit_off = self._observation_likelihoods(score_next, concept_next)
            p_error, p_recover = self._get_transitions(t + 1)

            new_beta_on = (
                (1.0 - p_error) * emit_on * beta_on
                + p_error * emit_off * beta_off
            )
            new_beta_off = (
                p_recover * emit_on * beta_on
                + (1.0 - p_recover) * emit_off * beta_off
            )

            norm = new_beta_on + new_beta_off
            if norm > 0:
                beta_on = new_beta_on / norm
                beta_off = new_beta_off / norm
            else:
                beta_on = 0.5
                beta_off = 0.5
            betas[t] = (beta_off, beta_on)

        # --- combine ---
        posteriors: list[float] = []
        for t in range(n_steps):
            a_off, a_on = alphas[t]
            b_off, b_on = betas[t]
            on_mass = a_on * b_on
            off_mass = a_off * b_off
            norm = on_mass + off_mass
            if norm > 0:
                posteriors.append(round(on_mass / norm, 12))
            else:
                posteriors.append(0.5)
        return posteriors
