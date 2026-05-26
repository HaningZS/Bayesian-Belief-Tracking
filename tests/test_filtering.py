from csbf.calibration import ConceptLikelihood, HistogramLikelihood
from csbf.filtering import (
    BayesianReliabilityFilter,
    HMMConfig,
    NonStationaryHMMConfig,
)


def test_bayesian_filter_combines_score_and_concept_observations_online():
    score_likelihood = HistogramLikelihood.fit(
        scores=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
        bins=4,
        smoothing=0.1,
    )
    concept_likelihood = ConceptLikelihood.fit(
        concept_codes=[3, 3, 7, 7],
        labels=[0, 0, 1, 1],
        smoothing=0.1,
    )
    reliability_filter = BayesianReliabilityFilter(
        config=HMMConfig(p_error=0.10, p_recover=0.20, initial_on_track=0.5),
        score_likelihood=score_likelihood,
        concept_likelihood=concept_likelihood,
    )

    beliefs = reliability_filter.run(scores=[0.15, 0.85], concept_codes=[3, 7])

    assert len(beliefs) == 2
    assert all(0.0 <= belief <= 1.0 for belief in beliefs)
    assert beliefs[1] > beliefs[0]


def test_viterbi_returns_most_likely_state_sequence():
    score_likelihood = HistogramLikelihood.fit(
        scores=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
        bins=4,
        smoothing=0.1,
    )
    f = BayesianReliabilityFilter(
        config=HMMConfig(p_error=0.05, p_recover=0.05, initial_on_track=0.9),
        score_likelihood=score_likelihood,
    )

    scores = [0.9, 0.85, 0.8, 0.15, 0.1, 0.05]
    path = f.decode(scores=scores)

    assert len(path) == 6
    assert all(s in (0, 1) for s in path)
    assert path[0] == 1
    assert path[-1] == 0


def test_forward_backward_smoothed_posteriors_use_full_trace():
    score_likelihood = HistogramLikelihood.fit(
        scores=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
        bins=4,
        smoothing=0.1,
    )
    f = BayesianReliabilityFilter(
        config=HMMConfig(p_error=0.05, p_recover=0.05, initial_on_track=0.5),
        score_likelihood=score_likelihood,
    )

    scores = [0.5, 0.85, 0.9]
    forward_only = f.run(scores=scores)
    smoothed = f.smooth(scores=scores)

    assert len(smoothed) == len(forward_only)
    assert smoothed != forward_only
    assert smoothed[0] > forward_only[0]


def test_nonstationary_transitions_increase_error_rate_with_position():
    score_likelihood = HistogramLikelihood.fit(
        scores=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
        bins=4,
        smoothing=0.1,
    )

    stationary_f = BayesianReliabilityFilter(
        config=HMMConfig(p_error=0.03, p_recover=0.15, initial_on_track=0.5),
        score_likelihood=score_likelihood,
    )

    nonstationary_f = BayesianReliabilityFilter(
        config=NonStationaryHMMConfig(
            p_error_base=0.03,
            p_error_slope=0.005,
            p_recover_base=0.15,
            p_recover_decay=0.02,
            initial_on_track=0.5,
        ),
        score_likelihood=score_likelihood,
    )

    scores = [0.5] * 10
    stationary_beliefs = stationary_f.run(scores=scores)
    nonstationary_beliefs = nonstationary_f.run(scores=scores)

    assert stationary_beliefs != nonstationary_beliefs


def test_viterbi_and_forward_agree_on_trivial_case():
    score_likelihood = HistogramLikelihood.fit(
        scores=[0.1, 0.9],
        labels=[0, 1],
        bins=2,
        smoothing=0.1,
    )
    f = BayesianReliabilityFilter(
        config=HMMConfig(p_error=0.01, p_recover=0.01, initial_on_track=0.5),
        score_likelihood=score_likelihood,
    )

    scores = [0.95, 0.95, 0.95]
    path = f.decode(scores=scores)
    beliefs = f.run(scores=scores)

    assert all(s == 1 for s in path)
    assert all(b > 0.5 for b in beliefs)
