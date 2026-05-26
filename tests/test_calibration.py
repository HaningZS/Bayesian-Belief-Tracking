from csbf.calibration import ConceptLikelihood, HistogramLikelihood, QuantileLikelihood


def test_histogram_likelihood_smooths_empty_bins():
    likelihood = HistogramLikelihood.fit(
        scores=[0.05, 0.15, 0.85, 0.95],
        labels=[0, 0, 1, 1],
        bins=4,
        smoothing=0.5,
    )

    off_track = likelihood.probability(0.65, state=0)
    on_track = likelihood.probability(0.65, state=1)

    assert off_track > 0
    assert on_track > 0


def test_concept_likelihood_smooths_unseen_codes():
    likelihood = ConceptLikelihood.fit(
        concept_codes=[1, 1, 2, 2],
        labels=[0, 0, 1, 1],
        smoothing=0.25,
    )

    assert likelihood.probability(99, state=0) > 0
    assert likelihood.probability(99, state=1) > 0


def test_concept_likelihood_treats_unseen_codes_as_neutral_evidence():
    likelihood = ConceptLikelihood.fit(
        concept_codes=[7, 8],
        labels=[1, 1],
        smoothing=0.25,
    )

    assert likelihood.probability(99, state=0) == likelihood.probability(99, state=1)


def test_quantile_likelihood_adapts_bins_to_score_distribution():
    scores = [0.01, 0.02, 0.03, 0.04, 0.05, 0.90, 0.92, 0.95, 0.97, 0.99]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    ql = QuantileLikelihood.fit(scores, labels, bins=5, smoothing=0.5)

    assert len(ql.bin_edges) >= 3
    actual_bins = len(ql.bin_edges) - 1
    assert actual_bins >= 2

    for score in [0.0, 0.5, 1.0]:
        for state in [0, 1]:
            p = ql.probability(score, state)
            assert 0.0 < p <= 1.0
