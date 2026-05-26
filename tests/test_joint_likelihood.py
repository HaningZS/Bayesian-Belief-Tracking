from csbf.calibration import JointLikelihood, ON_TRACK, OFF_TRACK


def test_joint_likelihood_models_score_concept_interaction():
    jl = JointLikelihood.fit(
        scores=[0.2, 0.3, 0.8, 0.9],
        concept_codes=["calc", "calc", "verify", "verify"],
        labels=[0, 0, 1, 1],
        score_bins=4,
        smoothing=0.5,
    )
    p_on = jl.probability(0.85, "verify", ON_TRACK)
    p_off = jl.probability(0.85, "verify", OFF_TRACK)
    assert p_on > p_off


def test_joint_likelihood_handles_unseen_concept():
    jl = JointLikelihood.fit(
        scores=[0.1, 0.9],
        concept_codes=["a", "a"],
        labels=[0, 1],
        score_bins=2,
        smoothing=1.0,
    )
    p = jl.probability(0.5, "never_seen", ON_TRACK)
    assert p > 0


def test_joint_likelihood_rejects_bad_bins():
    try:
        JointLikelihood.fit(
            scores=[0.5], concept_codes=["a"], labels=[1], score_bins=0,
        )
    except ValueError as error:
        assert "score_bins" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_joint_likelihood_rejects_bad_smoothing():
    try:
        JointLikelihood.fit(
            scores=[0.5], concept_codes=["a"], labels=[1], smoothing=0.0,
        )
    except ValueError as error:
        assert "smoothing" in str(error)
    else:
        raise AssertionError("expected ValueError")
