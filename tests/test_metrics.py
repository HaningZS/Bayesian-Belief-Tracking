from csbf.metrics import brier_score, expected_calibration_error, roc_auc_score


def test_brier_score_matches_mean_squared_probability_error():
    assert brier_score([0, 1, 1], [0.1, 0.8, 0.4]) == 0.136666666667


def test_expected_calibration_error_uses_confidence_bins():
    ece = expected_calibration_error([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4], bins=2)

    assert ece == 0.25


def test_roc_auc_score_handles_ordered_binary_scores():
    auc = roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])

    assert auc == 0.75
