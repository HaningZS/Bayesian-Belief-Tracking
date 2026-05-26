from csbf.baselines import exponential_moving_average, moving_average, temporal_score_metric


def test_moving_average_uses_prefix_only_window():
    scores = [0.2, 0.4, 0.8, 0.6]

    averaged = moving_average(scores, window=2)

    assert averaged == [0.2, 0.3, 0.6, 0.7]


def test_temporal_score_metric_penalizes_declining_scores():
    declining = temporal_score_metric([0.8, 0.5, 0.2])
    stable = temporal_score_metric([0.6, 0.6, 0.6])
    assert declining < stable


def test_exponential_moving_average_stays_online_and_bounded():
    scores = [0.2, 0.8, 0.4]

    averaged = exponential_moving_average(scores, alpha=0.5)

    assert averaged == [0.2, 0.5, 0.45]
