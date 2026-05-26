from csbf.evaluation import (
    bootstrap_ci,
    evaluate_hmm_prefix_diagnostics,
    evaluate_predictions,
    evaluate_trace_methods,
    evaluate_trace_methods_by_length,
)
from csbf.metrics import brier_score
from csbf.schema import Observation, TraceRecord


def _trace(trace_id: str, correct: bool, scores: list[float], codes: list[int]) -> TraceRecord:
    return TraceRecord(
        question_id=trace_id.split("-")[0],
        question="question",
        trace_id=trace_id,
        trace_text="trace",
        final_answer="1" if correct else "2",
        gold_answer="1",
        correct=correct,
        observations=[
            Observation(step_index=index, text=f"step {index}", score=score, concept_code=code)
            for index, (score, code) in enumerate(zip(scores, codes))
        ],
    )


def test_evaluate_predictions_returns_core_metrics():
    metrics = evaluate_predictions([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.8], bins=4)

    assert metrics["brier"] == 0.075
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_evaluate_predictions_marks_rank_metrics_none_for_single_class_labels():
    metrics = evaluate_predictions([1, 1], [0.6, 0.8], bins=2)

    assert metrics["auroc"] is None
    assert metrics["auprc"] == 1.0


def test_evaluate_trace_methods_scores_baselines_and_hmm_variants():
    calibration = [
        _trace("q1-t0", False, [0.2, 0.3], [1, 1]),
        _trace("q2-t0", True, [0.7, 0.8], [2, 2]),
    ]
    test = [
        _trace("q3-t0", False, [0.3, 0.2], [1, 1]),
        _trace("q4-t0", True, [0.6, 0.9], [2, 2]),
    ]

    report = evaluate_trace_methods(calibration, test, moving_average_window=2, ema_alpha=0.5)

    assert set(report.keys()) == {
        "last_step",
        "calibrated_last_step",
        "mean_score",
        "moving_average",
        "calibrated_moving_average",
        "ema",
        "calibrated_ema",
        "temporal_metric",
        "learned_prefix_baseline",
        "prefix_feature_classifier",
        "emission_only_concept",
        "emission_only_hybrid",
        "emission_only_joint",
        "hmm_score",
        "hmm_concept",
        "hmm_hybrid",
        "hmm_joint",
        "hmm_nonstationary",
        "hmm_smooth",
    }
    assert report["calibrated_last_step"]["metrics"]["auroc"] == 1.0
    assert report["hmm_hybrid"]["metrics"]["auroc"] == 1.0


def test_emission_only_hybrid_uses_final_prefix_without_temporal_recursion():
    calibration = [
        _trace("q1-t0", False, [0.1, 0.1], ["low", "low"]),
        _trace("q2-t0", True, [0.9, 0.9], ["high", "high"]),
        _trace("q3-t0", False, [0.1, 0.1], ["low", "low"]),
        _trace("q4-t0", True, [0.9, 0.9], ["high", "high"]),
    ]
    test = [
        _trace("q5-t0", False, [0.1, 0.5], ["low", "neutral"]),
        _trace("q6-t0", True, [0.9, 0.5], ["high", "neutral"]),
    ]

    report = evaluate_trace_methods(calibration, test, moving_average_window=2, ema_alpha=0.5)

    emission_only = report["emission_only_hybrid"]["predictions"]
    temporal = report["hmm_hybrid"]["predictions"]
    assert emission_only[0] == emission_only[1]
    assert temporal[1] > temporal[0]


def test_prefix_feature_classifier_uses_concept_observations_when_scores_tie():
    calibration = [
        _trace("q1-t0", False, [0.5, 0.5, 0.5], ["low", "low", "low"]),
        _trace("q2-t0", True, [0.5, 0.5, 0.5], ["high", "high", "high"]),
        _trace("q3-t0", False, [0.5, 0.5, 0.5], ["low", "low", "low"]),
        _trace("q4-t0", True, [0.5, 0.5, 0.5], ["high", "high", "high"]),
    ]
    test = [
        _trace("q5-t0", False, [0.5, 0.5, 0.5], ["low", "low", "low"]),
        _trace("q6-t0", True, [0.5, 0.5, 0.5], ["high", "high", "high"]),
        _trace("q7-t0", False, [0.5, 0.5, 0.5], ["low", "low", "low"]),
    ]

    report = evaluate_trace_methods(calibration, test, moving_average_window=2, ema_alpha=0.5)
    predictions = report["prefix_feature_classifier"]["predictions"]

    assert len(predictions) == len(test)
    assert all(0.0 <= value <= 1.0 for value in predictions)
    assert predictions[1] > predictions[0]
    assert predictions[1] > predictions[2]
    assert report["prefix_feature_classifier"]["metrics"]["auroc"] == 1.0


def test_evaluate_trace_methods_includes_calibrated_temporal_baselines():
    calibration = [
        _trace("q1-t0", False, [0.2, 0.4, 0.3], [1, 1, 1]),
        _trace("q2-t0", True, [0.6, 0.7, 0.8], [2, 2, 2]),
        _trace("q3-t0", False, [0.3, 0.2, 0.1], [1, 1, 1]),
        _trace("q4-t0", True, [0.8, 0.9, 0.9], [2, 2, 2]),
    ]
    test = [
        _trace("q5-t0", False, [0.4, 0.3, 0.2], [1, 1, 1]),
        _trace("q6-t0", True, [0.5, 0.8, 0.9], [2, 2, 2]),
    ]

    report = evaluate_trace_methods(calibration, test, moving_average_window=2, ema_alpha=0.5)

    for method in ("calibrated_ema", "calibrated_moving_average"):
        assert method in report
        predictions = report[method]["predictions"]
        assert len(predictions) == len(test)
        assert all(0.0 <= value <= 1.0 for value in predictions)


def test_evaluate_trace_methods_includes_learned_prefix_baseline():
    calibration = [
        _trace("q1-t0", False, [0.2, 0.4, 0.3], [1, 1, 1]),
        _trace("q2-t0", True, [0.6, 0.7, 0.8], [2, 2, 2]),
        _trace("q3-t0", False, [0.3, 0.2, 0.1], [1, 1, 1]),
        _trace("q4-t0", True, [0.8, 0.9, 0.9], [2, 2, 2]),
    ]
    test = [
        _trace("q5-t0", False, [0.4, 0.3, 0.2], [1, 1, 1]),
        _trace("q6-t0", True, [0.5, 0.8, 0.9], [2, 2, 2]),
        _trace("q7-t0", False, [0.7, 0.4, 0.2], [1, 1, 1]),
    ]

    report = evaluate_trace_methods(calibration, test, moving_average_window=2, ema_alpha=0.5)
    predictions = report["learned_prefix_baseline"]["predictions"]

    assert len(predictions) == len(test)
    assert all(0.0 <= value <= 1.0 for value in predictions)
    assert len({round(value, 6) for value in predictions}) > 1


def test_evaluate_trace_methods_supports_em_calibration():
    calibration = [
        _trace("q1-t0", False, [0.2, 0.3, 0.2], [1, 1, 1]),
        _trace("q2-t0", True, [0.7, 0.8, 0.9], [2, 2, 2]),
    ]
    test = [
        _trace("q3-t0", False, [0.3, 0.2], [1, 1]),
        _trace("q4-t0", True, [0.6, 0.9], [2, 2]),
    ]

    report = evaluate_trace_methods(calibration, test, calibration_mode="em")

    assert "hmm_hybrid" in report
    assert "hmm_joint" in report
    assert report["hmm_score"]["metrics"]["brier"] >= 0.0


def test_hmm_prefix_diagnostics_compare_online_and_offline_future_information():
    calibration = [
        _trace("q1-t0", False, [0.1, 0.1, 0.1], [1, 1, 1]),
        _trace("q2-t0", True, [0.9, 0.9, 0.9], [2, 2, 2]),
    ]
    test = [
        _trace("q3-t0", False, [0.5, 0.1, 0.1], [0, 1, 1]),
        _trace("q4-t0", True, [0.5, 0.9, 0.9], [0, 2, 2]),
    ]

    report = evaluate_hmm_prefix_diagnostics(calibration, test, fractions=(0.0,))

    assert report["p00"]["step_indices"] == [0, 0]
    assert report["p00"]["hmm_hybrid_online"]["metrics"]["auroc"] == 0.5
    assert report["p00"]["hmm_hybrid_smooth"]["metrics"]["auroc"] == 1.0
    assert report["p00"]["smooth_minus_online_auroc"] == 0.5
    assert report["p00"]["hmm_hybrid_viterbi"]["metrics"]["auroc"] == 1.0


def test_hmm_prefix_diagnostics_use_half_up_fraction_indices():
    calibration = [
        _trace("q1-t0", False, [0.1, 0.1], [1, 1]),
        _trace("q2-t0", True, [0.9, 0.9], [2, 2]),
    ]
    test = [
        _trace("q3-t0", False, [0.5, 0.1], [0, 1]),
        _trace("q4-t0", True, [0.5, 0.9], [0, 2]),
    ]

    report = evaluate_hmm_prefix_diagnostics(calibration, test, fractions=(0.5,))

    assert report["p50"]["step_indices"] == [1, 1]


def test_bootstrap_ci_contains_point_estimate():
    labels = [0, 0, 1, 1, 0, 1]
    preds = [0.1, 0.3, 0.7, 0.9, 0.4, 0.6]
    point, lo, hi = bootstrap_ci(labels, preds, brier_score)
    assert lo <= point <= hi


def test_evaluate_trace_methods_by_length_groups_correctly():
    calibration = [
        _trace("q1-t0", False, [0.2, 0.3], [1, 1]),
        _trace("q2-t0", True, [0.7, 0.8], [2, 2]),
    ]
    short_1 = _trace("q3-t0", False, [0.3, 0.2], [1, 1])
    short_2 = _trace("q4-t0", True, [0.6, 0.9], [2, 2])
    long_1 = _trace("q5-t0", False, [0.1, 0.2, 0.3, 0.2, 0.1], [1, 1, 1, 1, 1])
    long_2 = _trace("q6-t0", True, [0.7, 0.8, 0.9, 0.8, 0.9], [2, 2, 2, 2, 2])

    test = [short_1, short_2, long_1, long_2]
    buckets = {"short": (1, 3), "long": (4, 10)}
    report = evaluate_trace_methods_by_length(
        calibration, test, length_buckets=buckets, moving_average_window=2, ema_alpha=0.5,
    )
    assert "short" in report
    assert "long" in report
