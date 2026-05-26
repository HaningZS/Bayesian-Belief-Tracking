"""Method evaluation over pre-observed reasoning traces."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, log

from csbf.baselines import exponential_moving_average, moving_average, temporal_score_metric
from csbf.calibration import OFF_TRACK, ON_TRACK, ConceptLikelihood, HistogramLikelihood, JointLikelihood
from csbf.em import em_fit_score_likelihood
from csbf.filtering import BayesianReliabilityFilter, HMMConfig, NonStationaryHMMConfig
from csbf.metrics import average_precision_score, brier_score, expected_calibration_error, roc_auc_score
from csbf.schema import TraceRecord


@dataclass(frozen=True)
class _PrefixFeatureClassifier:
    weights: list[float]
    concept_vocab: tuple[int | str, ...]


def evaluate_predictions(labels: Sequence[int], probabilities: Sequence[float], bins: int = 10) -> dict[str, float | None]:
    """Evaluate binary probability predictions."""

    return {
        "brier": brier_score(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities, bins=bins),
        "auroc": _optional_metric(roc_auc_score, labels, probabilities),
        "auprc": _optional_metric(average_precision_score, labels, probabilities),
    }


def evaluate_trace_methods(
    calibration_records: list[TraceRecord],
    test_records: list[TraceRecord],
    moving_average_window: int = 3,
    ema_alpha: float = 0.4,
    hmm_config: HMMConfig | NonStationaryHMMConfig | None = None,
    calibration_mode: str = "all_steps",
) -> dict[str, dict[str, object]]:
    """Evaluate score baselines and HMM variants on held-out traces."""

    if not calibration_records:
        raise ValueError("calibration_records must not be empty")
    if not test_records:
        raise ValueError("test_records must not be empty")
    if calibration_mode not in ("all_steps", "final_step", "em"):
        raise ValueError("calibration_mode must be 'all_steps', 'final_step', or 'em'")

    config = hmm_config or HMMConfig(initial_on_track=_base_rate(calibration_records))
    labels = [1 if record.correct else 0 for record in test_records]

    score_likelihood, concept_likelihood, joint_likelihood = _fit_likelihoods(
        calibration_records,
        config,
        calibration_mode,
    )
    moving_average_likelihood = _fit_prediction_likelihood(
        calibration_records,
        lambda record: moving_average(_scores(record), moving_average_window)[-1],
    )
    ema_likelihood = _fit_prediction_likelihood(
        calibration_records,
        lambda record: exponential_moving_average(_scores(record), ema_alpha)[-1],
    )
    learned_prefix_model = _fit_learned_prefix_model(
        calibration_records,
        moving_average_window=moving_average_window,
        ema_alpha=ema_alpha,
    )
    prefix_feature_classifier = _fit_prefix_feature_classifier(
        calibration_records,
        moving_average_window=moving_average_window,
        ema_alpha=ema_alpha,
    )

    nonstationary_config = _default_nonstationary_config(config)

    methods = {
        "last_step": [_last_score(record) for record in test_records],
        "calibrated_last_step": [
            _calibrated_score_prediction(_last_score(record), score_likelihood, config.initial_on_track)
            for record in test_records
        ],
        "mean_score": [_mean(_scores(record)) for record in test_records],
        "moving_average": [moving_average(_scores(record), moving_average_window)[-1] for record in test_records],
        "calibrated_moving_average": [
            _calibrated_score_prediction(
                moving_average(_scores(record), moving_average_window)[-1],
                moving_average_likelihood,
                config.initial_on_track,
            )
            for record in test_records
        ],
        "ema": [exponential_moving_average(_scores(record), ema_alpha)[-1] for record in test_records],
        "calibrated_ema": [
            _calibrated_score_prediction(
                exponential_moving_average(_scores(record), ema_alpha)[-1],
                ema_likelihood,
                config.initial_on_track,
            )
            for record in test_records
        ],
        "temporal_metric": [temporal_score_metric(_scores(record)) for record in test_records],
        "learned_prefix_baseline": [
            _predict_learned_prefix_model(
                learned_prefix_model,
                _scores(record),
                moving_average_window=moving_average_window,
                ema_alpha=ema_alpha,
            )
            for record in test_records
        ],
        "prefix_feature_classifier": [
            _predict_prefix_feature_classifier(
                prefix_feature_classifier,
                record,
                moving_average_window=moving_average_window,
                ema_alpha=ema_alpha,
            )
            for record in test_records
        ],
        "emission_only_concept": [
            _concept_observation_only_prediction(record, concept_likelihood, config.initial_on_track)
            for record in test_records
        ],
        "emission_only_hybrid": [
            _hybrid_observation_only_prediction(
                record,
                score_likelihood,
                concept_likelihood,
                config.initial_on_track,
            )
            for record in test_records
        ],
        "emission_only_joint": [
            _joint_observation_only_prediction(record, joint_likelihood, config.initial_on_track)
            for record in test_records
        ],
        "hmm_score": [
            BayesianReliabilityFilter(config=config, score_likelihood=score_likelihood).run(scores=_scores(record))[-1]
            for record in test_records
        ],
        "hmm_concept": [
            BayesianReliabilityFilter(config=config, concept_likelihood=concept_likelihood).run(concept_codes=_codes(record))[-1]
            for record in test_records
        ],
        "hmm_hybrid": [
            _hybrid_prediction(record, config, score_likelihood, concept_likelihood)
            for record in test_records
        ],
        "hmm_joint": [
            _joint_prediction(record, config, joint_likelihood)
            for record in test_records
        ],
        "hmm_nonstationary": [
            _hybrid_prediction(record, nonstationary_config, score_likelihood, concept_likelihood)
            for record in test_records
        ],
        "hmm_smooth": [
            _smooth_prediction(record, config, score_likelihood, concept_likelihood)
            for record in test_records
        ],
    }

    return {
        name: {
            "predictions": predictions,
            "metrics": evaluate_predictions(labels, predictions),
        }
        for name, predictions in methods.items()
    }


def evaluate_hmm_prefix_diagnostics(
    calibration_records: list[TraceRecord],
    test_records: list[TraceRecord],
    fractions: Sequence[float] = (0.05, 0.50),
    hmm_config: HMMConfig | NonStationaryHMMConfig | None = None,
    calibration_mode: str = "all_steps",
) -> dict[str, dict[str, object]]:
    """Compare online vs offline HMM beliefs at fixed trace-prefix fractions."""

    if not calibration_records:
        raise ValueError("calibration_records must not be empty")
    if not test_records:
        raise ValueError("test_records must not be empty")
    if calibration_mode not in ("all_steps", "final_step", "em"):
        raise ValueError("calibration_mode must be 'all_steps', 'final_step', or 'em'")
    for fraction in fractions:
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("fractions must be in [0, 1]")

    config = hmm_config or HMMConfig(initial_on_track=_base_rate(calibration_records))
    score_likelihood, concept_likelihood, _ = _fit_likelihoods(
        calibration_records,
        config,
        calibration_mode,
    )
    labels = [1 if record.correct else 0 for record in test_records]
    report: dict[str, dict[str, object]] = {}
    for fraction in fractions:
        online_predictions: list[float] = []
        smooth_predictions: list[float] = []
        viterbi_predictions: list[float] = []
        step_indices: list[int] = []
        for record in test_records:
            scores, codes = _paired_scores_codes(record)
            step_index = _fraction_index(len(scores), float(fraction))
            reliability_filter = BayesianReliabilityFilter(
                config=config,
                score_likelihood=score_likelihood,
                concept_likelihood=concept_likelihood,
            )
            online_predictions.append(reliability_filter.run(scores=scores, concept_codes=codes)[step_index])
            smooth_predictions.append(reliability_filter.smooth(scores=scores, concept_codes=codes)[step_index])
            viterbi_predictions.append(float(reliability_filter.decode(scores=scores, concept_codes=codes)[step_index]))
            step_indices.append(step_index)

        online_metrics = evaluate_predictions(labels, online_predictions)
        smooth_metrics = evaluate_predictions(labels, smooth_predictions)
        viterbi_metrics = evaluate_predictions(labels, viterbi_predictions)
        online_auroc = online_metrics["auroc"]
        smooth_auroc = smooth_metrics["auroc"]
        report[_fraction_key(float(fraction))] = {
            "fraction": float(fraction),
            "step_indices": step_indices,
            "hmm_hybrid_online": {
                "predictions": online_predictions,
                "metrics": online_metrics,
            },
            "hmm_hybrid_smooth": {
                "predictions": smooth_predictions,
                "metrics": smooth_metrics,
            },
            "hmm_hybrid_viterbi": {
                "predictions": viterbi_predictions,
                "metrics": viterbi_metrics,
            },
            "smooth_minus_online_auroc": _subtract_optional(smooth_auroc, online_auroc),
        }
    return report


def bootstrap_ci(
    labels: Sequence[int],
    predictions: Sequence[float],
    metric_fn: Callable[[Sequence[int], Sequence[float]], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_lower, ci_upper) via bootstrap resampling."""
    point = metric_fn(list(labels), list(predictions))
    rng = random.Random(seed)
    n = len(labels)
    bootstrap_values: list[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        boot_labels = [int(labels[i]) for i in indices]
        boot_preds = [float(predictions[i]) for i in indices]
        try:
            bootstrap_values.append(metric_fn(boot_labels, boot_preds))
        except ValueError:
            continue
    if not bootstrap_values:
        return point, point, point
    bootstrap_values.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = max(0, int(alpha * len(bootstrap_values)))
    hi_idx = min(len(bootstrap_values) - 1, int((1.0 - alpha) * len(bootstrap_values)))
    return round(point, 12), round(bootstrap_values[lo_idx], 12), round(bootstrap_values[hi_idx], 12)


def evaluate_trace_methods_by_length(
    calibration_records: list[TraceRecord],
    test_records: list[TraceRecord],
    length_buckets: dict[str, tuple[int, int | float]] | None = None,
    moving_average_window: int = 3,
    ema_alpha: float = 0.4,
    hmm_config: HMMConfig | NonStationaryHMMConfig | None = None,
    calibration_mode: str = "all_steps",
) -> dict[str, dict[str, dict[str, object]]]:
    """Evaluate methods grouped by trace length (number of observations)."""

    buckets = length_buckets or {
        "short": (1, 3),
        "medium": (4, 7),
        "long": (8, float("inf")),
    }
    results: dict[str, dict[str, dict[str, object]]] = {}
    for bucket_name, (lo, hi) in buckets.items():
        bucket_test = [
            r for r in test_records
            if lo <= len(r.observations) <= hi
        ]
        if len(bucket_test) < 2:
            continue
        try:
            results[bucket_name] = evaluate_trace_methods(
                calibration_records, bucket_test,
                moving_average_window=moving_average_window,
                ema_alpha=ema_alpha,
                hmm_config=hmm_config,
                calibration_mode=calibration_mode,
            )
        except ValueError:
            continue
    return results


def _fit_score_likelihood(records: list[TraceRecord]) -> HistogramLikelihood:
    scores: list[float] = []
    labels: list[int] = []
    for record in records:
        for score in _scores(record):
            scores.append(score)
            labels.append(1 if record.correct else 0)
    return HistogramLikelihood.fit(scores=scores, labels=labels, bins=10, smoothing=0.5)


def _fit_prediction_likelihood(
    records: list[TraceRecord],
    prediction_fn: Callable[[TraceRecord], float],
) -> HistogramLikelihood:
    scores = [float(prediction_fn(record)) for record in records]
    labels = [1 if record.correct else 0 for record in records]
    return HistogramLikelihood.fit(scores=scores, labels=labels, bins=10, smoothing=0.5)


def _fit_likelihoods(
    records: list[TraceRecord],
    config: HMMConfig | NonStationaryHMMConfig,
    calibration_mode: str,
) -> tuple[HistogramLikelihood, ConceptLikelihood, JointLikelihood]:
    if calibration_mode == "em":
        return (
            em_fit_score_likelihood(records, _stationary_config(config)),
            _fit_concept_likelihood_final_step(records),
            _fit_joint_likelihood_final_step(records),
        )
    if calibration_mode == "final_step":
        return (
            _fit_score_likelihood_final_step(records),
            _fit_concept_likelihood_final_step(records),
            _fit_joint_likelihood_final_step(records),
        )
    return (
        _fit_score_likelihood(records),
        _fit_concept_likelihood(records),
        _fit_joint_likelihood(records),
    )


def _fit_concept_likelihood(records: list[TraceRecord]) -> ConceptLikelihood:
    codes: list[int | str] = []
    labels: list[int] = []
    for record in records:
        for code in _codes(record):
            codes.append(code)
            labels.append(1 if record.correct else 0)
    return ConceptLikelihood.fit(concept_codes=codes, labels=labels, smoothing=0.5)


def _fit_joint_likelihood(records: list[TraceRecord]) -> JointLikelihood:
    scores: list[float] = []
    codes: list[int | str] = []
    labels: list[int] = []
    for record in records:
        for score, code in _paired_score_code_observations(record):
            scores.append(score)
            codes.append(code)
            labels.append(1 if record.correct else 0)
    return JointLikelihood.fit(scores=scores, concept_codes=codes, labels=labels, score_bins=5, smoothing=0.5)


def _fit_score_likelihood_final_step(records: list[TraceRecord]) -> HistogramLikelihood:
    """Calibrate using only the final score observation per trace."""
    scores = [_scores(record)[-1] for record in records]
    labels = [1 if record.correct else 0 for record in records]
    return HistogramLikelihood.fit(scores=scores, labels=labels, bins=10, smoothing=0.5)


def _fit_concept_likelihood_final_step(records: list[TraceRecord]) -> ConceptLikelihood:
    """Calibrate using only the final concept observation per trace."""
    codes = [_codes(record)[-1] for record in records]
    labels = [1 if record.correct else 0 for record in records]
    return ConceptLikelihood.fit(concept_codes=codes, labels=labels, smoothing=0.5)


def _fit_joint_likelihood_final_step(records: list[TraceRecord]) -> JointLikelihood:
    """Calibrate joint score-concept likelihood using only each trace's final paired observation."""
    pairs = [_paired_score_code_observations(record)[-1] for record in records]
    scores = [score for score, _ in pairs]
    codes = [code for _, code in pairs]
    labels = [1 if record.correct else 0 for record in records]
    return JointLikelihood.fit(scores=scores, concept_codes=codes, labels=labels, score_bins=5, smoothing=0.5)


def _scores(record: TraceRecord) -> list[float]:
    scores = [float(observation.score) for observation in record.observations if observation.score is not None]
    if not scores:
        raise ValueError(f"trace {record.trace_id} has no score observations")
    return scores


def _codes(record: TraceRecord) -> list[int | str]:
    codes = [observation.concept_code for observation in record.observations if observation.concept_code is not None]
    if not codes:
        raise ValueError(f"trace {record.trace_id} has no concept_code observations")
    return codes


def _paired_score_code_observations(record: TraceRecord) -> list[tuple[float, int | str]]:
    pairs = [
        (float(observation.score), observation.concept_code)
        for observation in record.observations
        if observation.score is not None and observation.concept_code is not None
    ]
    if not pairs:
        raise ValueError(f"trace {record.trace_id} has no paired score/concept observations")
    return pairs


def _paired_scores_codes(record: TraceRecord) -> tuple[list[float], list[int | str]]:
    pairs = _paired_score_code_observations(record)
    return [score for score, _ in pairs], [code for _, code in pairs]


def _hybrid_prediction(
    record: TraceRecord,
    config: HMMConfig | NonStationaryHMMConfig,
    score_likelihood: HistogramLikelihood,
    concept_likelihood: ConceptLikelihood,
) -> float:
    scores, codes = _paired_scores_codes(record)
    return BayesianReliabilityFilter(
        config=config,
        score_likelihood=score_likelihood,
        concept_likelihood=concept_likelihood,
    ).run(scores=scores, concept_codes=codes)[-1]


def _joint_prediction(
    record: TraceRecord,
    config: HMMConfig | NonStationaryHMMConfig,
    joint_likelihood: JointLikelihood,
) -> float:
    scores, codes = _paired_scores_codes(record)
    return BayesianReliabilityFilter(
        config=config,
        joint_likelihood=joint_likelihood,
    ).run(scores=scores, concept_codes=codes)[-1]


def _smooth_prediction(
    record: TraceRecord,
    config: HMMConfig | NonStationaryHMMConfig,
    score_likelihood: HistogramLikelihood,
    concept_likelihood: ConceptLikelihood,
) -> float:
    scores, codes = _paired_scores_codes(record)
    return BayesianReliabilityFilter(
        config=config,
        score_likelihood=score_likelihood,
        concept_likelihood=concept_likelihood,
    ).smooth(scores=scores, concept_codes=codes)[-1]


def _last_score(record: TraceRecord) -> float:
    return _scores(record)[-1]


def _calibrated_score_prediction(
    score: float,
    likelihood: HistogramLikelihood,
    prior_on_track: float,
) -> float:
    prior = min(max(float(prior_on_track), 0.0), 1.0)
    p_on = likelihood.probability(score, ON_TRACK) * prior
    p_off = likelihood.probability(score, OFF_TRACK) * (1.0 - prior)
    denominator = p_on + p_off
    if denominator <= 0.0:
        return prior
    return p_on / denominator


def _concept_observation_only_prediction(
    record: TraceRecord,
    concept_likelihood: ConceptLikelihood,
    prior_on_track: float,
) -> float:
    code = _codes(record)[-1]
    return _normalize_observation_posterior(
        on_likelihood=concept_likelihood.probability(code, ON_TRACK),
        off_likelihood=concept_likelihood.probability(code, OFF_TRACK),
        prior_on_track=prior_on_track,
    )


def _hybrid_observation_only_prediction(
    record: TraceRecord,
    score_likelihood: HistogramLikelihood,
    concept_likelihood: ConceptLikelihood,
    prior_on_track: float,
) -> float:
    score, code = _paired_score_code_observations(record)[-1]
    return _normalize_observation_posterior(
        on_likelihood=score_likelihood.probability(score, ON_TRACK)
        * concept_likelihood.probability(code, ON_TRACK),
        off_likelihood=score_likelihood.probability(score, OFF_TRACK)
        * concept_likelihood.probability(code, OFF_TRACK),
        prior_on_track=prior_on_track,
    )


def _joint_observation_only_prediction(
    record: TraceRecord,
    joint_likelihood: JointLikelihood,
    prior_on_track: float,
) -> float:
    score, code = _paired_score_code_observations(record)[-1]
    return _normalize_observation_posterior(
        on_likelihood=joint_likelihood.probability(score, code, ON_TRACK),
        off_likelihood=joint_likelihood.probability(score, code, OFF_TRACK),
        prior_on_track=prior_on_track,
    )


def _normalize_observation_posterior(
    on_likelihood: float,
    off_likelihood: float,
    prior_on_track: float,
) -> float:
    prior = min(max(float(prior_on_track), 0.0), 1.0)
    p_on = float(on_likelihood) * prior
    p_off = float(off_likelihood) * (1.0 - prior)
    denominator = p_on + p_off
    if denominator <= 0.0:
        return prior
    return p_on / denominator


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 12)


def _fit_learned_prefix_model(
    records: list[TraceRecord],
    moving_average_window: int,
    ema_alpha: float,
) -> list[float]:
    labels = [1.0 if record.correct else 0.0 for record in records]
    features = [
        _prefix_score_features(_scores(record), moving_average_window=moving_average_window, ema_alpha=ema_alpha)
        for record in records
    ]
    return _fit_logistic_weights(features, labels, learning_rate=0.35, l2_penalty=0.01, iterations=400)


def _predict_learned_prefix_model(
    weights: Sequence[float],
    scores: Sequence[float],
    moving_average_window: int,
    ema_alpha: float,
) -> float:
    features = _prefix_score_features(
        scores,
        moving_average_window=moving_average_window,
        ema_alpha=ema_alpha,
    )
    return round(_sigmoid(_dot(weights, features)), 12)


def _fit_prefix_feature_classifier(
    records: list[TraceRecord],
    moving_average_window: int,
    ema_alpha: float,
) -> _PrefixFeatureClassifier:
    labels = [1.0 if record.correct else 0.0 for record in records]
    concept_vocab = _concept_vocab(records)
    features = [
        _prefix_feature_classifier_features(
            record,
            concept_vocab=concept_vocab,
            moving_average_window=moving_average_window,
            ema_alpha=ema_alpha,
        )
        for record in records
    ]
    weights = _fit_logistic_weights(features, labels, learning_rate=0.25, l2_penalty=0.02, iterations=500)
    return _PrefixFeatureClassifier(weights=weights, concept_vocab=concept_vocab)


def _predict_prefix_feature_classifier(
    model: _PrefixFeatureClassifier,
    record: TraceRecord,
    moving_average_window: int,
    ema_alpha: float,
) -> float:
    features = _prefix_feature_classifier_features(
        record,
        concept_vocab=model.concept_vocab,
        moving_average_window=moving_average_window,
        ema_alpha=ema_alpha,
    )
    return round(_sigmoid(_dot(model.weights, features)), 12)


def _prefix_feature_classifier_features(
    record: TraceRecord,
    concept_vocab: Sequence[int | str],
    moving_average_window: int,
    ema_alpha: float,
) -> list[float]:
    scores, codes = _paired_scores_codes(record)
    return (
        _prefix_score_features(scores, moving_average_window=moving_average_window, ema_alpha=ema_alpha)
        + _prefix_concept_features(codes, concept_vocab)
    )


def _prefix_score_features(
    scores: Sequence[float],
    moving_average_window: int,
    ema_alpha: float,
) -> list[float]:
    if not scores:
        raise ValueError("scores must not be empty")
    values = [float(score) for score in scores]
    final = values[-1]
    ema = exponential_moving_average(values, ema_alpha)[-1]
    ma = moving_average(values, moving_average_window)[-1]
    mean_score = sum(values) / len(values)
    delta = final - values[0] if len(values) > 1 else 0.0
    norm_len = min(len(values) / 100.0, 1.0)
    return [1.0, final, ema, ma, mean_score, delta, norm_len]


def _prefix_concept_features(
    codes: Sequence[int | str],
    concept_vocab: Sequence[int | str],
) -> list[float]:
    if not codes:
        raise ValueError("concept codes must not be empty")
    vocab_index = {code: index for index, code in enumerate(concept_vocab)}
    unknown_index = len(concept_vocab)
    final_features = [0.0 for _ in range(len(concept_vocab) + 1)]
    final_features[vocab_index.get(codes[-1], unknown_index)] = 1.0

    count_features = [0.0 for _ in range(len(concept_vocab) + 1)]
    for code in codes:
        count_features[vocab_index.get(code, unknown_index)] += 1.0 / len(codes)
    transition_rate = 0.0
    if len(codes) > 1:
        transition_rate = sum(1 for left, right in zip(codes, codes[1:]) if left != right) / (len(codes) - 1)
    return final_features + count_features + [transition_rate]


def _concept_vocab(records: Sequence[TraceRecord]) -> tuple[int | str, ...]:
    codes = {code for record in records for code in _codes(record)}
    return tuple(sorted(codes, key=lambda value: (type(value).__name__, str(value))))


def _fit_logistic_weights(
    features: Sequence[Sequence[float]],
    labels: Sequence[float],
    learning_rate: float,
    l2_penalty: float,
    iterations: int,
) -> list[float]:
    if not features:
        raise ValueError("features must not be empty")
    base_rate = min(max(sum(labels) / len(labels), 1e-6), 1.0 - 1e-6)
    weights = [log(base_rate / (1.0 - base_rate))] + [0.0 for _ in range(len(features[0]) - 1)]
    if len(set(labels)) < 2:
        return weights

    for _ in range(int(iterations)):
        gradients = [0.0 for _ in weights]
        for row, label in zip(features, labels):
            prediction = _sigmoid(_dot(weights, row))
            error = prediction - label
            for index, value in enumerate(row):
                gradients[index] += error * value
        n = float(len(features))
        for index in range(len(weights)):
            regularization = 0.0 if index == 0 else l2_penalty * weights[index]
            weights[index] -= learning_rate * ((gradients[index] / n) + regularization)
    return weights


def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(float(weight) * float(value) for weight, value in zip(weights, features))


def _sigmoid(value: float) -> float:
    if value >= 0:
        scale = exp(-value)
        return 1.0 / (1.0 + scale)
    scale = exp(value)
    return scale / (1.0 + scale)


def _fraction_index(length: int, fraction: float) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    return int((length - 1) * fraction + 0.5)


def _fraction_key(fraction: float) -> str:
    return f"p{int(round(fraction * 100)):02d}"


def _subtract_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 12)


def _base_rate(records: Sequence[TraceRecord]) -> float:
    return sum(1 for record in records if record.correct) / len(records)


def _stationary_config(config: HMMConfig | NonStationaryHMMConfig) -> HMMConfig:
    if isinstance(config, HMMConfig):
        return config
    return HMMConfig(
        p_error=config.p_error_base,
        p_recover=config.p_recover_base,
        initial_on_track=config.initial_on_track,
    )


def _default_nonstationary_config(
    config: HMMConfig | NonStationaryHMMConfig,
) -> NonStationaryHMMConfig:
    if isinstance(config, NonStationaryHMMConfig):
        return config
    return NonStationaryHMMConfig(
        p_error_base=max(0.0, config.p_error * 0.6),
        p_error_slope=max(0.001, config.p_error * 0.08),
        p_recover_base=min(1.0, config.p_recover * 1.4),
        p_recover_decay=max(0.001, config.p_recover * 0.04),
        initial_on_track=config.initial_on_track,
    )


def _optional_metric(
    metric: Callable[[Sequence[int], Sequence[float]], float],
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> float | None:
    try:
        return metric(labels, probabilities)
    except ValueError:
        return None
