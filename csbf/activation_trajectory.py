"""Activation-trajectory diagnostics built on hidden-state dynamics.

This module keeps the paper-facing diagnostic name explicit while reusing the
existing prefix-safe latent-trajectory implementation. It does not collect
attention tensors or new activations; callers must pass hidden feature JSONL
rows already produced by the local hidden extractor.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from csbf.latent_trajectory import (
    LatentTrajectoryConfig,
    build_latent_trajectory_records,
    compute_trajectory_metrics,
    run_latent_trajectory_split_seed_sweep,
    write_latent_trajectory_split_seed_sweep,
)

ActivationTrajectoryConfig = LatentTrajectoryConfig


def build_activation_trajectory_records(
    records_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    seed: int = 0,
    config: ActivationTrajectoryConfig | None = None,
) -> dict[str, Any]:
    """Rewrite records with activation-trajectory scores."""

    summary = build_latent_trajectory_records(
        records_path,
        features_path,
        output_path=output_path,
        report_path=report_path,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        seed=seed,
        config=config,
    )
    return _annotate_summary(summary)


def run_activation_trajectory_split_seed_sweep(
    records_path: str | Path,
    features_path: str | Path,
    seeds: Iterable[int],
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    config: ActivationTrajectoryConfig | None = None,
    use_model_selection: bool = False,
    calibration_mode: str = "all_steps",
) -> dict[str, Any]:
    """Run activation-trajectory split-seed diagnostics."""

    summary = run_latent_trajectory_split_seed_sweep(
        records_path,
        features_path,
        seeds=seeds,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        config=config,
        use_model_selection=use_model_selection,
        calibration_mode=calibration_mode,
    )
    return _annotate_summary(summary)


def write_activation_trajectory_split_seed_sweep(
    summary: dict[str, Any],
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, str]:
    """Write activation-trajectory split-seed robustness outputs."""

    return write_latent_trajectory_split_seed_sweep(summary, output_json, output_csv)


def _annotate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(summary)
    annotated["diagnostic_family"] = "activation_trajectory"
    annotated["base_method"] = "latent_trajectory"
    return annotated


__all__ = [
    "ActivationTrajectoryConfig",
    "build_activation_trajectory_records",
    "compute_trajectory_metrics",
    "run_activation_trajectory_split_seed_sweep",
    "write_activation_trajectory_split_seed_sweep",
]
