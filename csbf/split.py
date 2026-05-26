"""Question-level dataset splitting."""

from __future__ import annotations

import random
from collections import defaultdict

from csbf.schema import TraceRecord


def split_by_question(
    records: list[TraceRecord],
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    seed: int = 0,
) -> dict[str, list[TraceRecord]]:
    """Split records by question id to avoid trace leakage across splits."""

    if not records:
        raise ValueError("records must not be empty")
    if train_ratio <= 0.0 or calibration_ratio <= 0.0:
        raise ValueError("train_ratio and calibration_ratio must be positive")
    if train_ratio + calibration_ratio >= 1.0:
        raise ValueError("train_ratio + calibration_ratio must be less than 1")

    by_question: dict[str, list[TraceRecord]] = defaultdict(list)
    for record in records:
        record.validate()
        by_question[record.question_id].append(record)

    question_ids = sorted(by_question)
    random.Random(seed).shuffle(question_ids)
    n_questions = len(question_ids)
    if n_questions < 3:
        raise ValueError("at least three question ids are required for train/calibration/test splits")

    train_count = max(1, int(n_questions * train_ratio))
    calibration_count = max(1, int(n_questions * calibration_ratio))
    if train_count + calibration_count >= n_questions:
        calibration_count = max(1, n_questions - train_count - 1)
    if train_count + calibration_count >= n_questions:
        train_count = n_questions - calibration_count - 1

    train_ids = set(question_ids[:train_count])
    calibration_ids = set(question_ids[train_count : train_count + calibration_count])

    splits: dict[str, list[TraceRecord]] = {"train": [], "calibration": [], "test": []}
    for question_id in question_ids:
        if question_id in train_ids:
            split_name = "train"
        elif question_id in calibration_ids:
            split_name = "calibration"
        else:
            split_name = "test"
        splits[split_name].extend(by_question[question_id])
    return splits


def split_by_question_stratified(
    records: list[TraceRecord],
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    seed: int = 0,
) -> dict[str, list[TraceRecord]]:
    """Split records by question id with stratification by correctness.

    Questions are grouped by whether any trace is incorrect (mixed/wrong)
    vs all traces correct. Each stratum is split proportionally.
    """
    if not records:
        raise ValueError("records must not be empty")
    if train_ratio <= 0.0 or calibration_ratio <= 0.0:
        raise ValueError("train_ratio and calibration_ratio must be positive")
    if train_ratio + calibration_ratio >= 1.0:
        raise ValueError("train_ratio + calibration_ratio must be less than 1")

    by_question: dict[str, list[TraceRecord]] = defaultdict(list)
    for record in records:
        record.validate()
        by_question[record.question_id].append(record)

    n_questions = len(by_question)
    if n_questions < 3:
        raise ValueError("at least three question ids are required for train/calibration/test splits")

    all_correct_ids: list[str] = []
    has_incorrect_ids: list[str] = []
    for qid in sorted(by_question):
        if all(r.correct for r in by_question[qid]):
            all_correct_ids.append(qid)
        else:
            has_incorrect_ids.append(qid)

    rng = random.Random(seed)
    rng.shuffle(all_correct_ids)
    rng.shuffle(has_incorrect_ids)

    def _split_ids(ids: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(ids)
        if n == 0:
            return [], [], []
        train_n = max(1, int(n * train_ratio)) if n >= 3 else min(n, 1)
        cal_n = max(1, int(n * calibration_ratio)) if n >= 3 else min(n - train_n, 1) if n > train_n else 0
        if train_n + cal_n >= n:
            cal_n = max(0, n - train_n - 1)
        if train_n + cal_n >= n and train_n > 1:
            train_n = n - cal_n - 1
        return ids[:train_n], ids[train_n:train_n + cal_n], ids[train_n + cal_n:]

    ac_train, ac_cal, ac_test = _split_ids(all_correct_ids)
    hi_train, hi_cal, hi_test = _split_ids(has_incorrect_ids)

    train_ids = set(ac_train + hi_train)
    cal_ids = set(ac_cal + hi_cal)

    splits: dict[str, list[TraceRecord]] = {"train": [], "calibration": [], "test": []}
    for qid in sorted(by_question):
        if qid in train_ids:
            split_name = "train"
        elif qid in cal_ids:
            split_name = "calibration"
        else:
            split_name = "test"
        splits[split_name].extend(by_question[qid])
    return splits
