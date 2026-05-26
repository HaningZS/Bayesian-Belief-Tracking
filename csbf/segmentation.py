"""Trace segmentation helpers for offline experiments."""

from __future__ import annotations

import re


DEFAULT_STEP_MARKERS = (
    "Step",
    "Therefore",
    "Thus",
    "So",
    "Wait",
    "Let me",
    "Check",
    "Verify",
)


def rule_based_steps(trace: str, markers: tuple[str, ...] = DEFAULT_STEP_MARKERS) -> list[str]:
    """Split reasoning text into coarse steps using lines and common markers."""

    stripped = trace.strip()
    if not stripped:
        return []

    line_steps = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(line_steps) > 1:
        return line_steps

    marker_pattern = "|".join(re.escape(marker) for marker in markers)
    pieces = re.split(rf"(?=\b(?:{marker_pattern})\b)", stripped)
    steps = [piece.strip() for piece in pieces if piece.strip()]
    return steps or [stripped]


def fixed_token_chunks(trace: str, chunk_size: int) -> list[str]:
    """Split text into fixed-size whitespace-token chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    tokens = trace.split()
    return [" ".join(tokens[index : index + chunk_size]) for index in range(0, len(tokens), chunk_size)]
