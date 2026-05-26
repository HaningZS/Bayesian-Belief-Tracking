"""Deterministic placeholder concept coding for generated text chunks."""

from __future__ import annotations

import hashlib


def stable_concept_code(text: str, buckets: int = 64) -> int:
    """Map text to a stable integer bucket for pipeline plumbing."""

    if buckets <= 0:
        raise ValueError("buckets must be positive")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


REASONING_PATTERNS: dict[str, list[str]] = {
    "setup": ["let ", "define ", "suppose ", "given ", "assume "],
    "calculation": ["calculate", "compute", " = ", "multiply", "divide", "subtract", "add "],
    "verification": ["check", "verify", "confirm", "indeed", "correct"],
    "correction": ["wait", "actually", "mistake", "oops", "no,", "wrong"],
    "conclusion": ["therefore", "thus", "hence", "the answer", "final answer"],
    "exploration": ["alternatively", "another way", "what if", "try ", "consider"],
}


SELF_VERIFICATION_PATTERNS: dict[str, list[str]] = {
    "sv_correction": [
        "wait",
        "actually",
        "mistake",
        "wrong",
        "fix ",
        "no,",
        "oops",
        "reconsider",
    ],
    "sv_verification": [
        "check",
        "double-check",
        "verify",
        "confirm",
        "make sure",
        "to be sure",
        "sanity check",
    ],
    "sv_alternative": [
        "alternatively",
        "another way",
        "different way",
        "different approach",
        "from another angle",
        "try another",
    ],
    "sv_uncertainty": [
        "maybe",
        "perhaps",
        "not sure",
        "unclear",
        "might be",
        "could be",
    ],
}


def text_concept_code(text: str) -> str:
    """Classify a reasoning step by its dominant text pattern.

    Returns the pattern name with the most keyword matches,
    or 'other' if no pattern matches.
    """
    lower = text.lower()
    best_pattern = "other"
    best_count = 0
    for pattern_name, keywords in REASONING_PATTERNS.items():
        count = sum(1 for keyword in keywords if keyword in lower)
        if count > best_count:
            best_count = count
            best_pattern = pattern_name
    return best_pattern


def self_verification_concept_code(text: str) -> str:
    """Classify explicit self-verification language in a reasoning step.

    This is intentionally marker-based and interpretable. It is an observation
    variant for diagnostics, not a learned self-verification model.
    """

    lower = text.lower()
    best_pattern = "sv_none"
    best_count = 0
    for pattern_name, keywords in SELF_VERIFICATION_PATTERNS.items():
        count = sum(1 for keyword in keywords if keyword in lower)
        if count > best_count:
            best_count = count
            best_pattern = pattern_name
    return best_pattern
