"""PII masking for transaction descriptions."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

# Order matters: more specific patterns first.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email
    (
        re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE),
        "[EMAIL]",
    ),
    # SSN 123-45-6789
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    # Credit card-ish (13-19 digits, optional separators)
    (
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        "[CARD]",
    ),
    # Phone numbers (loose US/international)
    (
        re.compile(
            r"(?:(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4})"
        ),
        "[PHONE]",
    ),
    # Account-number-ish: 8+ consecutive digits
    (re.compile(r"\b\d{8,}\b"), "[ACCT]"),
]


def mask_pii(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    for pattern, token in _PATTERNS:
        s = pattern.sub(token, s)
    return s


def mask_series(values: Iterable[str]) -> pd.Series:
    return pd.Series([mask_pii(v) for v in values])
