"""Content-hash based deduplication."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")


def _canon_description(desc: Any) -> str:
    s = str(desc or "").strip().lower()
    s = _WHITESPACE_RE.sub(" ", s)
    return s


def _canon_date(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _canon_amount(value: Any) -> str:
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    try:
        return format(Decimal(str(value)).normalize(), "f")
    except Exception:
        return str(value)


def content_hash(transaction_date: Any, amount: Any, description: Any) -> str:
    payload = "|".join(
        [
            _canon_date(transaction_date),
            _canon_amount(amount),
            _canon_description(description),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "raw_hash" not in df.columns:
        return df.reset_index(drop=True)
    return df.drop_duplicates(subset="raw_hash", keep="first").reset_index(drop=True)
