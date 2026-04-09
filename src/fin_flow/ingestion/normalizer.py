"""Normalize heterogeneous bank exports into the canonical schema.

Different banks export CSVs with different column names, date formats, and
sign conventions. This module attempts to auto-detect common layouts and
produce a clean DataFrame that matches `CANONICAL_COLUMNS`.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import pandas as pd

from .dedupe import content_hash
from .schema import CANONICAL_COLUMNS

# Lowercased candidate column names -> canonical name
COLUMN_ALIASES: dict[str, str] = {
    # date
    "date": "transaction_date",
    "transaction date": "transaction_date",
    "posting date": "transaction_date",
    "post date": "transaction_date",
    "trans date": "transaction_date",
    "transaction_date": "transaction_date",
    # amount
    "amount": "amount",
    "amt": "amount",
    "transaction amount": "amount",
    # debit / credit split (handled specially below)
    "debit": "_debit",
    "credit": "_credit",
    "withdrawal": "_debit",
    "deposit": "_credit",
    # description
    "description": "description",
    "desc": "description",
    "memo": "description",
    "details": "description",
    "narration": "description",
    "payee": "description",
}


class IngestionError(ValueError):
    """Raised when a file cannot be normalized."""


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map known aliases to canonical names.

    If multiple source columns alias to the same canonical name (e.g. both
    "Transaction Date" and "Post Date" -> transaction_date), keep only the
    first match and drop the rest to avoid duplicate-key errors downstream.
    """
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    drop: list[str] = []
    for col in df.columns:
        key = str(col).strip().lower()
        if key in COLUMN_ALIASES:
            canonical = COLUMN_ALIASES[key]
            if canonical in taken:
                drop.append(col)
            else:
                mapping[col] = canonical
                taken.add(canonical)
    df = df.drop(columns=drop) if drop else df
    return df.rename(columns=mapping)


def _coerce_amount(value) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            d = Decimal(str(value))
        except InvalidOperation:
            return None
        if d.is_nan():
            return None
        return d
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    # Strip currency symbols, thousands separators, and handle parentheses
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    for ch in ("$", "€", "£", ",", " "):
        s = s.replace(ch, "")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    return -d if negative else d


def _merge_debit_credit(df: pd.DataFrame) -> pd.DataFrame:
    """If a file has separate debit/credit columns, collapse into `amount`."""
    has_debit = "_debit" in df.columns
    has_credit = "_credit" in df.columns
    if not (has_debit or has_credit):
        return df

    def _row_amount(row) -> Optional[Decimal]:
        debit = _coerce_amount(row.get("_debit")) if has_debit else None
        credit = _coerce_amount(row.get("_credit")) if has_credit else None
        if debit is not None and debit != 0:
            return -abs(debit)  # debits are expenses
        if credit is not None and credit != 0:
            return abs(credit)  # credits are income
        return None

    df = df.copy()
    df["amount"] = df.apply(_row_amount, axis=1)
    return df.drop(columns=[c for c in ("_debit", "_credit") if c in df.columns])


def normalize_dataframe(df: pd.DataFrame, source: str = "unknown") -> pd.DataFrame:
    """Convert a raw bank DataFrame into the canonical Fin-Flow schema.

    Returns a DataFrame with exactly `CANONICAL_COLUMNS`. Rows that cannot
    be parsed (missing date or amount) are dropped.
    """
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = _rename_columns(df)
    df = _merge_debit_credit(df)

    required = {"transaction_date", "amount", "description"}
    missing = required - set(df.columns)
    if missing:
        raise IngestionError(
            f"Could not locate required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["transaction_date"] = pd.to_datetime(
        df["transaction_date"], errors="coerce"
    ).dt.date
    out["amount"] = df["amount"].map(_coerce_amount)
    out["description"] = df["description"].astype(str).str.strip()
    out["source"] = source
    out["category"] = None
    out["ai_confidence_score"] = None

    # Drop rows we couldn't parse
    before = len(out)
    out = out.dropna(subset=["transaction_date", "amount", "description"])
    out = out[out["description"] != ""]
    dropped = before - len(out)
    if dropped:
        # Not an error — many bank CSVs contain header/footer summary rows.
        pass

    # Hash the canonical tuple for dedupe
    out["raw_hash"] = out.apply(
        lambda r: content_hash(r["transaction_date"], r["amount"], r["description"]),
        axis=1,
    )

    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def load_file(path: str | Path, source: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV, Excel, or JSON bank export and normalize it."""
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"File not found: {p}")

    src = source or p.stem
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    elif suffix == ".json":
        with p.open() as f:
            data = json.load(f)
        if isinstance(data, dict) and "transactions" in data:
            data = data["transactions"]
        df = pd.DataFrame(data)
    else:
        raise IngestionError(f"Unsupported file type: {suffix}")

    return normalize_dataframe(df, source=src)
