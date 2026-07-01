"""Normalize heterogeneous bank exports into the canonical schema."""

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
    # ── Date ────────────────────────────────────────────────────────────
    "date": "transaction_date",
    "transaction date": "transaction_date",
    "posting date": "transaction_date",
    "posted date": "transaction_date",       # Capital One
    "post date": "transaction_date",
    "trans date": "transaction_date",
    "transaction_date": "transaction_date",
    # ── Amount (single-column) ──────────────────────────────────────────
    "amount": "amount",
    "amt": "amount",
    "transaction amount": "amount",
    # ── Debit / credit split (merged into amount below) ─────────────────
    "debit": "_debit",
    "credit": "_credit",
    "withdrawal": "_debit",
    "withdrawals": "_debit",
    "deposit": "_credit",
    "deposits": "_credit",
    # ── Description ─────────────────────────────────────────────────────
    "description": "description",
    "desc": "description",
    "memo": "description",
    "details": "description",                # Capital One
    "narration": "description",
    "payee": "description",
    "original description": "description",   # Mint / some aggregators
    # ── Columns to ignore (mapped to _drop → silently skipped) ──────────
    "balance": "_drop",
    "running bal.": "_drop",
    "running balance": "_drop",
    "card no.": "_drop",                     # Capital One
    "card number": "_drop",
    "category": "_drop",                     # bank-provided categories
    "type": "_drop",                         # Chase "type" column
    "check or slip #": "_drop",              # Wells Fargo
    "reference number": "_drop",             # Citi
    "member name": "_drop",                  # Amex
    "account #": "_drop",
    "extended details": "_drop",             # Amex
    "appears on your statement as": "_drop", # Amex
    "address": "_drop",
    "city/state": "_drop",
    "zip code": "_drop",
    "country": "_drop",
}


class IngestionError(ValueError):
    """Raised when a file cannot be normalized."""


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = _rename_columns(df)
    df = df.drop(columns=[c for c in df.columns if c == "_drop"], errors="ignore")
    df = _merge_debit_credit(df)

    required = {"transaction_date", "amount", "description"}
    missing = required - set(df.columns)
    if missing:
        raise IngestionError(
            "We couldn't read this file format. We support CSV exports "
            "from Chase, Bank of America, Wells Fargo, Citi, Capital One, "
            "and American Express. Make sure you're uploading the "
            "transaction export (not a statement PDF). Need help? The "
            "'Download Activity' or 'Export Transactions' option in your "
            "bank's website usually gives the right format."
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

    before = len(out)
    out = out.dropna(subset=["transaction_date", "amount", "description"])
    out = out[out["description"] != ""]

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
    elif suffix == ".pdf":
        from .pdf_parser import parse_pdf
        df = parse_pdf(p)
    else:
        raise IngestionError(f"Unsupported file type: {suffix}")

    return normalize_dataframe(df, source=src)
