from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from fin_flow.ingestion import CANONICAL_COLUMNS, load_file, normalize_dataframe
from fin_flow.ingestion.normalizer import IngestionError, _coerce_amount

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_coerce_amount_handles_currency_and_parentheses():
    assert _coerce_amount("$1,234.56") == Decimal("1234.56")
    assert _coerce_amount("(42.18)") == Decimal("-42.18")
    assert _coerce_amount("") is None
    assert _coerce_amount("nan") is None
    assert _coerce_amount(-7.5) == Decimal("-7.5")


def test_normalize_chase_csv():
    df = load_file(SAMPLES / "chase_sample.csv")
    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) == 7
    # Arithmetic: sum should match the raw totals
    total = sum(df["amount"])
    assert total == Decimal("2180.33")
    assert df["raw_hash"].nunique() == len(df)


def test_normalize_bofa_debit_credit_split():
    df = load_file(SAMPLES / "bofa_sample.csv")
    assert len(df) == 5
    # One credit (deposit), four debits
    positives = [a for a in df["amount"] if a > 0]
    negatives = [a for a in df["amount"] if a < 0]
    assert len(positives) == 1
    assert len(negatives) == 4
    assert positives[0] == Decimal("4200.00")


def test_normalize_generic_json():
    df = load_file(SAMPLES / "generic_sample.json")
    assert len(df) == 3
    assert sum(df["amount"]) == Decimal("4152.07")


def test_missing_required_columns_raises():
    bad = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(IngestionError):
        normalize_dataframe(bad)


def test_empty_dataframe_returns_empty_canonical():
    out = normalize_dataframe(pd.DataFrame())
    assert list(out.columns) == CANONICAL_COLUMNS
    assert len(out) == 0
