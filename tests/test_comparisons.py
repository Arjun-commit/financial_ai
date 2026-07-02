"""Tests for agents/comparisons.py - monthly trends and YoY comparison."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from fin_flow.agents.comparisons import monthly_summary, yoy_comparison


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal transaction DataFrame from dicts."""
    df = pd.DataFrame(rows)
    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    if "raw_hash" not in df.columns:
        df["raw_hash"] = [f"h{i}" for i in range(len(df))]
    return df


# ── monthly_summary ─────────────────────────────────────────────────────


def test_monthly_summary_basic():
    df = _make_df([
        {"transaction_date": "2024-01-15", "amount": Decimal("500.00"), "category": "Income"},
        {"transaction_date": "2024-01-20", "amount": Decimal("-100.00"), "category": "Meals"},
        {"transaction_date": "2024-02-10", "amount": Decimal("600.00"), "category": "Income"},
        {"transaction_date": "2024-02-15", "amount": Decimal("-200.00"), "category": "Rent"},
    ])
    result = monthly_summary(df)
    assert len(result) == 2
    assert list(result.columns) == ["year", "month", "income", "expenses", "net", "txn_count"]

    jan = result[result["month"] == 1].iloc[0]
    assert jan["income"] == 500.0
    assert jan["expenses"] == 100.0
    assert jan["net"] == 400.0
    assert jan["txn_count"] == 2

    feb = result[result["month"] == 2].iloc[0]
    assert feb["income"] == 600.0
    assert feb["expenses"] == 200.0


def test_monthly_summary_empty():
    result = monthly_summary(pd.DataFrame())
    assert result.empty
    assert "year" in result.columns


def test_monthly_summary_sorted():
    df = _make_df([
        {"transaction_date": "2024-03-01", "amount": Decimal("100.00")},
        {"transaction_date": "2023-12-01", "amount": Decimal("-50.00")},
        {"transaction_date": "2024-01-01", "amount": Decimal("200.00")},
    ])
    result = monthly_summary(df)
    years_months = list(zip(result["year"], result["month"]))
    assert years_months == [(2023, 12), (2024, 1), (2024, 3)]


# ── yoy_comparison ──────────────────────────────────────────────────────

def _two_year_df() -> pd.DataFrame:
    """Synthetic 2-year data: Jan-Jun in both 2023 and 2024."""
    rows = []
    for year in (2023, 2024):
        for month in range(1, 7):
            # Income
            inc = 5000.0 if year == 2023 else 6000.0
            rows.append({
                "transaction_date": f"{year}-{month:02d}-05",
                "amount": Decimal(str(inc)),
                "category": "Income",
            })
            # Rent
            rows.append({
                "transaction_date": f"{year}-{month:02d}-01",
                "amount": Decimal("-1500.00"),
                "category": "Rent",
            })
            # Meals (increases in 2024)
            meals = -200.0 if year == 2023 else -350.0
            rows.append({
                "transaction_date": f"{year}-{month:02d}-15",
                "amount": Decimal(str(meals)),
                "category": "Meals",
            })
    return _make_df(rows)


def test_yoy_has_prior_year():
    yoy = yoy_comparison(_two_year_df())
    assert yoy.has_prior_year is True
    assert yoy.current_year == 2024
    assert yoy.prior_year == 2023


def test_yoy_no_prior_year_single_year():
    df = _make_df([
        {"transaction_date": "2024-01-01", "amount": Decimal("100.00")},
        {"transaction_date": "2024-06-01", "amount": Decimal("-50.00")},
    ])
    yoy = yoy_comparison(df)
    assert yoy.has_prior_year is False


def test_yoy_empty_df():
    yoy = yoy_comparison(pd.DataFrame())
    assert yoy.has_prior_year is False


def test_yoy_ytd_label_partial_year():
    """When current year has Jan-Jun, YTD label reflects those months."""
    yoy = yoy_comparison(_two_year_df())
    assert "Jan" in yoy.ytd_label
    assert "Jun" in yoy.ytd_label
    assert "YTD" in yoy.ytd_label


def test_yoy_overlapping_months_only():
    """YoY only compares overlapping months (no partial vs full year)."""
    rows = []
    # 2023: full year (Jan-Dec)
    for month in range(1, 13):
        rows.append({
            "transaction_date": f"2023-{month:02d}-01",
            "amount": Decimal("1000.00"),
            "category": "Income",
        })
    # 2024: Jan-Mar only
    for month in range(1, 4):
        rows.append({
            "transaction_date": f"2024-{month:02d}-01",
            "amount": Decimal("1200.00"),
            "category": "Income",
        })
    df = _make_df(rows)
    yoy = yoy_comparison(df)
    assert yoy.has_prior_year is True
    # Monthly should only have 3 rows (Jan, Feb, Mar overlap)
    assert len(yoy.monthly) == 3
    assert set(yoy.monthly["month"].tolist()) == {1, 2, 3}
    # YTD label should be Jan-Mar
    assert "Jan" in yoy.ytd_label
    assert "Mar" in yoy.ytd_label


def test_yoy_income_delta():
    yoy = yoy_comparison(_two_year_df())
    # 2023: 6 months * 5000 = 30000 income
    # 2024: 6 months * 6000 = 36000 income
    assert yoy.ytd_prior_income == 30000.0
    assert yoy.ytd_current_income == 36000.0
    assert yoy.ytd_income_delta == 6000.0
    assert yoy.ytd_income_delta_pct == pytest.approx(20.0, abs=0.1)


def test_yoy_expenses_delta():
    yoy = yoy_comparison(_two_year_df())
    # 2023: 6*(1500+200) = 10200 expenses
    # 2024: 6*(1500+350) = 11100 expenses
    assert yoy.ytd_prior_expenses == 10200.0
    assert yoy.ytd_current_expenses == 11100.0
    assert yoy.ytd_expenses_delta == 900.0


def test_yoy_top_category_changes():
    yoy = yoy_comparison(_two_year_df())
    assert len(yoy.top_category_changes) > 0
    # Meals should be in top changes: went from 1200 to 2100
    cats = {ch["category"] for ch in yoy.top_category_changes}
    assert "Meals" in cats
    meals_ch = [ch for ch in yoy.top_category_changes if ch["category"] == "Meals"][0]
    assert meals_ch["prior_amount"] == 1200.0
    assert meals_ch["current_amount"] == 2100.0
    assert meals_ch["delta"] == 900.0


def test_yoy_monthly_breakdown():
    yoy = yoy_comparison(_two_year_df())
    assert not yoy.monthly.empty
    # Each overlapping month should have columns
    assert "month_name" in yoy.monthly.columns
    assert "current_income" in yoy.monthly.columns
    assert "prior_income" in yoy.monthly.columns
    assert "income_delta" in yoy.monthly.columns


def test_yoy_no_overlap():
    """No overlapping months → has_prior_year=False."""
    rows = [
        {"transaction_date": "2023-01-01", "amount": Decimal("100.00")},
        {"transaction_date": "2024-07-01", "amount": Decimal("100.00")},
    ]
    df = _make_df(rows)
    # 2023 has Jan, 2024 has Jul → no overlap
    yoy = yoy_comparison(df)
    assert yoy.has_prior_year is False
