"""Tests for reports/tax_report.py - Schedule C mapping and export."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from fin_flow.reports.tax_report import (
    TaxReport,
    build_tax_report,
    tax_report_to_csv_rows,
    generate_tax_pdf,
    SCHEDULE_C_MAPPING,
    EXCLUDED_CATEGORIES,
    TAX_DISCLAIMER,
    MEALS_NOTE,
)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "description" not in df.columns:
        df["description"] = [f"Transaction {i}" for i in range(len(df))]
    if "raw_hash" not in df.columns:
        df["raw_hash"] = [f"h{i}" for i in range(len(df))]
    if "ai_confidence_score" not in df.columns:
        df["ai_confidence_score"] = 0.9
    return df


def _sample_year_df() -> pd.DataFrame:
    """Realistic mix of categories for 2024."""
    return _make_df([
        {"transaction_date": "2024-01-05", "amount": Decimal("8000.00"), "category": "Income"},
        {"transaction_date": "2024-01-01", "amount": Decimal("-1850.00"), "category": "Rent"},
        {"transaction_date": "2024-01-10", "amount": Decimal("-45.50"), "category": "Meals"},
        {"transaction_date": "2024-01-12", "amount": Decimal("-120.00"), "category": "Advertising"},
        {"transaction_date": "2024-01-15", "amount": Decimal("-500.00"), "category": "Transfers"},
        {"transaction_date": "2024-01-18", "amount": Decimal("-87.33"), "category": "Groceries"},
        {"transaction_date": "2024-01-20", "amount": Decimal("-42.18"), "category": "Shopping"},
        {"transaction_date": "2024-01-22", "amount": Decimal("-150.00"), "category": "Software & Subscriptions"},
        {"transaction_date": "2024-01-25", "amount": Decimal("-200.00"), "category": "Professional Services"},
        {"transaction_date": "2024-01-28", "amount": Decimal("-75.00"), "category": "Repairs & Maintenance"},
        {"transaction_date": "2024-02-05", "amount": Decimal("8000.00"), "category": "Income"},
        {"transaction_date": "2024-02-15", "amount": Decimal("-30.00"), "category": "Entertainment"},
        {"transaction_date": "2024-02-20", "amount": Decimal("-15.00"), "category": "Bank Fees"},
        {"transaction_date": "2024-02-25", "amount": Decimal("-999.00"), "category": "Uncategorized"},
    ])


# ── build_tax_report ────────────────────────────────────────────────────


def test_build_report_income():
    report = build_tax_report(_sample_year_df(), 2024)
    assert report.income_total == 16000.0


def test_build_report_expense_lines():
    report = build_tax_report(_sample_year_df(), 2024)
    cats = {e["category"] for e in report.expense_lines}
    # Deductible categories present in the data
    assert "Rent" in cats
    assert "Meals" in cats
    assert "Advertising" in cats
    assert "Software & Subscriptions" in cats
    assert "Professional Services" in cats
    assert "Repairs & Maintenance" in cats
    assert "Bank Fees" in cats


def test_transfers_excluded():
    """Transfers must be in excluded, not expense_lines."""
    report = build_tax_report(_sample_year_df(), 2024)
    expense_cats = {e["category"] for e in report.expense_lines}
    excluded_cats = {e["category"] for e in report.excluded}
    assert "Transfers" not in expense_cats
    assert "Transfers" in excluded_cats


def test_groceries_excluded():
    report = build_tax_report(_sample_year_df(), 2024)
    excluded_cats = {e["category"] for e in report.excluded}
    assert "Groceries" in excluded_cats


def test_entertainment_excluded():
    report = build_tax_report(_sample_year_df(), 2024)
    excluded_cats = {e["category"] for e in report.excluded}
    assert "Entertainment" in excluded_cats


def test_shopping_excluded():
    report = build_tax_report(_sample_year_df(), 2024)
    excluded_cats = {e["category"] for e in report.excluded}
    assert "Shopping" in excluded_cats


def test_uncategorized_excluded():
    report = build_tax_report(_sample_year_df(), 2024)
    excluded_cats = {e["category"] for e in report.excluded}
    assert "Uncategorized" in excluded_cats


def test_meals_note_present():
    """Meals line should carry the 50% deductibility note."""
    report = build_tax_report(_sample_year_df(), 2024)
    meals_line = [e for e in report.expense_lines if e["category"] == "Meals"]
    assert len(meals_line) == 1
    assert meals_line[0]["note"] == MEALS_NOTE


def test_schedule_c_line_mapping():
    """Each expense line should have the correct Schedule C line."""
    report = build_tax_report(_sample_year_df(), 2024)
    for line in report.expense_lines:
        cat = line["category"]
        assert cat in SCHEDULE_C_MAPPING
        assert line["schedule_c_line"] == SCHEDULE_C_MAPPING[cat]


def test_totals_reconcile():
    """expense_total + excluded_total + income should match sum of all amounts."""
    df = _sample_year_df()
    report = build_tax_report(df, 2024)

    # All negative amounts in 2024
    all_negative = sum(
        abs(float(a)) for a in df["amount"] if float(a) < 0
    )
    # The total should equal deductible + excluded
    assert report.expense_total + report.excluded_total == pytest.approx(all_negative, abs=0.01)


def test_expense_lines_sorted_descending():
    report = build_tax_report(_sample_year_df(), 2024)
    amounts = [e["amount"] for e in report.expense_lines]
    assert amounts == sorted(amounts, reverse=True)


def test_excluded_sorted_descending():
    report = build_tax_report(_sample_year_df(), 2024)
    if report.excluded:
        amounts = [e["amount"] for e in report.excluded]
        assert amounts == sorted(amounts, reverse=True)


def test_disclaimer_present():
    report = build_tax_report(_sample_year_df(), 2024)
    assert report.disclaimer == TAX_DISCLAIMER


def test_empty_year():
    report = build_tax_report(_sample_year_df(), 2020)
    assert report.transaction_count == 0
    assert report.income_total == 0.0


def test_empty_df():
    report = build_tax_report(pd.DataFrame(), 2024)
    assert report.transaction_count == 0


# ── tax_report_to_csv_rows ─────────────────────────────────────────────


def test_csv_rows_have_schedule_c_column():
    df = _sample_year_df()
    report = build_tax_report(df, 2024)
    rows = tax_report_to_csv_rows(report, df)
    assert len(rows) > 0
    for row in rows:
        assert "schedule_c_line" in row
        assert "category" in row
        assert "date" in row
        assert "amount" in row


def test_csv_rows_transfers_marked_excluded():
    df = _sample_year_df()
    report = build_tax_report(df, 2024)
    rows = tax_report_to_csv_rows(report, df)
    transfer_rows = [r for r in rows if r["category"] == "Transfers"]
    for r in transfer_rows:
        assert r["schedule_c_line"] == "Excluded"


def test_csv_rows_income_marked():
    df = _sample_year_df()
    report = build_tax_report(df, 2024)
    rows = tax_report_to_csv_rows(report, df)
    income_rows = [r for r in rows if r["category"] == "Income"]
    for r in income_rows:
        assert r["schedule_c_line"] == "Income"


# ── generate_tax_pdf ───────────────────────────────────────────────────


def test_pdf_returns_bytes():
    report = build_tax_report(_sample_year_df(), 2024)
    try:
        pdf_bytes = generate_tax_pdf(report)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 100
        # PDF signature
        assert pdf_bytes[:5] == b"%PDF-"
    except ImportError:
        pytest.skip("fpdf2 not installed")
