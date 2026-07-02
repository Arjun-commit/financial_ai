"""Tax-Ready Report - Schedule C mapping, PDF/CSV export."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import pandas as pd


def _to_float(x) -> float:
    if isinstance(x, Decimal):
        return float(x)
    return float(x)


# ── Schedule C line mapping ─────────────────────────────────────────────

SCHEDULE_C_MAPPING: dict[str, str] = {
    "Advertising": "Line 8",
    "Insurance": "Line 15",
    "Professional Services": "Line 17",
    "Office Supplies": "Line 18",
    "Rent": "Line 20b",
    "Repairs & Maintenance": "Line 21",
    "Taxes & Fees": "Line 23",
    "Travel": "Line 24a",
    "Meals": "Line 24b",
    "Utilities": "Line 25",
    "Payroll": "Line 26",
    "Bank Fees": "Line 27a (Other expenses)",
    "Software & Subscriptions": "Line 27a (Other expenses)",
}

EXCLUDED_CATEGORIES: dict[str, str] = {
    "Transfers": "Excluded to avoid double-counting between accounts",
    "Groceries": "Typically personal review for legitimate business expenses",
    "Entertainment": "Typically personal review for legitimate business expenses",
    "Shopping": "Typically personal review for legitimate business expenses",
    "Uncategorized": "Typically personal review for legitimate business expenses",
}

TAX_DISCLAIMER = (
    "This is an informational summary generated from your uploaded "
    "transactions, not tax advice. Mappings are suggestions; deductibility "
    "depends on your situation. Review with a qualified tax professional "
    "before filing."
)

MEALS_NOTE = (
    "Generally 50% deductible - confirm with your tax professional. "
    "Gross amount shown."
)


# ── Data class ──────────────────────────────────────────────────────────


@dataclass
class TaxReport:
    year: int
    income_total: float
    expense_lines: list[dict] = field(default_factory=list)
    expense_total: float = 0.0
    excluded: list[dict] = field(default_factory=list)
    excluded_total: float = 0.0
    disclaimer: str = TAX_DISCLAIMER
    transaction_count: int = 0


# ── Build ───────────────────────────────────────────────────────────────


def build_tax_report(df: pd.DataFrame, year: int) -> TaxReport:
    """Build a tax report for the given calendar year."""
    if df.empty:
        return TaxReport(year=year, income_total=0.0)

    work = df.copy()
    work["_date"] = pd.to_datetime(work["transaction_date"])
    work = work[work["_date"].dt.year == year]

    if work.empty:
        return TaxReport(year=year, income_total=0.0)

    work["_amt"] = work["amount"].map(_to_float)
    txn_count = len(work)

    # Income
    income_total = round(float(work[work["_amt"] > 0]["_amt"].sum()), 2)

    # Deductible expense lines
    expense_lines: list[dict] = []
    for cat, line in SCHEDULE_C_MAPPING.items():
        subset = work[(work["category"] == cat) & (work["_amt"] < 0)]
        if subset.empty:
            continue
        amount = round(float(subset["_amt"].abs().sum()), 2)
        note = MEALS_NOTE if cat == "Meals" else ""
        expense_lines.append({
            "category": cat,
            "schedule_c_line": line,
            "amount": amount,
            "note": note,
        })
    expense_lines.sort(key=lambda x: x["amount"], reverse=True)
    expense_total = round(sum(e["amount"] for e in expense_lines), 2)

    # Excluded categories
    excluded: list[dict] = []
    for cat, reason in EXCLUDED_CATEGORIES.items():
        subset = work[(work["category"] == cat) & (work["_amt"] < 0)]
        if subset.empty:
            continue
        amount = round(float(subset["_amt"].abs().sum()), 2)
        excluded.append({
            "category": cat,
            "amount": amount,
            "reason": reason,
        })
    excluded.sort(key=lambda x: x["amount"], reverse=True)
    excluded_total = round(sum(e["amount"] for e in excluded), 2)

    return TaxReport(
        year=year,
        income_total=income_total,
        expense_lines=expense_lines,
        expense_total=expense_total,
        excluded=excluded,
        excluded_total=excluded_total,
        disclaimer=TAX_DISCLAIMER,
        transaction_count=txn_count,
    )


# ── CSV export ──────────────────────────────────────────────────────────


def tax_report_to_csv_rows(report: TaxReport, df: pd.DataFrame) -> list[dict]:
    """Return transaction detail for the year with schedule_c_line column."""
    work = df.copy()
    work["_date"] = pd.to_datetime(work["transaction_date"])
    work = work[work["_date"].dt.year == report.year]

    rows: list[dict] = []
    for _, r in work.iterrows():
        cat = str(r.get("category", "Uncategorized"))
        if cat in SCHEDULE_C_MAPPING:
            sc_line = SCHEDULE_C_MAPPING[cat]
        elif cat in EXCLUDED_CATEGORIES:
            sc_line = "Excluded"
        elif cat == "Income":
            sc_line = "Income"
        else:
            sc_line = ""
        rows.append({
            "date": str(r["transaction_date"]),
            "description": str(r["description"]),
            "amount": round(_to_float(r["amount"]), 2),
            "category": cat,
            "schedule_c_line": sc_line,
        })
    return rows


# ── PDF export ──────────────────────────────────────────────────────────


def _pdf_safe(text: str) -> str:
    """Replace characters that fpdf2 core fonts can't encode."""
    return (
        text
        .replace("-", "--")   # em dash
        .replace("–", "-")    # en dash
        .replace("‘", "'")    # left single quote
        .replace("’", "'")    # right single quote
        .replace("“", '"')    # left double quote
        .replace("”", '"')    # right double quote
    )


def generate_tax_pdf(report: TaxReport) -> bytes:
    """Generate a one-page Schedule C summary PDF. Returns raw bytes."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        0, 10, _pdf_safe(f"Schedule C Expense Summary -- {report.year}"),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C",
    )
    pdf.ln(5)

    # Business Income
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Business Income", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(
        0, 7, f"  Total Income: ${report.income_total:,.2f}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(3)

    # Deductible Expenses table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0, 8, "Business Expenses by Schedule C Line",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    if report.expense_lines:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 7, "Category", border=1)
        pdf.cell(50, 7, "Schedule C Line", border=1)
        pdf.cell(40, 7, "Amount", border=1, align="R")
        pdf.ln()

        pdf.set_font("Helvetica", "", 10)
        for line in report.expense_lines:
            pdf.cell(70, 7, _pdf_safe(line["category"]), border=1)
            pdf.cell(50, 7, line["schedule_c_line"], border=1)
            pdf.cell(40, 7, f"${line['amount']:,.2f}", border=1, align="R")
            pdf.ln()
            if line["note"]:
                pdf.set_font("Helvetica", "I", 8)
                pdf.cell(
                    0, 5, _pdf_safe(f"    {line['note']}"),
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                )
                pdf.set_font("Helvetica", "", 10)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(120, 7, "Total Deductible Expenses", border=1)
        pdf.cell(40, 7, f"${report.expense_total:,.2f}", border=1, align="R")
        pdf.ln()
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0, 7, "  No deductible expenses found.",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

    pdf.ln(5)

    # Excluded categories
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0, 8, "Excluded from Schedule C",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    if report.excluded:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(50, 7, "Category", border=1)
        pdf.cell(35, 7, "Amount", border=1, align="R")
        pdf.cell(75, 7, "Reason", border=1)
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for item in report.excluded:
            pdf.cell(50, 7, _pdf_safe(item["category"]), border=1)
            pdf.cell(35, 7, f"${item['amount']:,.2f}", border=1, align="R")
            pdf.cell(75, 7, _pdf_safe(item["reason"][:45]), border=1)
            pdf.ln()

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(50, 7, "Total Excluded", border=1)
        pdf.cell(35, 7, f"${report.excluded_total:,.2f}", border=1, align="R")
        pdf.cell(75, 7, "", border=1)
        pdf.ln()

    pdf.ln(8)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _pdf_safe(report.disclaimer))

    return bytes(pdf.output())
