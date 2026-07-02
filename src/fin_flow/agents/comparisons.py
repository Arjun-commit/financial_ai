"""Monthly trends and year-over-year comparisons."""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import pandas as pd


def _to_float(x) -> float:
    if isinstance(x, Decimal):
        return float(x)
    return float(x)


# ── Monthly summary ─────────────────────────────────────────────────────


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions by year/month.

    Returns a DataFrame with columns:
        year, month, income, expenses, net, txn_count
    Sorted ascending by year then month.
    """
    if df.empty:
        return pd.DataFrame(columns=["year", "month", "income", "expenses", "net", "txn_count"])

    work = df.copy()
    work["_date"] = pd.to_datetime(work["transaction_date"])
    work["_year"] = work["_date"].dt.year
    work["_month"] = work["_date"].dt.month
    work["_amt"] = work["amount"].map(_to_float)

    rows: list[dict] = []
    for (yr, mo), grp in work.groupby(["_year", "_month"]):
        amounts = grp["_amt"]
        income = float(amounts[amounts > 0].sum())
        expenses = float(amounts[amounts < 0].abs().sum())
        rows.append({
            "year": int(yr),
            "month": int(mo),
            "income": round(income, 2),
            "expenses": round(expenses, 2),
            "net": round(income - expenses, 2),
            "txn_count": len(grp),
        })

    result = pd.DataFrame(rows)
    return result.sort_values(["year", "month"]).reset_index(drop=True)


# ── Year-over-year comparison ───────────────────────────────────────────


@dataclass
class YoYComparison:
    has_prior_year: bool
    current_year: int = 0
    prior_year: int = 0
    ytd_label: str = ""
    ytd_current_income: float = 0.0
    ytd_prior_income: float = 0.0
    ytd_income_delta: float = 0.0
    ytd_income_delta_pct: Optional[float] = None
    ytd_current_expenses: float = 0.0
    ytd_prior_expenses: float = 0.0
    ytd_expenses_delta: float = 0.0
    ytd_expenses_delta_pct: Optional[float] = None
    monthly: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    top_category_changes: list[dict] = field(default_factory=list)


def yoy_comparison(df: pd.DataFrame) -> YoYComparison:
    """Compare the latest calendar year against the prior year.

    Only overlapping months are compared (partial-year safety).
    Returns has_prior_year=False when the data doesn't span two
    calendar years with at least one overlapping month.
    """
    if df.empty:
        return YoYComparison(has_prior_year=False)

    work = df.copy()
    work["_date"] = pd.to_datetime(work["transaction_date"])
    work["_year"] = work["_date"].dt.year
    work["_month"] = work["_date"].dt.month
    work["_amt"] = work["amount"].map(_to_float)

    years = sorted(work["_year"].unique())
    if len(years) < 2:
        return YoYComparison(has_prior_year=False)

    current_year = int(years[-1])
    prior_year = int(years[-2])

    cur = work[work["_year"] == current_year]
    pri = work[work["_year"] == prior_year]

    cur_months = set(cur["_month"].unique())
    pri_months = set(pri["_month"].unique())
    overlap = sorted(cur_months & pri_months)

    if not overlap:
        return YoYComparison(has_prior_year=False)

    # Filter to overlapping months only
    cur_overlap = cur[cur["_month"].isin(overlap)]
    pri_overlap = pri[pri["_month"].isin(overlap)]

    # YTD label
    month_names = [calendar.month_abbr[m] for m in overlap]
    if len(month_names) == 1:
        ytd_label = f"{month_names[0]} YTD"
    else:
        ytd_label = f"{month_names[0]}-{month_names[-1]} YTD"

    # YTD totals
    def _totals(subset: pd.DataFrame) -> tuple[float, float]:
        amounts = subset["_amt"]
        inc = float(amounts[amounts > 0].sum())
        exp = float(amounts[amounts < 0].abs().sum())
        return round(inc, 2), round(exp, 2)

    cur_inc, cur_exp = _totals(cur_overlap)
    pri_inc, pri_exp = _totals(pri_overlap)

    def _pct(cur_val: float, pri_val: float) -> Optional[float]:
        if pri_val == 0:
            return None
        return round(((cur_val - pri_val) / abs(pri_val)) * 100, 1)

    # Monthly breakdown (overlapping months only)
    monthly_rows: list[dict] = []
    for mo in overlap:
        c = cur[cur["_month"] == mo]["_amt"]
        p = pri[pri["_month"] == mo]["_amt"]
        c_inc = float(c[c > 0].sum())
        p_inc = float(p[p > 0].sum())
        c_exp = float(c[c < 0].abs().sum())
        p_exp = float(p[p < 0].abs().sum())
        monthly_rows.append({
            "month": int(mo),
            "month_name": calendar.month_abbr[mo],
            "current_income": round(c_inc, 2),
            "prior_income": round(p_inc, 2),
            "income_delta": round(c_inc - p_inc, 2),
            "current_expenses": round(c_exp, 2),
            "prior_expenses": round(p_exp, 2),
            "expenses_delta": round(c_exp - p_exp, 2),
        })
    monthly_df = pd.DataFrame(monthly_rows)

    # Top category changes (by absolute dollar delta in expenses)
    def _cat_totals(subset: pd.DataFrame) -> dict[str, float]:
        expenses = subset[subset["_amt"] < 0]
        if expenses.empty:
            return {}
        totals: dict[str, float] = {}
        for _, r in expenses.iterrows():
            cat = str(r.get("category", "Uncategorized"))
            totals[cat] = round(totals.get(cat, 0.0) + abs(r["_amt"]), 2)
        return totals

    cur_cats = _cat_totals(cur_overlap)
    pri_cats = _cat_totals(pri_overlap)
    all_cats = set(cur_cats.keys()) | set(pri_cats.keys())

    cat_changes: list[dict] = []
    for cat in all_cats:
        c_amt = cur_cats.get(cat, 0.0)
        p_amt = pri_cats.get(cat, 0.0)
        delta = round(c_amt - p_amt, 2)
        cat_changes.append({
            "category": cat,
            "current_amount": c_amt,
            "prior_amount": p_amt,
            "delta": delta,
            "delta_pct": _pct(c_amt, p_amt),
        })

    cat_changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    top_5 = cat_changes[:5]

    return YoYComparison(
        has_prior_year=True,
        current_year=current_year,
        prior_year=prior_year,
        ytd_label=ytd_label,
        ytd_current_income=cur_inc,
        ytd_prior_income=pri_inc,
        ytd_income_delta=round(cur_inc - pri_inc, 2),
        ytd_income_delta_pct=_pct(cur_inc, pri_inc),
        ytd_current_expenses=cur_exp,
        ytd_prior_expenses=pri_exp,
        ytd_expenses_delta=round(cur_exp - pri_exp, 2),
        ytd_expenses_delta_pct=_pct(cur_exp, pri_exp),
        monthly=monthly_df,
        top_category_changes=top_5,
    )
