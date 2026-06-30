from datetime import date, timedelta
from decimal import Decimal

import pandas as pd

from fin_flow.agents import ForecasterAgent
from fin_flow.agents.forecaster import LinearBackend, _build_daily_series, compute_period_deltas


def _burning_business(start: date, days: int = 30, daily_burn: float = -100.0) -> pd.DataFrame:
    rows = []
    for i in range(days):
        rows.append(
            {
                "transaction_date": start + timedelta(days=i),
                "amount": Decimal(str(daily_burn)),
                "description": f"daily expense {i}",
            }
        )
    return pd.DataFrame(rows)


def _profitable_business(start: date, days: int = 30, daily_net: float = 250.0) -> pd.DataFrame:
    rows = []
    for i in range(days):
        rows.append(
            {
                "transaction_date": start + timedelta(days=i),
                "amount": Decimal(str(daily_net)),
                "description": f"daily revenue {i}",
            }
        )
    return pd.DataFrame(rows)


def test_daily_series_aggregates_by_day():
    df = pd.DataFrame(
        {
            "transaction_date": [date(2026, 3, 1), date(2026, 3, 1), date(2026, 3, 3)],
            "amount": [Decimal("-10"), Decimal("-5"), Decimal("100")],
            "description": ["a", "b", "c"],
        }
    )
    daily = _build_daily_series(df)
    # Should fill the gap on 3/2 with 0
    assert len(daily) == 3
    assert daily.loc[date(2026, 3, 1)] == -15
    assert daily.loc[date(2026, 3, 2)] == 0
    assert daily.loc[date(2026, 3, 3)] == 100


def test_burning_business_has_death_date():
    df = _burning_business(start=date(2026, 1, 1), days=30, daily_burn=-100.0)
    agent = ForecasterAgent(prefer_prophet=False)
    fc = agent.forecast(df, starting_balance=1000.0, horizon_days=60)
    assert fc.backend == "linear"
    assert fc.death_date is not None
    # 1000 starting + 30 days of -100 = -2000 cum balance at end of history.
    # Already underwater, so death_date should be the very next projected day.
    assert fc.death_date == date(2026, 1, 31)
    assert fc.mean_daily_burn == 100.0
    # Summary string mentions runway
    assert "runway" in fc.summary().lower() or "balance hits $0" in fc.summary()


def test_profitable_business_has_no_death_date():
    df = _profitable_business(start=date(2026, 1, 1), days=30, daily_net=250.0)
    agent = ForecasterAgent(prefer_prophet=False)
    fc = agent.forecast(df, starting_balance=5000.0, horizon_days=90)
    assert fc.death_date is None
    assert fc.mean_daily_net == 250.0
    assert fc.mean_daily_burn == 0.0
    assert "positive" in fc.summary().lower()


def test_runway_with_starting_cushion():
    # Burns $100/day with $3000 starting cash. Should die ~30 days out.
    df = _burning_business(start=date(2026, 1, 1), days=10, daily_burn=-100.0)
    agent = ForecasterAgent(prefer_prophet=False)
    fc = agent.forecast(df, starting_balance=3000.0, horizon_days=60)
    assert fc.death_date is not None
    # Last historical day is 2026-01-10, balance = 3000 - 1000 = 2000.
    # At -100/day, hits 0 in 20 days -> 2026-01-30.
    assert fc.death_date == date(2026, 1, 30)


def test_empty_dataframe_returns_safe_forecast():
    agent = ForecasterAgent(prefer_prophet=False)
    fc = agent.forecast(pd.DataFrame(), starting_balance=100.0)
    assert fc.death_date is None
    assert fc.projection.empty


# ── mean_daily_income tests ──────────────────────────────────────────

def test_mean_daily_income_mixed_business():
    """Gross daily income is total positive amounts / number of days."""
    rows = []
    for i in range(10):
        d = date(2026, 3, 1) + timedelta(days=i)
        rows.append({"transaction_date": d, "amount": Decimal("200"), "description": "sale"})
        rows.append({"transaction_date": d, "amount": Decimal("-80"), "description": "cost"})
    df = pd.DataFrame(rows)
    agent = ForecasterAgent(prefer_prophet=False)
    fc = agent.forecast(df, starting_balance=0.0)
    # 200/day income over 10 days
    assert fc.mean_daily_income == 200.0
    # Net is positive every day so mean_daily_burn stays 0
    assert fc.mean_daily_burn == 0.0


def test_mean_daily_income_zero_for_expense_only():
    df = _burning_business(start=date(2026, 1, 1), days=10)
    agent = ForecasterAgent(prefer_prophet=False)
    fc = agent.forecast(df, starting_balance=1000.0)
    assert fc.mean_daily_income == 0.0


def test_mean_daily_income_profitable():
    df = _profitable_business(start=date(2026, 1, 1), days=20, daily_net=300.0)
    agent = ForecasterAgent(prefer_prophet=False)
    fc = agent.forecast(df, starting_balance=0.0)
    assert fc.mean_daily_income == 300.0


# ── compute_period_deltas tests ──────────────────────────────────────

def test_period_deltas_splits_evenly():
    rows = []
    # First 15 days: $100/day income
    for i in range(15):
        d = date(2026, 1, 1) + timedelta(days=i)
        rows.append({"transaction_date": d, "amount": Decimal("100"), "description": f"rev{i}"})
    # Last 15 days: $200/day income
    for i in range(15, 30):
        d = date(2026, 1, 1) + timedelta(days=i)
        rows.append({"transaction_date": d, "amount": Decimal("200"), "description": f"rev{i}"})
    df = pd.DataFrame(rows)
    delta = compute_period_deltas(df)
    assert delta.has_prior
    assert delta.income_delta > 0  # second half has more income
    # Floor-division midpoint gives a slight asymmetry (16 vs 14)
    assert abs(delta.transactions_delta) <= 2
    assert delta.prior_count + delta.current_count == 30


def test_period_deltas_insufficient_history():
    rows = [
        {"transaction_date": date(2026, 1, 1), "amount": Decimal("100"), "description": "a"},
        {"transaction_date": date(2026, 1, 5), "amount": Decimal("200"), "description": "b"},
    ]
    df = pd.DataFrame(rows)
    delta = compute_period_deltas(df)
    assert not delta.has_prior


def test_period_deltas_empty_dataframe():
    delta = compute_period_deltas(pd.DataFrame())
    assert not delta.has_prior


def test_period_deltas_expenses_tracked():
    rows = []
    # First half: small expenses
    for i in range(15):
        d = date(2026, 1, 1) + timedelta(days=i)
        rows.append({"transaction_date": d, "amount": Decimal("-50"), "description": f"cost{i}"})
    # Second half: larger expenses
    for i in range(15, 30):
        d = date(2026, 1, 1) + timedelta(days=i)
        rows.append({"transaction_date": d, "amount": Decimal("-120"), "description": f"cost{i}"})
    df = pd.DataFrame(rows)
    delta = compute_period_deltas(df)
    assert delta.has_prior
    assert delta.expenses_delta > 0  # expenses grew
    assert delta.expenses_delta_pct is not None
    assert delta.expenses_delta_pct > 0


def test_period_deltas_thin_prior_has_counts():
    """When prior half has very few transactions, counts reflect that."""
    rows = []
    # 2 transactions early in the month
    rows.append({"transaction_date": date(2026, 1, 1), "amount": Decimal("500"), "description": "deposit"})
    rows.append({"transaction_date": date(2026, 1, 3), "amount": Decimal("-100"), "description": "bill"})
    # 15 transactions late in the month
    for i in range(15, 30):
        d = date(2026, 1, 1) + timedelta(days=i)
        rows.append({"transaction_date": d, "amount": Decimal("-80"), "description": f"cost{i}"})
    df = pd.DataFrame(rows)
    delta = compute_period_deltas(df)
    assert delta.has_prior
    assert delta.prior_count == 2
    assert delta.current_count == 15
    # Dollar delta is always populated
    assert delta.income_delta != 0 or delta.expenses_delta != 0
