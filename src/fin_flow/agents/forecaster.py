"""Forecaster Agent — daily cash-flow projection and runway."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Forecast:
    starting_balance: float
    mean_daily_net: float
    mean_daily_burn: float  # absolute value of mean negative net (>=0)
    death_date: Optional[date]
    horizon_days: int
    projection: pd.DataFrame = field(repr=False)
    backend: str = "linear"

    def summary(self) -> str:
        if self.death_date is None:
            return (
                f"Cash-flow positive: avg daily net "
                f"${self.mean_daily_net:,.2f}. No runway risk in "
                f"the next {self.horizon_days} days."
            )
        days = (self.death_date - date.today()).days
        return (
            f"Projected runway: {days} days. "
            f"Balance hits $0 on {self.death_date.isoformat()} "
            f"at avg daily burn ${self.mean_daily_burn:,.2f}."
        )


def _to_float(x) -> float:
    if isinstance(x, Decimal):
        return float(x)
    return float(x)


def _build_daily_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    work = df[["transaction_date", "amount"]].copy()
    work["transaction_date"] = pd.to_datetime(work["transaction_date"])
    work["amount"] = work["amount"].map(_to_float)
    daily = (
        work.groupby(work["transaction_date"].dt.date)["amount"]
        .sum()
        .sort_index()
    )
    if not daily.empty:
        full = pd.date_range(daily.index.min(), daily.index.max(), freq="D").date
        daily = daily.reindex(full, fill_value=0.0)
    return daily


class LinearBackend:
    name = "linear"

    def project(
        self,
        daily: pd.Series,
        starting_balance: float,
        horizon_days: int,
    ) -> pd.DataFrame:
        if daily.empty:
            return pd.DataFrame(columns=["date", "projected_balance"])
        cum = daily.cumsum() + starting_balance
        slope = float(daily.mean())
        last_date = list(daily.index)[-1]
        last_balance = float(cum.iloc[-1])
        future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]
        future_balances = [last_balance + slope * (i + 1) for i in range(horizon_days)]
        return pd.DataFrame(
            {"date": future_dates, "projected_balance": future_balances}
        )


class ProphetBackend:
    name = "prophet"

    def __init__(self) -> None:
        self._available = False
        try:
            import prophet  # type: ignore  # noqa: F401

            self._available = True
        except Exception:  # noqa: BLE001
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def project(
        self,
        daily: pd.Series,
        starting_balance: float,
        horizon_days: int,
    ) -> pd.DataFrame:
        from prophet import Prophet  # type: ignore

        cum = daily.cumsum() + starting_balance
        history = pd.DataFrame(
            {"ds": pd.to_datetime(list(cum.index)), "y": cum.values}
        )
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(history)
        future = m.make_future_dataframe(periods=horizon_days, freq="D")
        fcst = m.predict(future)
        tail = fcst.tail(horizon_days)
        return pd.DataFrame(
            {
                "date": [d.date() for d in tail["ds"]],
                "projected_balance": tail["yhat"].astype(float).tolist(),
            }
        )


class ForecasterAgent:
    def __init__(self, prefer_prophet: bool = True) -> None:
        self.linear = LinearBackend()
        self.prophet: Optional[ProphetBackend] = None
        if prefer_prophet:
            backend = ProphetBackend()
            if backend.available:
                self.prophet = backend

    @property
    def active_backend(self) -> str:
        return self.prophet.name if self.prophet else self.linear.name

    def forecast(
        self,
        df: pd.DataFrame,
        starting_balance: float = 0.0,
        horizon_days: int = 90,
    ) -> Forecast:
        daily = _build_daily_series(df)
        if daily.empty:
            empty_proj = pd.DataFrame(columns=["date", "projected_balance"])
            return Forecast(
                starting_balance=starting_balance,
                mean_daily_net=0.0,
                mean_daily_burn=0.0,
                death_date=None,
                horizon_days=horizon_days,
                projection=empty_proj,
                backend=self.active_backend,
            )

        backend = self.prophet or self.linear
        projection = backend.project(daily, starting_balance, horizon_days)

        mean_net = float(daily.mean())
        negatives = daily[daily < 0]
        mean_burn = float(-negatives.mean()) if not negatives.empty else 0.0

        death_date: Optional[date] = None
        if not projection.empty:
            below = projection[projection["projected_balance"] <= 0]
            if not below.empty:
                death_date = below.iloc[0]["date"]

        return Forecast(
            starting_balance=starting_balance,
            mean_daily_net=mean_net,
            mean_daily_burn=mean_burn,
            death_date=death_date,
            horizon_days=horizon_days,
            projection=projection,
            backend=backend.name,
        )
