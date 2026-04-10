"""Advisor Agent — grounded, retrieval-augmented financial Q&A.

Design principles:

1. **Grounded answers only.** Every financial claim is backed by a list
   of transaction IDs (and/or note IDs). The agent never asserts a dollar
   amount without a citation the user can trace.

2. **Deterministic by default, LLM-upgradable.** The default backend is a
   pattern-matching interpreter that handles three high-value question
   families — affordability, category-spend, and runway — plus a generic
   retrieval fallback. If the optional `GEMINI_API_KEY` is set, the agent
   passes the retrieved context to Gemini 1.5 Flash for a more fluent
   answer while still returning the original citations.

3. **PII never leaves the process untouched.** Before any text is sent
   outbound, transaction descriptions are passed through `mask_pii`.

4. **Uses the existing Forecaster and Categorizer.** The Advisor does
   not re-implement cashflow math; it composes agents.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Iterable, Optional

import pandas as pd

from ..storage import InMemoryVectorStore, VectorHit, best_available_store
from ..utils.pii import mask_pii
from .categorizer import TAX_CATEGORIES
from .forecaster import ForecasterAgent


@dataclass
class AdvisorAnswer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)  # transaction ids / note ids
    retrieved_notes: list[VectorHit] = field(default_factory=list)
    intent: str = "general"
    backend: str = "rules"

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "retrieved_notes": [
                {"id": h.id, "score": round(h.score, 3), "text": h.text}
                for h in self.retrieved_notes
            ],
            "intent": self.intent,
            "backend": self.backend,
        }


# ---------- utilities ----------

_AMOUNT_RE = re.compile(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_AFFORD_RE = re.compile(r"\b(afford|can i (buy|spend|get)|should i (buy|get))\b", re.I)
_SPEND_RE = re.compile(r"\b(spend|spent|spending|cost)\b", re.I)
_RUNWAY_RE = re.compile(r"\b(runway|cash out|death date|go broke|out of cash)\b", re.I)


def _parse_amount(text: str) -> Optional[float]:
    m = _AMOUNT_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _to_float(x) -> float:
    if isinstance(x, Decimal):
        return float(x)
    return float(x)


def _window(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["transaction_date"] = pd.to_datetime(work["transaction_date"]).dt.date
    cutoff = max(work["transaction_date"]) - timedelta(days=days)
    return work[work["transaction_date"] >= cutoff]


def _match_category(question: str) -> Optional[str]:
    q = question.lower()
    for cat in TAX_CATEGORIES:
        if cat.lower() in q:
            return cat
    # common synonyms
    synonyms = {
        "food": "Meals",
        "restaurant": "Meals",
        "dining": "Meals",
        "coffee": "Meals",
        "gas": "Travel",
        "flights": "Travel",
        "uber": "Travel",
        "subscription": "Software & Subscriptions",
        "saas": "Software & Subscriptions",
        "ads": "Advertising",
        "marketing": "Advertising",
        "rent": "Rent",
        "office": "Office Supplies",
    }
    for k, v in synonyms.items():
        if k in q:
            return v
    return None


# ---------- Advisor ----------


class AdvisorAgent:
    def __init__(
        self,
        vector_store: Optional[InMemoryVectorStore] = None,
        prefer_llm: bool = True,
    ) -> None:
        # NOTE: do not use `or` here — InMemoryVectorStore defines __len__
        # so an empty store is Python-falsy and would be silently replaced.
        self.store = vector_store if vector_store is not None else best_available_store()
        self.forecaster = ForecasterAgent()
        self.prefer_llm = prefer_llm
        self._gemini = None
        if prefer_llm:
            self._gemini = self._init_gemini()

    def _init_gemini(self):
        try:
            import google.generativeai as genai  # type: ignore

            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                return None
            genai.configure(api_key=key)
            return genai.GenerativeModel("gemini-1.5-flash")
        except Exception:  # noqa: BLE001
            return None

    @property
    def active_backend(self) -> str:
        return "gemini" if self._gemini else "rules"

    # ---- note ingestion ----

    def add_note(self, text: str, **metadata) -> str:
        return self.store.add(text, metadata=metadata or None)

    def add_notes(self, items: Iterable[dict]) -> list[str]:
        return self.store.add_many(items)

    # ---- core Q&A ----

    def ask(
        self,
        question: str,
        transactions: pd.DataFrame,
        starting_balance: float = 0.0,
    ) -> AdvisorAnswer:
        retrieved = self.store.query(question, k=4)

        intent, handler = self._route(question)
        answer_text, citations = handler(question, transactions, starting_balance, retrieved)

        result = AdvisorAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_notes=retrieved,
            intent=intent,
            backend=self.active_backend,
        )

        if self._gemini is not None:
            try:
                result.answer = self._rewrite_with_gemini(result)
            except Exception:  # noqa: BLE001
                pass  # keep deterministic answer

        return result

    # ---- intent routing ----

    def _route(self, question: str):
        if _RUNWAY_RE.search(question):
            return "runway", self._answer_runway
        if _AFFORD_RE.search(question):
            return "affordability", self._answer_affordability
        if _SPEND_RE.search(question):
            return "category_spend", self._answer_category_spend
        return "general", self._answer_general

    # ---- handlers ----

    def _answer_runway(
        self, q: str, df: pd.DataFrame, start: float, notes: list[VectorHit]
    ) -> tuple[str, list[str]]:
        if df.empty:
            return ("No transactions available to compute runway.", [])
        fc = self.forecaster.forecast(df, starting_balance=start, horizon_days=180)
        # Ground with every transaction used in the window
        cites = _cite_window(df, days=30)
        return (fc.summary(), cites)

    def _answer_affordability(
        self, q: str, df: pd.DataFrame, start: float, notes: list[VectorHit]
    ) -> tuple[str, list[str]]:
        amount = _parse_amount(q)
        win = _window(df, days=30)
        if win.empty:
            return (
                "I don't have enough recent transactions to judge affordability.",
                [],
            )
        net_30d = float(sum(_to_float(a) for a in win["amount"]))
        cites = _cite_window(win, days=30)

        if amount is None:
            # No price given — advise on discretionary headroom
            verdict = (
                f"Your last-30-day net cashflow is ${net_30d:,.2f}. "
                f"Anything meaningfully below that fits within recent headroom."
            )
            return (verdict, cites)

        # Simple rule: affordable if price <= 50% of last-30d net cashflow
        # and does not push starting balance below one month of burn.
        fc = self.forecaster.forecast(df, starting_balance=start, horizon_days=90)
        burn = fc.mean_daily_burn * 30
        cushion = start - burn
        if net_30d <= 0:
            verdict = (
                f"Not advisable: last-30-day net cashflow is ${net_30d:,.2f} "
                f"(negative). Covering ${amount:,.2f} would deepen the burn."
            )
        elif amount > 0.5 * net_30d and amount > cushion:
            verdict = (
                f"Risky: ${amount:,.2f} is more than half of last-30-day net "
                f"cashflow (${net_30d:,.2f}) and larger than your one-month "
                f"cushion (${cushion:,.2f} after expected burn)."
            )
        else:
            verdict = (
                f"Yes, affordable: ${amount:,.2f} fits within last-30-day net "
                f"cashflow of ${net_30d:,.2f} and leaves a one-month cushion "
                f"of ${cushion:,.2f}."
            )
        return (verdict, cites)

    def _answer_category_spend(
        self, q: str, df: pd.DataFrame, start: float, notes: list[VectorHit]
    ) -> tuple[str, list[str]]:
        if df.empty or "category" not in df.columns:
            return ("I need categorized transactions to answer spend questions.", [])

        category = _match_category(q)
        win = _window(df, days=30)
        if win.empty:
            return ("No transactions in the last 30 days.", [])

        if category is None:
            # Total spend in window
            total = float(sum(_to_float(a) for a in win["amount"] if _to_float(a) < 0))
            cites = _cite_window(win, days=30, only_expenses=True)
            return (
                f"Total spend in the last 30 days: ${abs(total):,.2f}.",
                cites,
            )

        subset = win[win["category"] == category]
        if subset.empty:
            return (
                f"No transactions in category `{category}` in the last 30 days.",
                [],
            )
        total = float(sum(_to_float(a) for a in subset["amount"] if _to_float(a) < 0))
        cites = list(subset["raw_hash"].astype(str))
        return (
            f"Spent ${abs(total):,.2f} on `{category}` across "
            f"{len(subset)} transactions in the last 30 days.",
            cites,
        )

    def _answer_general(
        self, q: str, df: pd.DataFrame, start: float, notes: list[VectorHit]
    ) -> tuple[str, list[str]]:
        if notes:
            top = notes[0]
            return (
                f"Based on your business context ({top.metadata.get('type', 'note')}): "
                f"{top.text}",
                [top.id],
            )
        # fall back to a summary
        if df.empty:
            return ("I don't have any transactions or notes to draw on yet.", [])
        total = float(sum(_to_float(a) for a in df["amount"]))
        return (
            f"Net cashflow across the loaded period is ${total:,.2f}. "
            f"Ask me about spend by category, runway, or affordability.",
            _cite_window(df, days=30),
        )

    # ---- optional Gemini rewrite ----

    def _rewrite_with_gemini(self, draft: AdvisorAnswer) -> str:
        assert self._gemini is not None
        context_notes = "\n".join(
            f"- [{h.id}] {mask_pii(h.text)}" for h in draft.retrieved_notes
        ) or "(none)"
        prompt = (
            "You are Fin-Flow CFO. Rewrite the draft answer below so it sounds "
            "natural, but DO NOT change any numbers and DO NOT add new financial "
            "claims. If the draft cites no transactions, do not invent any.\n\n"
            f"Question: {draft.question}\n"
            f"Draft: {draft.answer}\n"
            f"Context notes:\n{context_notes}\n\n"
            "Rewritten answer:"
        )
        resp = self._gemini.generate_content(prompt)  # type: ignore[attr-defined]
        text = (resp.text or "").strip()
        return text or draft.answer


def _cite_window(df: pd.DataFrame, days: int = 30, only_expenses: bool = False) -> list[str]:
    win = _window(df, days=days)
    if only_expenses:
        win = win[win["amount"].map(lambda a: _to_float(a) < 0)]
    if "raw_hash" not in win.columns:
        return []
    return list(win["raw_hash"].astype(str))
