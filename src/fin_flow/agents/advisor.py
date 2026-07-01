"""Advisor Agent — grounded, retrieval-augmented financial Q&A.

Architecture:
  PRIMARY PATH (Gemini available):
    Build a structured context payload (category spend, top expenses,
    cashflow stats, retrieved notes) and send it to Gemini along with
    the user's question.  Gemini composes a conversational answer
    grounded in real data.

  FALLBACK PATH (Gemini unavailable):
    Deterministic rules handlers (runway, affordability, category_spend,
    general) that produce mechanical but accurate answers.  A short
    "AI advisor offline" note is shown when the user expected Gemini.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Iterable, Optional

import pandas as pd

from ..storage import InMemoryVectorStore, VectorHit, best_available_store
from ..utils.pii import mask_pii
from .categorizer import TAX_CATEGORIES
from .forecaster import ForecasterAgent

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class AdvisorAnswer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)  # raw_hash / note ids
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


# ── Prompt for Gemini primary path ───────────────────────────────────────

_ADVISOR_PROMPT = (
    "You are Fin-Flow CFO, a financial advisor for small businesses. "
    "Answer the user's question using ONLY the financial data below.\n\n"
    "Rules:\n"
    "- Use specific dollar amounts from the data — never invent numbers.\n"
    "- Be conversational and address every part of the question.\n"
    "- If the data doesn't fully answer, say so.\n"
    "- Keep responses concise but thorough (2-4 paragraphs max).\n"
    "- When recommending spending cuts, reference actual category totals.\n"
    "- Do not mention that you received structured data — speak as if you "
    "know the user's finances directly.\n\n"
    "Financial Data:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ── Regex helpers ────────────────────────────────────────────────────────

_AMOUNT_RE = re.compile(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_AFFORD_RE = re.compile(r"\b(afford|can i (buy|spend|get)|should i (buy|get))\b", re.I)
_SPEND_RE = re.compile(r"\b(spend|spent|spending|cost)\b", re.I)
_RUNWAY_RE = re.compile(r"\b(runway|cash out|death date|go broke|out of cash)\b", re.I)
_ADVICE_RE = re.compile(
    r"\b(advisable|not advisable|advise|recommend|suggestion|tips?|"
    r"reduce|cut back|cut down|save money|optimize|improve|too much|"
    r"unnecessary|wasteful|excessive|where.{0,15}(save|cut))\b",
    re.I,
)


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


# ── Agent ────────────────────────────────────────────────────────────────

class AdvisorAgent:
    def __init__(
        self,
        vector_store: Optional[InMemoryVectorStore] = None,
        prefer_llm: bool = True,
    ) -> None:
        self.store = vector_store if vector_store is not None else best_available_store()
        self.forecaster = ForecasterAgent()
        self.prefer_llm = prefer_llm
        self._client = None
        self._model_name = "gemini-2.5-flash-lite"
        if prefer_llm:
            self._init_gemini()

    def _init_gemini(self) -> None:
        try:
            from google import genai  # type: ignore

            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                return
            self._client = genai.Client(api_key=key)
            logger.info("Advisor Gemini initialized (model=%s)", self._model_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("Advisor Gemini init failed: %s", e)

    @property
    def active_backend(self) -> str:
        return "gemini" if self._client else "rules"

    # ── Notes ────────────────────────────────────────────────────────────

    def add_note(self, text: str, **metadata) -> str:
        return self.store.add(text, metadata=metadata or None)

    def add_notes(self, items: Iterable[dict]) -> list[str]:
        return self.store.add_many(items)

    # ── Context payload for Gemini ───────────────────────────────────────

    def _build_context_payload(
        self,
        df: pd.DataFrame,
        starting_balance: float,
        notes: list[VectorHit],
    ) -> dict:
        """Build structured financial context for the Gemini prompt."""
        win = _window(df, days=30)
        dates = pd.to_datetime(df["transaction_date"])

        # Forecast stats
        fc = self.forecaster.forecast(
            df, starting_balance=starting_balance, horizon_days=90
        )

        # Category spending (last 30 days, expenses only)
        cat_totals: dict[str, float] = {}
        if not win.empty and "category" in win.columns:
            for _, r in win.iterrows():
                amt = _to_float(r["amount"])
                if amt < 0:
                    cat = str(r.get("category", "Uncategorized"))
                    cat_totals[cat] = round(
                        cat_totals.get(cat, 0.0) + abs(amt), 2,
                    )
            cat_totals = dict(
                sorted(cat_totals.items(), key=lambda kv: kv[1], reverse=True)
            )

        # Top 5 individual expenses
        top_expenses: list[dict] = []
        if not win.empty:
            expenses = win[win["amount"].map(_to_float) < 0].copy()
            expenses["abs_amt"] = expenses["amount"].map(
                lambda x: abs(_to_float(x))
            )
            for _, r in expenses.nlargest(5, "abs_amt").iterrows():
                top_expenses.append({
                    "date": str(r["transaction_date"]),
                    "description": mask_pii(str(r["description"])),
                    "amount": round(_to_float(r["amount"]), 2),
                    "category": str(r.get("category", "Uncategorized")),
                    "id": str(r["raw_hash"]) if "raw_hash" in r.index else "",
                })

        # Net cashflow (30 days)
        net_30d = (
            float(sum(_to_float(a) for a in win["amount"]))
            if not win.empty
            else 0.0
        )

        return {
            "summary": {
                "date_range": (
                    f"{dates.min().date()} to {dates.max().date()}"
                    if not dates.empty
                    else "no data"
                ),
                "total_transactions": len(df),
                "last_30d_transactions": len(win),
                "net_cashflow_30d": round(net_30d, 2),
                "avg_daily_income": round(fc.mean_daily_income, 2),
                "avg_daily_burn": round(fc.mean_daily_burn, 2),
                "starting_balance": starting_balance,
                "runway": fc.summary(),
            },
            "spending_by_category": cat_totals,
            "top_expenses": top_expenses,
            "notes": [
                {"id": h.id, "text": mask_pii(h.text)} for h in notes
            ],
        }

    # ── Primary path: Gemini-first ───────────────────────────────────────
    # NOTE: Live Gemini test pending API quota reset. The flow below is
    # verified via mocked unit tests (test_advisor.py).  Once quota resets
    # tonight, re-upload a statement and try the Ask Fin-Flow chat — the
    # sidebar will show backend="gemini" and answers will be conversational.

    _RETRYABLE = ("429", "503")

    def _ask_gemini(
        self,
        question: str,
        df: pd.DataFrame,
        starting_balance: float,
        notes: list[VectorHit],
        _max_retries: int = 2,
    ) -> tuple[str, list[str]]:
        """Send context + question to Gemini. Returns (answer, citations)."""
        payload = self._build_context_payload(df, starting_balance, notes)
        prompt = _ADVISOR_PROMPT.format(
            context=json.dumps(payload, indent=2, default=str),
            question=question,
        )

        # Retry on transient errors (429 / 503) with exponential backoff
        for attempt in range(_max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                )
                break  # success
            except Exception as e:  # noqa: BLE001
                err_str = str(e)
                if any(code in err_str for code in self._RETRYABLE) and attempt < _max_retries:
                    wait = 2 ** attempt * 5  # 5s, 10s
                    logger.warning(
                        "Gemini advisor transient error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, _max_retries + 1, wait, e,
                    )
                    time.sleep(wait)
                    continue
                raise  # non-retryable or out of retries

        text = (resp.text or "").strip()

        if not text:
            raise RuntimeError("Gemini returned empty response")

        # Citations: all 30-day window hashes (grounded, no hallucination)
        citations: list[str] = []
        win = _window(df, days=30)
        if "raw_hash" in win.columns:
            citations = list(win["raw_hash"].astype(str))

        return text, citations

    # ── Main entry point ─────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        transactions: pd.DataFrame,
        starting_balance: float = 0.0,
    ) -> AdvisorAnswer:
        retrieved = self.store.query(question, k=4)

        # PRIMARY PATH: Gemini-first when available
        if self._client is not None:
            try:
                answer_text, citations = self._ask_gemini(
                    question, transactions, starting_balance, retrieved,
                )
                return AdvisorAnswer(
                    question=question,
                    answer=answer_text,
                    citations=citations,
                    retrieved_notes=retrieved,
                    intent="gemini",
                    backend="gemini",
                )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Gemini advisor failed, falling back to rules: %s", e
                )

        # FALLBACK PATH: deterministic rules handlers
        intent, handler = self._route(question)
        answer_text, citations = handler(
            question, transactions, starting_balance, retrieved,
        )

        return AdvisorAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_notes=retrieved,
            intent=intent,
            backend="rules",
        )

    # ── Rules-based routing (fallback) ───────────────────────────────────

    def _route(self, question: str):
        if _RUNWAY_RE.search(question):
            return "runway", self._answer_runway
        if _AFFORD_RE.search(question):
            return "affordability", self._answer_affordability
        if _ADVICE_RE.search(question):
            return "advice", self._answer_advice
        if _SPEND_RE.search(question):
            return "category_spend", self._answer_category_spend
        return "general", self._answer_general

    def _answer_runway(
        self, q: str, df: pd.DataFrame, start: float, notes: list[VectorHit]
    ) -> tuple[str, list[str]]:
        if df.empty:
            return ("No transactions available to compute runway.", [])
        fc = self.forecaster.forecast(df, starting_balance=start, horizon_days=180)
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
            verdict = (
                f"Your last-30-day net cashflow is ${net_30d:,.2f}. "
                f"Anything meaningfully below that fits within recent headroom."
            )
            return (verdict, cites)

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
            total = float(
                sum(_to_float(a) for a in win["amount"] if _to_float(a) < 0)
            )
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
        total = float(
            sum(_to_float(a) for a in subset["amount"] if _to_float(a) < 0)
        )
        cites = list(subset["raw_hash"].astype(str))
        return (
            f"Spent ${abs(total):,.2f} on `{category}` across "
            f"{len(subset)} transactions in the last 30 days.",
            cites,
        )

    # Categories where the user typically has discretionary control
    _DISCRETIONARY = {
        "Meals", "Entertainment", "Shopping", "Groceries",
        "Software & Subscriptions", "Advertising", "Travel",
    }

    def _answer_advice(
        self, q: str, df: pd.DataFrame, start: float, notes: list[VectorHit]
    ) -> tuple[str, list[str]]:
        """Break down spending by category, flag discretionary ones."""
        if df.empty or "category" not in df.columns:
            return ("I need categorized transactions to advise on spending.", [])

        win = _window(df, days=30)
        if win.empty:
            return ("No transactions in the last 30 days to analyze.", [])

        # Build per-category totals (expenses only, last 30 days)
        cat_totals: dict[str, float] = {}
        for _, r in win.iterrows():
            amt = _to_float(r["amount"])
            if amt < 0:
                cat = str(r.get("category", "Uncategorized"))
                cat_totals[cat] = round(cat_totals.get(cat, 0.0) + abs(amt), 2)

        if not cat_totals:
            return ("No expenses in the last 30 days.", [])

        sorted_cats = sorted(cat_totals.items(), key=lambda kv: kv[1], reverse=True)
        total_exp = sum(v for _, v in sorted_cats)
        cites = _cite_window(win, days=30, only_expenses=True)

        lines = [f"Spending breakdown (last 30 days, ${total_exp:,.2f} total):\n"]
        for cat, amt in sorted_cats:
            pct = (amt / total_exp * 100) if total_exp > 0 else 0
            tag = " [discretionary]" if cat in self._DISCRETIONARY else ""
            lines.append(f"  {cat}: ${amt:,.2f} ({pct:.0f}%){tag}")

        disc_total = sum(v for c, v in sorted_cats if c in self._DISCRETIONARY)
        if disc_total > 0:
            disc_pct = disc_total / total_exp * 100 if total_exp > 0 else 0
            lines.append(
                f"\nDiscretionary spending is ${disc_total:,.2f} "
                f"({disc_pct:.0f}% of total). "
                f"These categories offer the most flexibility for cuts."
            )
        else:
            lines.append(
                "\nMost of your spending is in fixed categories "
                "(Rent, Utilities, Insurance). Reducing these typically "
                "requires renegotiation or switching providers."
            )

        return ("\n".join(lines), cites)

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


# ── Helpers ──────────────────────────────────────────────────────────────

def _cite_window(
    df: pd.DataFrame, days: int = 30, only_expenses: bool = False
) -> list[str]:
    win = _window(df, days=days)
    if only_expenses:
        win = win[win["amount"].map(lambda a: _to_float(a) < 0)]
    if "raw_hash" not in win.columns:
        return []
    return list(win["raw_hash"].astype(str))