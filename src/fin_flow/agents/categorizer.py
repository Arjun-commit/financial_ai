"""Categorizer Agent — assigns IRS / Schedule-C-aligned tax categories.

Two execution modes:

* `RulesBackend` (default, zero-cost, deterministic) — keyword scoring against
  a curated lexicon. Runs offline, no API key required, fast enough for
  hundreds of thousands of rows.
* `GeminiBackend` (optional) — wraps `google.generativeai` and asks
  Gemini 1.5 Flash to classify a batch of descriptions when the
  `GEMINI_API_KEY` env var is present. Falls back to the rules backend
  automatically when the SDK or key is missing.

Both backends share a `Categorization` result type so the pipeline never
cares which one ran.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from ..utils.pii import mask_pii

# IRS Schedule C-style buckets that small businesses and freelancers use.
# Order is significant: more specific categories should come before broad
# ones so the rules backend prefers them in the case of a tie.
TAX_CATEGORIES: tuple[str, ...] = (
    "Income",
    "Payroll",
    "Rent",
    "Utilities",
    "Office Supplies",
    "Software & Subscriptions",
    "Travel",
    "Meals",
    "Advertising",
    "Professional Services",
    "Insurance",
    "Taxes & Fees",
    "Bank Fees",
    "Groceries",
    "Entertainment",
    "Shopping",
    "Transfers",
    "Uncategorized",
)

# Lower-cased keyword -> category. Multi-word phrases are matched as-is.
_KEYWORD_LEXICON: dict[str, str] = {
    # income
    "payroll": "Income",
    "salary": "Income",
    "direct deposit": "Income",
    "stripe payout": "Income",
    "invoice": "Income",
    "ach credit": "Income",
    # payroll out
    "gusto": "Payroll",
    "adp payroll": "Payroll",
    "paychex": "Payroll",
    # rent
    "rent": "Rent",
    "lease": "Rent",
    "wework": "Rent",
    # utilities
    "electric": "Utilities",
    "pg&e": "Utilities",
    "comcast": "Utilities",
    "xfinity": "Utilities",
    "att internet": "Utilities",
    "verizon": "Utilities",
    "water": "Utilities",
    # office supplies
    "staples": "Office Supplies",
    "office depot": "Office Supplies",
    "best buy": "Office Supplies",
    # software / subscriptions
    "github": "Software & Subscriptions",
    "openai": "Software & Subscriptions",
    "anthropic": "Software & Subscriptions",
    "aws": "Software & Subscriptions",
    "google cloud": "Software & Subscriptions",
    "gcp": "Software & Subscriptions",
    "digitalocean": "Software & Subscriptions",
    "heroku": "Software & Subscriptions",
    "notion": "Software & Subscriptions",
    "figma": "Software & Subscriptions",
    "slack": "Software & Subscriptions",
    "zoom": "Software & Subscriptions",
    "netflix": "Software & Subscriptions",
    "spotify": "Software & Subscriptions",
    "adobe": "Software & Subscriptions",
    # travel
    "uber": "Travel",
    "lyft": "Travel",
    "delta air": "Travel",
    "united airlines": "Travel",
    "southwest air": "Travel",
    "airbnb": "Travel",
    "hotel": "Travel",
    "marriott": "Travel",
    "hilton": "Travel",
    # meals
    "starbucks": "Meals",
    "chipotle": "Meals",
    "doordash": "Meals",
    "ubereats": "Meals",
    "uber eats": "Meals",
    "restaurant": "Meals",
    "cafe": "Meals",
    "coffee": "Meals",
    # advertising
    "facebook ads": "Advertising",
    "meta ads": "Advertising",
    "google ads": "Advertising",
    "linkedin ads": "Advertising",
    "tiktok ads": "Advertising",
    # professional services
    "legalzoom": "Professional Services",
    "law firm": "Professional Services",
    "accountant": "Professional Services",
    "consulting": "Professional Services",
    # insurance
    "insurance": "Insurance",
    "geico": "Insurance",
    "state farm": "Insurance",
    # taxes & fees
    "irs": "Taxes & Fees",
    "tax payment": "Taxes & Fees",
    "franchise tax": "Taxes & Fees",
    # bank fees
    "service fee": "Bank Fees",
    "overdraft": "Bank Fees",
    "atm fee": "Bank Fees",
    "wire fee": "Bank Fees",
    # groceries
    "whole foods": "Groceries",
    "trader joe": "Groceries",
    "safeway": "Groceries",
    "kroger": "Groceries",
    "costco": "Groceries",
    # entertainment
    "amc theatres": "Entertainment",
    "ticketmaster": "Entertainment",
    "steam games": "Entertainment",
    # shopping
    "amazon": "Shopping",
    "amzn": "Shopping",
    "target": "Shopping",
    "walmart": "Shopping",
    "ebay": "Shopping",
    # transfers
    "transfer to": "Transfers",
    "transfer from": "Transfers",
    "venmo": "Transfers",
    "zelle": "Transfers",
    "cash app": "Transfers",
}


@dataclass
class Categorization:
    category: str
    confidence: float  # 0.0 - 1.0
    rationale: str = ""


class RulesBackend:
    """Deterministic keyword-scoring categorizer."""

    name = "rules"

    def __init__(self, lexicon: Optional[dict[str, str]] = None) -> None:
        self.lexicon = lexicon or _KEYWORD_LEXICON

    def classify_one(self, description: str, amount: float) -> Categorization:
        text = (description or "").lower()
        # Income heuristic: positive amount + income-flavored keyword
        scores: dict[str, float] = {}
        for keyword, category in self.lexicon.items():
            if keyword in text:
                # Longer keywords are more specific -> higher weight
                scores[category] = scores.get(category, 0.0) + len(keyword)

        if not scores:
            # Bias positive amounts toward Income when nothing else hits
            if amount > 0:
                return Categorization("Income", 0.55, "positive amount, no keyword match")
            return Categorization("Uncategorized", 0.30, "no keyword matched")

        best_cat, best_score = max(scores.items(), key=lambda kv: kv[1])
        total = sum(scores.values())
        confidence = min(0.99, 0.55 + (best_score / total) * 0.4)
        return Categorization(best_cat, round(confidence, 3), f"matched keywords -> {best_cat}")


class GeminiBackend:
    """Optional Gemini 1.5 Flash backend.

    Activates only when `google.generativeai` is importable AND a
    `GEMINI_API_KEY` is set in the environment. Otherwise the agent
    silently falls back to the rules backend.
    """

    name = "gemini"
    _SYSTEM_PROMPT = (
        "You are the Fin-Flow Categorizer. Classify each transaction into "
        "exactly one of these categories: {categories}. "
        "Return JSON: a list of objects with keys "
        "'index' (int), 'category' (string), 'confidence' (0..1)."
    )

    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        self.model_name = model_name
        self._model = None
        try:
            import google.generativeai as genai  # type: ignore

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model_name)
        except Exception as e:  # noqa: BLE001
            self._init_error = str(e)

    @property
    def available(self) -> bool:
        return self._model is not None

    def classify_batch(
        self, descriptions: list[str], amounts: list[float]
    ) -> list[Categorization]:
        if not self.available:
            raise RuntimeError("GeminiBackend not available")
        prompt = self._SYSTEM_PROMPT.format(categories=", ".join(TAX_CATEGORIES))
        rows = [
            {"index": i, "description": mask_pii(d), "amount": float(a)}
            for i, (d, a) in enumerate(zip(descriptions, amounts))
        ]
        body = prompt + "\n\nTransactions:\n" + json.dumps(rows)
        resp = self._model.generate_content(body)  # type: ignore[attr-defined]
        text = (resp.text or "").strip()
        # Strip code fences if Gemini wraps in ```json
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Gemini returned non-JSON: {e}") from e
        out = [Categorization("Uncategorized", 0.0, "gemini returned no row")] * len(rows)
        for item in parsed:
            idx = int(item.get("index", -1))
            cat = item.get("category", "Uncategorized")
            if cat not in TAX_CATEGORIES:
                cat = "Uncategorized"
            conf = float(item.get("confidence", 0.5))
            if 0 <= idx < len(out):
                out[idx] = Categorization(cat, conf, "gemini")
        return out


class CategorizerAgent:
    """Top-level Categorizer. Tries Gemini if available, else rules."""

    def __init__(self, prefer_llm: bool = True) -> None:
        self.rules = RulesBackend()
        self.gemini: Optional[GeminiBackend] = None
        if prefer_llm:
            backend = GeminiBackend()
            if backend.available:
                self.gemini = backend

    @property
    def active_backend(self) -> str:
        return self.gemini.name if self.gemini else self.rules.name

    def classify(self, description: str, amount: float) -> Categorization:
        if self.gemini:
            try:
                return self.gemini.classify_batch([description], [amount])[0]
            except Exception:  # noqa: BLE001
                pass  # graceful fallback
        return self.rules.classify_one(description, amount)

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add `category` and `ai_confidence_score` columns in-place-ish.

        Returns a new DataFrame; the original is not mutated.
        """
        if df.empty:
            return df.copy()

        out = df.copy()
        cats: list[str] = []
        confs: list[float] = []

        if self.gemini:
            try:
                results = self.gemini.classify_batch(
                    out["description"].astype(str).tolist(),
                    [float(a) for a in out["amount"]],
                )
                cats = [r.category for r in results]
                confs = [r.confidence for r in results]
            except Exception:  # noqa: BLE001
                cats, confs = [], []

        if not cats:
            for desc, amt in zip(out["description"], out["amount"]):
                r = self.rules.classify_one(str(desc), float(amt))
                cats.append(r.category)
                confs.append(r.confidence)

        out["category"] = cats
        out["ai_confidence_score"] = confs
        return out
