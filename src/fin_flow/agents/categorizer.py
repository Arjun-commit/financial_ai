"""Categorizer Agent — assigns IRS / Schedule-C-aligned tax categories."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from ..utils.pii import mask_pii

logger = logging.getLogger(__name__)

# IRS Schedule C-style buckets. Order matters: more specific first.
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
    "Foods": "Groceries",
    "whole foods": "Groceries",
    "trader joe": "Groceries",
    "safeway": "Groceries",
    "kroger": "Groceries",
    "costco": "Groceries",
    "frys": "Groceries",
    "fry's": "Groceries",
    "food and drug": "Groceries",
    "albertsons": "Groceries",
    "sprouts": "Groceries",
    "aldi": "Groceries",
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
    "Bank Payment": "Transfers",
    # convenience / gas
    "circle k": "Shopping",
    "7-eleven": "Shopping",
    "wawa": "Shopping",
    "cvs": "Shopping",
    "walgreens": "Shopping",
    # ride share
    "lime": "Travel",
    "bird": "Travel",
    "scooter": "Travel",
    # mobile / telecom
    "tello": "Software & Subscriptions",
    "t-mobile": "Software & Subscriptions",
    "mint mobile": "Software & Subscriptions",
    # rent
    "bilt": "Rent",
    "biltrent": "Rent",
    # gas stations
    "shell oil": "Shopping",
    "chevron": "Shopping",
    "exxon": "Shopping",
    "bp gas": "Shopping",
    # common bank descriptors
    "autopay": "Transfers",
    "credit crd": "Transfers",
    "atm cash deposit": "Income",
    "atm deposit": "Income",
    "atm withdraw": "Transfers",
    "non-chase atm": "Transfers",
}


@dataclass
class Categorization:
    category: str
    confidence: float  # 0.0 - 1.0
    rationale: str = ""


@dataclass
class DataQuality:
    """Categorization quality stats for a classified DataFrame."""

    total: int
    uncategorized_count: int
    uncategorized_pct: float  # 0-100
    needs_review_count: int
    needs_review_threshold: float


class RulesBackend:
    name = "rules"

    def __init__(self, lexicon: Optional[dict[str, str]] = None) -> None:
        self.lexicon = lexicon or _KEYWORD_LEXICON

    def classify_one(self, description: str, amount: float) -> Categorization:
        text = (description or "").lower()
        scores: dict[str, float] = {}
        for keyword, category in self.lexicon.items():
            if keyword in text:
                scores[category] = scores.get(category, 0.0) + len(keyword)

        if not scores:
            if amount > 0:
                return Categorization("Income", 0.55, "positive amount, no keyword match")
            return Categorization("Uncategorized", 0.30, "no keyword matched")

        best_cat, best_score = max(scores.items(), key=lambda kv: kv[1])
        total = sum(scores.values())
        confidence = min(0.99, 0.55 + (best_score / total) * 0.4)
        return Categorization(best_cat, round(confidence, 3), f"matched keywords -> {best_cat}")


class GeminiBackend:
    name = "gemini"
    _SYSTEM_PROMPT = (
        "Classify each bank transaction into exactly one category.\n\n"
        "Categories: {categories}\n\n"
        "Guidelines:\n"
        "- Positive amounts are usually Income or Transfers.\n"
        "- Zelle/Venmo/CashApp payments are Transfers.\n"
        "- ATM deposits are Income. ATM withdrawals are Transfers.\n"
        "- Credit card autopay, loan payments are Transfers.\n"
        "- Gas stations, convenience stores (Circle K, 7-Eleven, Shell) are Shopping.\n"
        "- Ride shares (Uber, Lyft, Lime) are Travel.\n"
        "- Mobile plans, streaming, SaaS are Software & Subscriptions.\n"
        "- Rent payments (Bilt, lease) are Rent.\n"
        "- Grocery/pharmacy chains (Fry's, Safeway, Kroger, CVS) are Groceries.\n"
        "- Only use Uncategorized if you truly cannot determine the type.\n\n"
        "Examples of messy real bank descriptions and their categories:\n"
        '- "Frys Food And Drug 64 Tucson AZ" -> Groceries\n'
        '- "Non-Chase ATM Withdraw 05/20 W University Tucson AZ" -> Transfers\n'
        '- "Shell Oil 57442706 Tucson AZ" -> Shopping\n'
        '- "Recurring Card Purchase Tello Mobile Tello.Com GA" -> Software & Subscriptions\n'
        '- "Chase Credit Crd Autopay PPD ID: 4760039224" -> Transfers\n\n'
        "Return ONLY a JSON array of objects with keys: "
        "index (int), category (string), confidence (float 0-1). "
        "No markdown fences, no explanation."
    )

    def __init__(self, model_name: str = "gemini-2.5-flash-lite") -> None:
        self.model_name = model_name
        self._client = None
        self._init_error: str = ""
        try:
            from google import genai  # type: ignore

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
            logger.info("GeminiBackend initialized (model=%s)", model_name)
        except Exception as e:  # noqa: BLE001
            self._init_error = str(e)
            logger.warning("GeminiBackend init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._client is not None

    def classify_batch(
        self, descriptions: list[str], amounts: list[float],
        _max_retries: int = 2,
    ) -> list[Categorization]:
        if not self.available:
            raise RuntimeError(f"GeminiBackend not available: {self._init_error}")
        prompt = self._SYSTEM_PROMPT.format(categories=", ".join(TAX_CATEGORIES))
        rows = [
            {"index": i, "description": mask_pii(d), "amount": float(a)}
            for i, (d, a) in enumerate(zip(descriptions, amounts))
        ]
        body = prompt + "\n\nTransactions:\n" + json.dumps(rows)

        # Retry on transient errors (429 / 503) with exponential backoff
        _RETRYABLE = ("429", "503")
        last_err: Optional[Exception] = None
        for attempt in range(_max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_name,
                    contents=body,
                )
                break  # success
            except Exception as e:  # noqa: BLE001
                last_err = e
                err_str = str(e)
                if any(code in err_str for code in _RETRYABLE) and attempt < _max_retries:
                    wait = 2 ** attempt * 5  # 5s, 10s
                    logger.warning(
                        "Gemini transient error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, _max_retries + 1, wait, e,
                    )
                    time.sleep(wait)
                    continue
                raise  # non-retryable or out of retries

        text = (resp.text or "").strip()
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
    def __init__(self, prefer_llm: bool = True) -> None:
        self.rules = RulesBackend()
        self.gemini: Optional[GeminiBackend] = None
        self.last_backend_used: str = "rules"
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
            except Exception as e:  # noqa: BLE001
                logger.error("Gemini classify() failed, falling back to rules: %s", e)
        return self.rules.classify_one(description, amount)

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
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
                self.last_backend_used = "gemini"
            except Exception as e:  # noqa: BLE001
                logger.error("Gemini batch classification failed, falling back to rules: %s", e)
                cats, confs = [], []

        if not cats:
            self.last_backend_used = "rules"
            for desc, amt in zip(out["description"], out["amount"]):
                r = self.rules.classify_one(str(desc), float(amt))
                cats.append(r.category)
                confs.append(r.confidence)

        out["category"] = cats
        out["ai_confidence_score"] = confs
        return out

    @staticmethod
    def data_quality(
        df: pd.DataFrame, threshold: float = 0.6
    ) -> DataQuality:
        """Compute categorization quality metrics."""
        if df.empty:
            return DataQuality(0, 0, 0.0, 0, threshold)
        total = len(df)
        uncat = int((df["category"] == "Uncategorized").sum())
        uncat_pct = (uncat / total) * 100 if total > 0 else 0.0
        scores = df["ai_confidence_score"].map(lambda x: float(x or 0))
        low_conf = int((scores < threshold).sum())
        return DataQuality(total, uncat, round(uncat_pct, 1), low_conf, threshold)