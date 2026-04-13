"""Canonical transaction schema used across Fin-Flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional

CANONICAL_COLUMNS = [
    "transaction_date",
    "amount",
    "description",
    "source",
    "raw_hash",
    "category",
    "ai_confidence_score",
]


@dataclass
class Transaction:
    transaction_date: date
    amount: Decimal
    description: str
    source: str = "unknown"
    raw_hash: str = ""
    category: Optional[str] = None
    ai_confidence_score: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "transaction_date": self.transaction_date.isoformat(),
            "amount": str(self.amount),
            "description": self.description,
            "source": self.source,
            "raw_hash": self.raw_hash,
            "category": self.category,
            "ai_confidence_score": self.ai_confidence_score,
        }
