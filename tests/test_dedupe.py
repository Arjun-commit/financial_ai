from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from fin_flow.ingestion import deduplicate, load_file
from fin_flow.ingestion.dedupe import content_hash

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_content_hash_stable_across_formats():
    h1 = content_hash(date(2026, 3, 1), Decimal("-5.75"), "Starbucks #8842")
    h2 = content_hash("2026-03-01", "-5.75", "  starbucks   #8842 ")
    assert h1 == h2


def test_content_hash_differs_on_amount():
    h1 = content_hash(date(2026, 3, 1), Decimal("-5.75"), "Starbucks")
    h2 = content_hash(date(2026, 3, 1), Decimal("-5.76"), "Starbucks")
    assert h1 != h2


def test_deduplicate_same_file_twice():
    df = load_file(SAMPLES / "chase_sample.csv")
    doubled = pd.concat([df, df], ignore_index=True)
    assert len(doubled) == 2 * len(df)
    cleaned = deduplicate(doubled)
    assert len(cleaned) == len(df)
    # Arithmetic preserved after dedupe
    assert sum(cleaned["amount"]) == sum(df["amount"])
