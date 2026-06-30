from pathlib import Path

import pandas as pd

from fin_flow.agents import CategorizerAgent, TAX_CATEGORIES
from fin_flow.agents.categorizer import RulesBackend
from fin_flow.ingestion import load_file

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_rules_classifies_known_merchants():
    rb = RulesBackend()
    assert rb.classify_one("STARBUCKS STORE #8842", -5.75).category == "Meals"
    assert rb.classify_one("UBER TRIP 1234ABC", -18.42).category == "Travel"
    assert rb.classify_one("WHOLE FOODS MARKET #123", -87.33).category == "Groceries"
    assert rb.classify_one("NETFLIX.COM", -15.99).category == "Software & Subscriptions"
    assert rb.classify_one("PAYROLL DEPOSIT ACME INC", 4200.0).category == "Income"
    assert rb.classify_one("AMAZON.COM AMZN.COM/BILL", -42.18).category == "Shopping"


def test_rules_unknown_falls_through_sensibly():
    rb = RulesBackend()
    # No keyword, negative amount -> Uncategorized
    assert rb.classify_one("XJ91 BLAH BLAH", -10.0).category == "Uncategorized"
    # No keyword, positive amount -> Income (heuristic)
    assert rb.classify_one("XJ91 BLAH BLAH", 100.0).category == "Income"


def test_categorizer_dataframe_round_trip():
    df = load_file(SAMPLES / "chase_sample.csv")
    agent = CategorizerAgent(prefer_llm=False)
    out = agent.classify_dataframe(df)
    assert "category" in out.columns
    assert "ai_confidence_score" in out.columns
    assert len(out) == len(df)
    # Every assigned category must be in our known taxonomy
    assert set(out["category"]).issubset(set(TAX_CATEGORIES))
    # Confidence must be in [0,1]
    assert out["ai_confidence_score"].between(0, 1).all()
    # Backend should be the deterministic rules path when LLM disabled
    assert agent.active_backend == "rules"


def test_categorizer_handles_empty_dataframe():
    agent = CategorizerAgent(prefer_llm=False)
    out = agent.classify_dataframe(pd.DataFrame())
    assert out.empty


# ── data_quality tests ───────────────────────────────────────────────

def test_data_quality_counts_uncategorized_and_low_confidence():
    df = pd.DataFrame({
        "category": ["Income", "Uncategorized", "Meals", "Uncategorized"],
        "ai_confidence_score": [0.9, 0.3, 0.7, 0.4],
    })
    dq = CategorizerAgent.data_quality(df)
    assert dq.total == 4
    assert dq.uncategorized_count == 2
    assert dq.uncategorized_pct == 50.0
    assert dq.needs_review_count == 2  # 0.3 and 0.4 both below 0.6


def test_data_quality_empty_dataframe():
    df = pd.DataFrame(columns=["category", "ai_confidence_score"])
    dq = CategorizerAgent.data_quality(df)
    assert dq.total == 0
    assert dq.uncategorized_pct == 0.0
    assert dq.needs_review_count == 0


def test_data_quality_all_high_confidence():
    df = pd.DataFrame({
        "category": ["Income", "Meals", "Travel"],
        "ai_confidence_score": [0.95, 0.8, 0.75],
    })
    dq = CategorizerAgent.data_quality(df)
    assert dq.needs_review_count == 0
    assert dq.uncategorized_count == 0
    assert dq.uncategorized_pct == 0.0


def test_data_quality_custom_threshold():
    df = pd.DataFrame({
        "category": ["Income", "Meals"],
        "ai_confidence_score": [0.85, 0.70],
    })
    dq = CategorizerAgent.data_quality(df, threshold=0.8)
    assert dq.needs_review_count == 1  # 0.70 below 0.8
    assert dq.needs_review_threshold == 0.8
