from pathlib import Path

import pandas as pd

from fin_flow.agents import AdvisorAgent, CategorizerAgent
from fin_flow.ingestion import load_file
from fin_flow.storage import HashingEmbedder, InMemoryVectorStore

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def _categorized_chase() -> pd.DataFrame:
    df = load_file(SAMPLES / "chase_sample.csv")
    return CategorizerAgent(prefer_llm=False).classify_dataframe(df)


def _advisor() -> AdvisorAgent:
    store = InMemoryVectorStore(embedder=HashingEmbedder(dim=128))
    return AdvisorAgent(vector_store=store, prefer_llm=False)


def test_category_spend_question_grounds_citations():
    df = _categorized_chase()
    advisor = _advisor()
    ans = advisor.ask(
        "How much did I spend on Meals in the last 30 days?",
        transactions=df,
        starting_balance=0.0,
    )
    assert ans.intent == "category_spend"
    # Starbucks in the sample -> Meals category
    assert "Meals" in ans.answer
    assert "5.75" in ans.answer
    assert ans.citations, "expected at least one transaction cited"
    # Every citation must map back to an actual raw_hash in the source df
    for c in ans.citations:
        assert c in set(df["raw_hash"]), f"citation {c} not in transactions"


def test_total_spend_when_category_not_specified():
    df = _categorized_chase()
    advisor = _advisor()
    ans = advisor.ask(
        "How much did I spend recently?",
        transactions=df,
        starting_balance=0.0,
    )
    assert ans.intent == "category_spend"
    assert "last 30 days" in ans.answer.lower()
    assert ans.citations


def test_affordability_yes_and_no_paths():
    df = _categorized_chase()
    advisor = _advisor()

    cheap = advisor.ask("Can I afford a $20 lunch?", df, starting_balance=5000.0)
    assert cheap.intent == "affordability"
    assert cheap.citations

    # Pricey purchase that exceeds half of 30d net cashflow (~2180.33)
    pricey = advisor.ask("Can I afford a $4000 laptop?", df, starting_balance=500.0)
    assert pricey.intent == "affordability"
    assert "Risky" in pricey.answer or "Not advisable" in pricey.answer


def test_runway_question_uses_forecaster():
    df = _categorized_chase()
    advisor = _advisor()
    ans = advisor.ask("What's my runway?", df, starting_balance=100.0)
    assert ans.intent == "runway"
    # The chase sample is cash-flow positive (payroll deposit), so summary
    # should reflect that.
    assert "positive" in ans.answer.lower() or "runway" in ans.answer.lower()


def test_general_question_uses_retrieved_note():
    df = _categorized_chase()
    advisor = _advisor()
    advisor.add_note(
        "We plan to scale marketing spend by 20% in Q3.",
        type="strategic_goal",
        priority="high",
    )
    ans = advisor.ask("What are our Q3 marketing plans?", df)
    assert ans.intent == "general"
    assert ans.retrieved_notes
    assert ans.citations  # should include the note id


def test_citations_reference_only_real_transaction_ids():
    """Grounding contract: no hallucinated citations."""
    df = _categorized_chase()
    advisor = _advisor()
    hashes = set(df["raw_hash"])
    for q in [
        "How much did I spend on Travel?",
        "Can I afford a $100 purchase?",
        "What's my runway?",
    ]:
        ans = advisor.ask(q, df, starting_balance=1000.0)
        for c in ans.citations:
            assert c in hashes, f"{q!r} produced hallucinated citation {c}"
