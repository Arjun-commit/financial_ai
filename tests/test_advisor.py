from pathlib import Path
from unittest.mock import MagicMock

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


# ── Existing rules-fallback tests (unchanged) ───────────────────────────


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


# ── Context payload tests ────────────────────────────────────────────────


def test_context_payload_structure():
    """Context payload has all required fields for Gemini."""
    df = _categorized_chase()
    advisor = _advisor()
    notes = advisor.store.query("test", k=4)
    payload = advisor._build_context_payload(df, 5000.0, notes)

    assert "summary" in payload
    assert "spending_by_category" in payload
    assert "top_expenses" in payload
    assert "notes" in payload

    summary = payload["summary"]
    assert summary["total_transactions"] == 7
    assert "date_range" in summary
    assert "net_cashflow_30d" in summary
    assert "avg_daily_income" in summary
    assert "avg_daily_burn" in summary
    assert "starting_balance" in summary
    assert "runway" in summary

    # Category totals should be positive (absolute expense values)
    for cat, amt in payload["spending_by_category"].items():
        assert amt > 0, f"category {cat} has non-positive amount"

    # Top expenses should have IDs for citation grounding
    for tx in payload["top_expenses"]:
        assert "id" in tx
        assert "amount" in tx
        assert "category" in tx
        assert "description" in tx


def test_context_payload_category_totals_match_data():
    """Category spend totals are consistent with raw data."""
    df = _categorized_chase()
    advisor = _advisor()
    payload = advisor._build_context_payload(df, 0.0, [])

    # Rent should be 1850.00 (single rent payment in chase_sample)
    assert "Rent" in payload["spending_by_category"]
    assert payload["spending_by_category"]["Rent"] == 1850.0

    # Meals should be 5.75 (Starbucks)
    assert "Meals" in payload["spending_by_category"]
    assert payload["spending_by_category"]["Meals"] == 5.75


def test_context_payload_top_expenses_sorted():
    """Top expenses are sorted by amount descending."""
    df = _categorized_chase()
    advisor = _advisor()
    payload = advisor._build_context_payload(df, 0.0, [])
    expenses = payload["top_expenses"]
    amounts = [abs(t["amount"]) for t in expenses]
    assert amounts == sorted(amounts, reverse=True)


# ── Gemini primary path tests (mocked) ───────────────────────────────────


def _advisor_with_mock_gemini(mock_response_text: str) -> AdvisorAgent:
    """Create an advisor with a mocked Gemini client."""
    store = InMemoryVectorStore(embedder=HashingEmbedder(dim=128))
    advisor = AdvisorAgent(vector_store=store, prefer_llm=False)
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = mock_response_text
    mock_client.models.generate_content.return_value = mock_resp
    advisor._client = mock_client
    return advisor


def test_gemini_primary_path():
    """When Gemini is available, it answers directly (not just rewrites)."""
    df = _categorized_chase()
    advisor = _advisor_with_mock_gemini(
        "Your total spending over the last 30 days is $2,019.67. "
        "Rent is your largest expense at $1,850.00, followed by "
        "groceries at $87.33 and shopping at $42.18."
    )
    ans = advisor.ask("What's my financial situation?", df, starting_balance=5000.0)

    assert ans.backend == "gemini"
    assert ans.intent == "gemini"
    assert "$" in ans.answer
    assert "1,850" in ans.answer
    assert ans.citations
    # All citations must be real hashes
    hashes = set(df["raw_hash"])
    for c in ans.citations:
        assert c in hashes


def test_gemini_receives_context_payload():
    """Verify Gemini is called with a prompt containing the context data."""
    df = _categorized_chase()
    advisor = _advisor_with_mock_gemini("Great question. Here's your answer.")
    advisor.ask("How much did I spend?", df, starting_balance=1000.0)

    call_args = advisor._client.models.generate_content.call_args
    prompt = call_args.kwargs.get("contents") or call_args[1].get("contents", "")
    # Prompt should contain the structured context
    assert "spending_by_category" in prompt
    assert "top_expenses" in prompt
    assert "net_cashflow_30d" in prompt
    assert "How much did I spend?" in prompt


def test_gemini_multipart_question():
    """Multi-part questions get a single conversational answer."""
    df = _categorized_chase()
    advisor = _advisor_with_mock_gemini(
        "You spent $2,019.67 total in the last 30 days. Your largest "
        "category is Rent at $1,850.00. To reduce spending, consider "
        "reviewing your Groceries ($87.33) and Shopping ($42.18) — "
        "those are the areas with the most flexibility."
    )
    ans = advisor.ask(
        "What is my spend in most categories and how do I reduce unnecessary spending?",
        df,
        starting_balance=5000.0,
    )
    assert ans.backend == "gemini"
    # Answer addresses both parts
    assert "1,850" in ans.answer
    assert "reduce" in ans.answer.lower() or "review" in ans.answer.lower()


# ── Fallback behavior tests ─────────────────────────────────────────────


def test_fallback_shows_offline_note():
    """When prefer_llm=True but Gemini unavailable, show offline note."""
    df = _categorized_chase()
    store = InMemoryVectorStore(embedder=HashingEmbedder(dim=128))
    advisor = AdvisorAgent(vector_store=store, prefer_llm=True)
    advisor._client = None  # force Gemini unavailable

    ans = advisor.ask("How much did I spend?", df, starting_balance=0.0)

    assert ans.backend == "rules"
    assert "AI advisor offline" in ans.answer
    assert "basic summary" in ans.answer
    # The actual data should still be present
    assert "$" in ans.answer


def test_fallback_no_offline_note_when_llm_not_preferred():
    """When prefer_llm=False, no offline note even without Gemini."""
    df = _categorized_chase()
    advisor = _advisor()  # prefer_llm=False
    ans = advisor.ask("How much did I spend?", df, starting_balance=0.0)

    assert ans.backend == "rules"
    assert "AI advisor offline" not in ans.answer


def test_gemini_error_falls_back_with_note():
    """When Gemini errors, fallback runs and offline note appears."""
    df = _categorized_chase()
    store = InMemoryVectorStore(embedder=HashingEmbedder(dim=128))
    advisor = AdvisorAgent(vector_store=store, prefer_llm=True)
    # Mock client that raises
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("429 quota exceeded")
    advisor._client = mock_client

    ans = advisor.ask("How much did I spend on Meals?", df, starting_balance=0.0)

    assert ans.backend == "rules"
    assert "AI advisor offline" in ans.answer
    # Rules handler still produces accurate data
    assert "Meals" in ans.answer
    assert "5.75" in ans.answer
    # Citations are grounded
    hashes = set(df["raw_hash"])
    for c in ans.citations:
        assert c in hashes


def test_gemini_empty_response_falls_back():
    """Empty Gemini response triggers fallback."""
    df = _categorized_chase()
    advisor = _advisor_with_mock_gemini("")
    advisor.prefer_llm = True
    ans = advisor.ask("What's my runway?", df, starting_balance=100.0)

    # Empty response raises RuntimeError, triggers fallback
    assert ans.backend == "rules"
    assert "AI advisor offline" in ans.answer
