"""Tests for chunked Gemini classification in categorizer.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from fin_flow.agents.categorizer import (
    GeminiBackend,
    Categorization,
    RulesBackend,
    TAX_CATEGORIES,
)


def _mock_gemini_response(descriptions: list[str]) -> str:
    """Build a mock JSON response matching Gemini classify format."""
    return json.dumps([
        {"index": i, "category": "Meals", "confidence": 0.9}
        for i in range(len(descriptions))
    ])


def _make_backend(mock_responses: list[str]) -> GeminiBackend:
    """Create a GeminiBackend with a mocked client that returns successive responses."""
    backend = GeminiBackend.__new__(GeminiBackend)
    backend.model_name = "gemini-2.5-flash-lite"
    backend._init_error = ""

    mock_client = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        idx = call_count[0]
        call_count[0] += 1
        resp = MagicMock()
        if idx < len(mock_responses):
            resp.text = mock_responses[idx]
        else:
            resp.text = mock_responses[-1]
        return resp

    mock_client.models.generate_content.side_effect = side_effect
    backend._client = mock_client
    return backend


# ── Chunk size enforcement ──────────────────────────────────────────────


def test_chunk_size_constant():
    assert GeminiBackend._CHUNK_SIZE == 150


def test_small_batch_single_chunk():
    """≤150 transactions should produce exactly one API call."""
    n = 50
    descs = [f"starbucks {i}" for i in range(n)]
    amounts = [-5.0] * n
    response = _mock_gemini_response(descs)
    backend = _make_backend([response])

    results = backend.classify_batch(descs, amounts)
    assert len(results) == n
    assert backend._client.models.generate_content.call_count == 1


def test_large_batch_multiple_chunks():
    """320 transactions should produce 3 chunks (150 + 150 + 20)."""
    n = 320
    descs = [f"starbucks {i}" for i in range(n)]
    amounts = [-5.0] * n

    # Each chunk returns its own response
    responses = [
        _mock_gemini_response(descs[:150]),
        _mock_gemini_response(descs[150:300]),
        _mock_gemini_response(descs[300:]),
    ]
    backend = _make_backend(responses)

    results = backend.classify_batch(descs, amounts)
    assert len(results) == n
    assert backend._client.models.generate_content.call_count == 3


def test_exact_chunk_boundary():
    """Exactly 150 transactions = 1 chunk, 151 = 2 chunks."""
    descs_150 = [f"item {i}" for i in range(150)]
    descs_151 = [f"item {i}" for i in range(151)]
    amounts_150 = [-10.0] * 150
    amounts_151 = [-10.0] * 151

    b1 = _make_backend([_mock_gemini_response(descs_150)])
    b1.classify_batch(descs_150, amounts_150)
    assert b1._client.models.generate_content.call_count == 1

    b2 = _make_backend([
        _mock_gemini_response(descs_151[:150]),
        _mock_gemini_response(descs_151[150:]),
    ])
    b2.classify_batch(descs_151, amounts_151)
    assert b2._client.models.generate_content.call_count == 2


# ── Per-chunk fallback ──────────────────────────────────────────────────


def test_per_chunk_fallback():
    """If chunk 2 fails, chunk 1 and 3 still use Gemini results."""
    n = 320  # 3 chunks: 150 + 150 + 20

    chunk1_resp = json.dumps([
        {"index": i, "category": "Travel", "confidence": 0.95}
        for i in range(150)
    ])
    chunk3_resp = json.dumps([
        {"index": i, "category": "Rent", "confidence": 0.85}
        for i in range(20)
    ])

    backend = GeminiBackend.__new__(GeminiBackend)
    backend.model_name = "gemini-2.5-flash-lite"
    backend._init_error = ""

    mock_client = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 1:  # second chunk fails
            raise RuntimeError("503 service unavailable")
        resp = MagicMock()
        resp.text = chunk1_resp if idx == 0 else chunk3_resp
        return resp

    mock_client.models.generate_content.side_effect = side_effect
    backend._client = mock_client

    descs = [f"item {i}" for i in range(n)]
    amounts = [-10.0] * n

    results = backend.classify_batch(descs, amounts, _max_retries=0)
    assert len(results) == n

    # Chunk 1: all Travel from Gemini
    for r in results[:150]:
        assert r.category == "Travel"
        assert r.rationale == "gemini"

    # Chunk 2: fell back to rules (various categories based on keyword matching)
    for r in results[150:300]:
        # Rules backend will categorize these - just check they're valid
        assert r.category in TAX_CATEGORIES

    # Chunk 3: all Rent from Gemini
    for r in results[300:]:
        assert r.category == "Rent"
        assert r.rationale == "gemini"


# ── Progress callback ───────────────────────────────────────────────────


def test_progress_callback():
    """Progress callback is called once per chunk with correct counts."""
    n = 320  # 3 chunks
    descs = [f"item {i}" for i in range(n)]
    amounts = [-10.0] * n
    responses = [
        _mock_gemini_response(descs[:150]),
        _mock_gemini_response(descs[150:300]),
        _mock_gemini_response(descs[300:]),
    ]
    backend = _make_backend(responses)

    progress_calls = []
    backend.classify_batch(
        descs, amounts,
        progress_callback=lambda done, total: progress_calls.append((done, total)),
    )
    assert progress_calls == [(1, 3), (2, 3), (3, 3)]


def test_progress_callback_error_ignored():
    """A failing progress callback doesn't break classification."""
    n = 50
    descs = [f"item {i}" for i in range(n)]
    amounts = [-10.0] * n
    backend = _make_backend([_mock_gemini_response(descs)])

    def bad_callback(done, total):
        raise ValueError("oops")

    results = backend.classify_batch(descs, amounts, progress_callback=bad_callback)
    assert len(results) == n


# ── Repairs & Maintenance category ──────────────────────────────────────


def test_repairs_maintenance_in_categories():
    assert "Repairs & Maintenance" in TAX_CATEGORIES


def test_rules_backend_repairs_keywords():
    rules = RulesBackend()
    for keyword in ("repair shop", "hvac service", "handyman joe", "home depot supplies"):
        result = rules.classify_one(keyword, -150.0)
        assert result.category == "Repairs & Maintenance", f"'{keyword}' → {result.category}"
