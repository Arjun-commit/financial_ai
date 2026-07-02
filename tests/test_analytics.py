"""Tests for utils/analytics.py - silent failure, event format."""

import json
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from fin_flow.utils.analytics import log_event, submit_email, save_email_locally, _get_session_id


# ── log_event ───────────────────────────────────────────────────────────


def test_log_event_prints_json(capsys):
    log_event("test_event", foo="bar")
    captured = capsys.readouterr().out.strip()
    record = json.loads(captured)
    assert record["event"] == "test_event"
    assert record["foo"] == "bar"
    assert "ts" in record
    assert "session_id" in record


def test_log_event_has_timestamp(capsys):
    log_event("ts_check")
    record = json.loads(capsys.readouterr().out.strip())
    ts = record["ts"]
    assert ts.endswith("Z")
    assert "T" in ts


def test_log_event_session_id_stable():
    """Same session_id across multiple calls."""
    sid1 = _get_session_id()
    sid2 = _get_session_id()
    assert sid1 == sid2
    assert len(sid1) == 32  # uuid4 hex


def test_log_event_never_raises():
    """Even with bad arguments, log_event must not raise."""
    # Pass something that's not JSON serializable
    class BadObj:
        pass

    try:
        log_event("bad_event", data=BadObj())
    except Exception:
        pytest.fail("log_event raised an exception - it must never raise")


def test_log_event_extra_fields(capsys):
    log_event("upload", row_count=42, backend="gemini")
    record = json.loads(capsys.readouterr().out.strip())
    assert record["row_count"] == 42
    assert record["backend"] == "gemini"


# ── submit_email ────────────────────────────────────────────────────────


def test_submit_email_success(capsys):
    mock_resp = MagicMock()
    mock_resp.ok = True

    with patch("fin_flow.utils.analytics.requests", create=True) as mock_requests:
        # Need to patch at import time since it's a lazy import
        pass

    # Test via the actual function with mocked requests
    import fin_flow.utils.analytics as analytics_mod

    with patch.dict("sys.modules", {"requests": MagicMock()}):
        import importlib
        mock_req = sys.modules["requests"]
        mock_req.post.return_value = mock_resp

        # Re-run the function
        result = submit_email("test@example.com", "https://formspree.io/f/test")
        assert result is True


def test_submit_email_failure():
    mock_resp = MagicMock()
    mock_resp.ok = False

    with patch.dict("sys.modules", {"requests": MagicMock()}):
        mock_req = sys.modules["requests"]
        mock_req.post.return_value = mock_resp
        result = submit_email("test@example.com", "https://formspree.io/f/test")
        assert result is False


def test_submit_email_network_error():
    """Network errors return False, never raise."""
    with patch.dict("sys.modules", {"requests": MagicMock()}):
        mock_req = sys.modules["requests"]
        mock_req.post.side_effect = ConnectionError("network down")
        result = submit_email("test@example.com", "https://formspree.io/f/test")
        assert result is False


def test_submit_email_no_requests_module():
    """If requests isn't installed, submit_email returns False."""
    with patch.dict("sys.modules", {"requests": None}):
        # This should trigger ImportError inside the function
        result = submit_email("test@example.com", "https://example.com")
        assert result is False


# ── save_email_locally ─────────────────────────────────────────────────

import tempfile
import shutil


@pytest.fixture
def _tmp_dir():
    """Create a temp dir under /tmp to avoid mounted-fs permission issues."""
    d = tempfile.mkdtemp(prefix="finflow_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_save_email_locally_creates_file(_tmp_dir):
    """First call creates the JSON file with one entry."""
    p = str(_tmp_dir / "subs.json")
    assert save_email_locally("alice@example.com", p) is True

    entries = json.loads(Path(p).read_text())
    assert len(entries) == 1
    assert entries[0]["email"] == "alice@example.com"
    assert "ts" in entries[0]


def test_save_email_locally_appends(_tmp_dir):
    """Subsequent calls append without overwriting."""
    p = str(_tmp_dir / "subs.json")
    save_email_locally("alice@example.com", p)
    save_email_locally("bob@example.com", p)

    entries = json.loads(Path(p).read_text())
    assert len(entries) == 2
    assert entries[0]["email"] == "alice@example.com"
    assert entries[1]["email"] == "bob@example.com"


def test_save_email_locally_creates_parent_dirs(_tmp_dir):
    """Missing parent directories are created automatically."""
    p = str(_tmp_dir / "deep" / "nested" / "subs.json")
    assert save_email_locally("deep@example.com", p) is True
    assert Path(p).exists()


def test_save_email_locally_never_raises(_tmp_dir):
    """Write failure returns False, never raises."""
    bad_path = str(_tmp_dir / "dir_not_file")
    os.makedirs(bad_path, exist_ok=True)
    bad_path = bad_path + "/"
    try:
        result = save_email_locally("fail@example.com", bad_path)
        assert result is False
    except Exception:
        pytest.fail("save_email_locally raised - it must never raise")
