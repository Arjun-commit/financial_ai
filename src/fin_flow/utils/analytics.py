"""Lightweight usage analytics - JSON events to stdout.

NEVER log question text, descriptions, amounts, or emails.
A logging failure must never break the user experience.

Expected events (not enforced):
  session_start, file_uploaded (row_count), sample_data_used,
  gemini_called (chunks, backend), chat_question (intent, backend),
  yoy_viewed, tax_report_generated (year),
  tax_download_clicked (format), email_captured
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

_session_id: Optional[str] = None


def _get_session_id() -> str:
    global _session_id
    if _session_id is None:
        _session_id = uuid.uuid4().hex
    return _session_id


def log_event(event: str, **fields) -> None:
    """Print a single-line JSON event to stdout. Never raises."""
    try:
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": event,
            "session_id": _get_session_id(),
            **fields,
        }
        print(json.dumps(record), flush=True)
    except Exception:  # noqa: BLE001
        pass


def submit_email(email: str, endpoint: str) -> bool:
    """POST email to Formspree (or similar). Returns True on 2xx."""
    try:
        import requests  # lazy import

        resp = requests.post(endpoint, json={"email": email}, timeout=5)
        if resp.ok:
            log_event("email_captured")
            return True
        return False
    except Exception:  # noqa: BLE001
        return False


def save_email_locally(email: str, path: str) -> bool:
    """Append email to a local JSON list file. Never raises.

    Used as a fallback when FORMSPREE_ENDPOINT is not configured
    so that emails collected during local development are not
    silently discarded.
    """
    try:
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        entries: list[dict] = []
        if p.exists():
            entries = json.loads(p.read_text(encoding="utf-8"))

        entries.append({
            "email": email,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        p.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        log_event("email_captured")
        return True
    except Exception:  # noqa: BLE001
        return False
