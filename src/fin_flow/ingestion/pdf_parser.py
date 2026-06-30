"""Extract transaction tables from bank statement PDFs."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]


# Date at line start: MM/DD/YYYY, MM/DD/YY, or just MM/DD
_DATE_FULL_RE = re.compile(r"^(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\s+(.+)")
_DATE_SHORT_RE = re.compile(r"^(\d{1,2}/\d{1,2})\s+(.+)")

_AMT_RE = re.compile(r"-?\$?[\d,]+\.\d{2}")

_HEADER_WORDS = {
    "date", "description", "withdrawals", "deposits", "balance",
    "debit", "credit", "amount", "details", "reference", "posting",
    "transaction", "memo", "check", "beginning balance", "ending balance",
    "totals", "total", "page", "statement", "account summary",
    "checking summary",
}

# Statement period patterns to extract the year
_PERIOD_RE = re.compile(
    r"(\w+ \d{1,2},?\s*\d{4})\s*(?:through|to|thru|-|–)\s*(\w+ \d{1,2},?\s*\d{4})",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _is_junk_line(text: str) -> bool:
    lower = text.strip().lower()
    if not lower or len(lower) < 3:
        return True
    if re.match(r"^[\d\s]+$", lower):
        return True
    if lower in ("(continued)",):
        return True
    return False


def _strip_section_markers(line: str) -> str:
    """Remove *start*/*end* markers and salvage any transaction text after them.

    Real PDFs sometimes corrupt these markers (e.g. '*end*transac0tion detail6/05 ...')
    so we strip everything before the first date-like pattern.
    """
    cleaned = re.sub(r"\*(?:start|end)\*\S*\s*", "", line)
    # If leftover junk is glued before a date, find and extract from the date onward
    m = re.search(r"(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+", cleaned)
    if m and m.start() > 0:
        cleaned = cleaned[m.start():]
    return cleaned.strip()


def _is_header_line(text: str) -> bool:
    lower = text.strip().lower()
    tokens = [t for t in re.split(r"\s+", lower) if t]
    if not tokens or len(lower) > 80:
        return False
    # Only treat as header if majority of words are header words
    header_count = sum(1 for t in tokens if t in _HEADER_WORDS)
    return header_count >= len(tokens) * 0.5 and len(tokens) <= 8


def _extract_year(full_text: str) -> Optional[int]:
    """Pull the statement year from the period header."""
    m = _PERIOD_RE.search(full_text)
    if m:
        try:
            end_date = datetime.strptime(m.group(2).replace(",", ""), "%B %d %Y")
            return end_date.year
        except ValueError:
            pass
    # fallback: grab any 4-digit year
    years = _YEAR_RE.findall(full_text[:1000])
    if years:
        return int(years[-1])
    return None


def _extract_statement_period(full_text: str) -> tuple[Optional[datetime], Optional[datetime]]:
    m = _PERIOD_RE.search(full_text)
    if not m:
        return None, None
    try:
        start = datetime.strptime(m.group(1).replace(",", ""), "%B %d %Y")
        end = datetime.strptime(m.group(2).replace(",", ""), "%B %d %Y")
        return start, end
    except ValueError:
        return None, None


def _resolve_date(short_date: str, year: int, period_start=None, period_end=None) -> str:
    """Turn 'MM/DD' into 'MM/DD/YYYY', handling year boundaries."""
    parts = short_date.split("/")
    if len(parts) == 2:
        month, day = int(parts[0]), int(parts[1])
        # If the statement spans Dec-Jan, dates in Jan get the next year
        if period_start and period_end and period_end.year > period_start.year:
            if month <= period_end.month:
                return f"{month:02d}/{day:02d}/{period_end.year}"
        return f"{month:02d}/{day:02d}/{year}"
    return short_date  # already has year


def _try_table_extraction(pdf) -> Optional[pd.DataFrame]:
    all_rows = []
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            if not table:
                continue
            for row in table:
                cleaned = [str(cell).strip() if cell else "" for cell in row]
                if any(cleaned):
                    all_rows.append(cleaned)

    if len(all_rows) < 2:
        return None

    header_idx = 0
    for i, row in enumerate(all_rows[:5]):
        alpha_count = sum(1 for cell in row if cell and re.match(r"[A-Za-z]", cell))
        if alpha_count >= 2:
            header_idx = i
            break

    header = all_rows[header_idx]
    data = all_rows[header_idx + 1:]
    if not data:
        return None

    ncols = len(header)
    data = [r + [""] * (ncols - len(r)) if len(r) < ncols else r[:ncols] for r in data]

    df = pd.DataFrame(data, columns=header)
    df = df[~df.apply(lambda r: _is_header_line(" ".join(str(v) for v in r)), axis=1)]
    return df if not df.empty else None


def _try_line_extraction(pdf) -> Optional[pd.DataFrame]:
    """Parse transaction lines from free-text PDF pages.

    Handles MM/DD (no year), multi-line descriptions, and
    amount+balance on the same line.
    """
    full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    year = _extract_year(full_text)
    period_start, period_end = _extract_statement_period(full_text)

    if not year:
        year = datetime.now().year

    # Collect all lines across pages
    all_lines = []
    for page in pdf.pages:
        text = page.extract_text() or ""
        all_lines.extend(text.split("\n"))

    # Find transaction section boundaries
    in_txn_section = False
    txn_lines: list[str] = []
    for line in all_lines:
        lower = line.strip().lower()
        has_start = "*start*transaction" in lower
        has_end = "*end*transaction" in lower or "*end*transac" in lower

        if has_start:
            in_txn_section = True
            # In case a transaction is glued onto the marker line
            cleaned = _strip_section_markers(line)
            if cleaned and _DATE_SHORT_RE.match(cleaned):
                txn_lines.append(cleaned)
            continue
        if has_end:
            # Salvage any transaction text stuck to the end marker
            cleaned = _strip_section_markers(line)
            if cleaned and (_DATE_SHORT_RE.match(cleaned) or _DATE_FULL_RE.match(cleaned)):
                txn_lines.append(cleaned)
            in_txn_section = False
            continue
        if in_txn_section:
            txn_lines.append(line.strip())

    if not txn_lines:
        txn_lines = [l.strip() for l in all_lines]

    # Parse: group lines into transactions
    # A new transaction starts with a date; continuation lines get
    # appended to the previous transaction's description
    raw_txns: list[dict] = []

    for line in txn_lines:
        if _is_junk_line(line) or _is_header_line(line):
            continue
        if "beginning balance" in line.lower() or "ending balance" in line.lower():
            continue

        # Try full date first (MM/DD/YYYY), then short (MM/DD)
        m = _DATE_FULL_RE.match(line)
        if m:
            date_str = m.group(1)
            rest = m.group(2)
        else:
            m = _DATE_SHORT_RE.match(line)
            if m:
                date_str = _resolve_date(m.group(1), year, period_start, period_end)
                rest = m.group(2)
            else:
                # Continuation line — append to previous txn description
                if raw_txns and line.strip():
                    amounts_here = _AMT_RE.findall(line)
                    if not amounts_here:
                        raw_txns[-1]["_desc_parts"].append(line.strip())
                continue

        amounts = _AMT_RE.findall(rest)
        desc = rest
        for amt in amounts:
            desc = desc.replace(amt, "", 1)
        desc = re.sub(r"\s{2,}", " ", desc).strip()

        raw_txns.append({
            "date": date_str,
            "_desc_parts": [desc] if desc else [],
            "amounts": amounts,
        })

    if not raw_txns:
        return None

    # Build DataFrame rows
    rows = []
    for t in raw_txns:
        desc = " ".join(t["_desc_parts"]).strip()
        if not desc or not t["amounts"]:
            continue

        # First amount is the transaction amount,
        # second (if present) is the running balance — drop it
        amt_str = t["amounts"][0]
        rows.append({
            "Date": t["date"],
            "Description": desc,
            "Amount": amt_str,
        })

    return pd.DataFrame(rows) if rows else None


def parse_pdf(path: str | Path) -> pd.DataFrame:
    from .normalizer import IngestionError

    if pdfplumber is None:
        raise IngestionError(
            "PDF support requires pdfplumber: pip install pdfplumber"
        )

    p = Path(path)
    with pdfplumber.open(p) as pdf:
        df = _try_table_extraction(pdf)
        if df is not None and len(df) > 0:
            return df

        df = _try_line_extraction(pdf)
        if df is not None and len(df) > 0:
            return df

    raise IngestionError(
        f"Could not extract transactions from {p.name}. "
        "The PDF may be scanned/image-based (OCR not supported) "
        "or use a layout this parser doesn't handle yet."
    )
