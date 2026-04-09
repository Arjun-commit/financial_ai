"""Ingestion layer — file parsers, normalizers, deduplication."""

from .schema import Transaction, CANONICAL_COLUMNS
from .normalizer import normalize_dataframe, load_file
from .dedupe import deduplicate, content_hash

__all__ = [
    "Transaction",
    "CANONICAL_COLUMNS",
    "normalize_dataframe",
    "load_file",
    "deduplicate",
    "content_hash",
]
