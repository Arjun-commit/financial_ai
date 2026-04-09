"""Command-line entry point for Phase 1 ingestion.

Usage:
    python -m fin_flow.ingestion.cli INPUT [INPUT ...] --out OUTPUT.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .dedupe import deduplicate
from .normalizer import IngestionError, load_file
from ..utils.pii import mask_series


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fin-flow-ingest",
        description="Normalize and deduplicate bank export files.",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="One or more CSV/XLSX/JSON files to ingest.",
    )
    p.add_argument(
        "--out",
        "-o",
        default="data/processed/clean.csv",
        help="Output CSV path (default: data/processed/clean.csv).",
    )
    p.add_argument(
        "--mask-pii",
        action="store_true",
        help="Also write a PII-masked description column for LLM use.",
    )
    return p


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    frames: list[pd.DataFrame] = []
    for raw_path in args.inputs:
        try:
            df = load_file(raw_path)
        except IngestionError as e:
            print(f"[error] {raw_path}: {e}", file=sys.stderr)
            return 2
        print(f"[ok]    {raw_path}: {len(df)} rows")
        frames.append(df)

    if not frames:
        print("[error] no input files processed", file=sys.stderr)
        return 2

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = deduplicate(combined)
    removed = before - len(combined)
    print(f"[dedupe] removed {removed} duplicate rows, {len(combined)} remain")

    if args.mask_pii:
        combined["description_masked"] = mask_series(combined["description"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"[write] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
