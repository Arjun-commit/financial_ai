"""High-level pipeline that wires Phase 1 + Phase 2 together.

Usage from Python:
    from fin_flow.pipeline import run_pipeline
    result = run_pipeline(["data/samples/chase_sample.csv"], starting_balance=10_000)
    print(result["forecast"].summary())

Usage from CLI:
    python -m fin_flow.pipeline data/samples/chase_sample.csv \
        --starting-balance 10000 --horizon 60 --out data/processed/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from .agents import CategorizerAgent, ForecasterAgent
from .ingestion import deduplicate, load_file
from .ingestion.normalizer import IngestionError


def run_pipeline(
    inputs: Iterable[str | Path],
    starting_balance: float = 0.0,
    horizon_days: int = 90,
    prefer_llm: bool = True,
) -> dict:
    """Ingest -> Categorize -> Forecast and return everything as a dict."""
    frames: list[pd.DataFrame] = []
    for raw_path in inputs:
        frames.append(load_file(raw_path))
    if not frames:
        raise IngestionError("no input files provided")

    transactions = deduplicate(pd.concat(frames, ignore_index=True))

    categorizer = CategorizerAgent(prefer_llm=prefer_llm)
    categorized = categorizer.classify_dataframe(transactions)

    forecaster = ForecasterAgent()
    forecast = forecaster.forecast(
        categorized,
        starting_balance=starting_balance,
        horizon_days=horizon_days,
    )

    return {
        "transactions": categorized,
        "forecast": forecast,
        "categorizer_backend": categorizer.active_backend,
        "forecaster_backend": forecaster.active_backend,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fin-flow-pipeline",
        description="Run the full Fin-Flow Phase 1 + Phase 2 pipeline.",
    )
    p.add_argument("inputs", nargs="+", help="CSV/XLSX/JSON bank exports.")
    p.add_argument("--starting-balance", type=float, default=0.0)
    p.add_argument("--horizon", type=int, default=90, help="Forecast horizon in days")
    p.add_argument(
        "--out",
        default="data/processed",
        help="Output directory (categorized transactions + forecast CSVs).",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Force the deterministic rules backend even if Gemini is configured.",
    )
    return p


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_pipeline(
            args.inputs,
            starting_balance=args.starting_balance,
            horizon_days=args.horizon,
            prefer_llm=not args.no_llm,
        )
    except IngestionError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    txn_path = out_dir / "transactions_categorized.csv"
    fc_path = out_dir / "forecast.csv"

    result["transactions"].to_csv(txn_path, index=False)
    result["forecast"].projection.to_csv(fc_path, index=False)

    print(f"[ok] {len(result['transactions'])} transactions categorized "
          f"using `{result['categorizer_backend']}` backend -> {txn_path}")
    print(f"[ok] {result['forecaster_backend']} forecast -> {fc_path}")
    print()
    print(result["forecast"].summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
