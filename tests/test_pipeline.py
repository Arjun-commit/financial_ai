from pathlib import Path

import pandas as pd

from fin_flow.pipeline import run_pipeline, run

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "data" / "samples"
OUT_DIR = REPO / "data" / "processed"


def test_pipeline_python_api():
    result = run_pipeline(
        [SAMPLES / "chase_sample.csv", SAMPLES / "bofa_sample.csv"],
        starting_balance=10_000.0,
        horizon_days=60,
        prefer_llm=False,
    )
    df = result["transactions"]
    fc = result["forecast"]

    assert len(df) == 12  # 7 chase + 5 bofa
    assert "category" in df.columns
    assert df["category"].notna().all()
    assert result["categorizer_backend"] == "rules"
    # Forecast should produce a 60-day projection
    assert len(fc.projection) == 60
    # Either a death date or "positive" summary
    assert isinstance(fc.summary(), str)


def test_pipeline_cli_writes_outputs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rc = run(
        [
            str(SAMPLES / "chase_sample.csv"),
            "--starting-balance",
            "5000",
            "--horizon",
            "30",
            "--out",
            str(OUT_DIR),
            "--no-llm",
        ]
    )
    assert rc == 0
    txn = pd.read_csv(OUT_DIR / "transactions_categorized.csv")
    fc = pd.read_csv(OUT_DIR / "forecast.csv")
    assert "category" in txn.columns
    assert len(fc) == 30
    assert "projected_balance" in fc.columns
