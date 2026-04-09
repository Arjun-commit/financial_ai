from pathlib import Path

import pandas as pd

from fin_flow.ingestion.cli import run

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "data" / "samples"
OUT_DIR = REPO / "data" / "processed"


def test_cli_end_to_end():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "test_cli_output.csv"

    rc = run(
        [
            str(SAMPLES / "chase_sample.csv"),
            str(SAMPLES / "chase_sample.csv"),  # duplicate on purpose
            str(SAMPLES / "bofa_sample.csv"),
            "--out",
            str(out),
            "--mask-pii",
        ]
    )
    assert rc == 0
    assert out.exists()

    df = pd.read_csv(out)
    # Chase (7 unique) + BofA (5 unique) = 12 after dedupe
    assert len(df) == 12
    assert "description_masked" in df.columns
    # BofA row contains "555123456789" which should be masked
    assert not df["description_masked"].str.contains("555123456789").any()
    # Sanity: raw descriptions should still contain it pre-masking
    assert df["description"].str.contains("555123456789").any()
