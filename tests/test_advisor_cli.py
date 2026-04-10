import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fin_flow.advisor_cli import run
from fin_flow.agents import CategorizerAgent
from fin_flow.ingestion import load_file

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "data" / "samples"
OUT_DIR = REPO / "data" / "processed"


def _prepare_categorized_csv() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_file(SAMPLES / "chase_sample.csv")
    df = CategorizerAgent(prefer_llm=False).classify_dataframe(df)
    path = OUT_DIR / "test_advisor_cli_txn.csv"
    df.to_csv(path, index=False)
    return path


def test_advisor_cli_note_and_ask_roundtrip():
    txn_path = _prepare_categorized_csv()
    notes_path = OUT_DIR / "test_advisor_cli_notes.json"

    rc = run(
        [
            "note",
            "--text",
            "We plan to scale marketing spend by 20% in Q3.",
            "--type",
            "strategic_goal",
            "--priority",
            "high",
            "--store",
            str(notes_path),
        ]
    )
    assert rc == 0
    assert notes_path.exists()

    # Ask a category-spend question and capture JSON output
    buf = StringIO()
    with patch("sys.stdout", buf):
        rc = run(
            [
                "ask",
                "How much did I spend on Meals?",
                "--transactions",
                str(txn_path),
                "--store",
                str(notes_path),
                "--no-llm",
                "--json",
            ]
        )
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["intent"] == "category_spend"
    assert payload["citations"]
    assert payload["backend"] == "rules"
