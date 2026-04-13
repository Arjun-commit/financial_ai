# Fin-Flow

An autonomous financial agent system that ingests bank data, categorizes expenses, forecasts cash flow, and answers strategic questions with citations.

### Quick start

```bash
pip install -r requirements.txt

# Phase 1 only - clean and dedupe a bank export
python -m fin_flow.ingestion.cli data/samples/chase_sample.csv --out data/processed/clean.csv

# Phase 1 + 2 - full pipeline: ingest -> categorize -> forecast
python -m fin_flow.pipeline data/samples/chase_sample.csv data/samples/bofa_sample.csv \
    --starting-balance 8000 --horizon 60 --out data/processed --no-llm

# Phase 3 - add a business note, then ask a grounded question
python -m fin_flow.advisor_cli note \
    --text "We plan to scale marketing spend by 20% in Q3." \
    --type strategic_goal --priority high \
    --store data/processed/notes.json

python -m fin_flow.advisor_cli ask \
    "How much did I spend on Meals in the last 30 days?" \
    --transactions data/processed/transactions_categorized.csv \
    --starting-balance 8000 \
    --store data/processed/notes.json --no-llm

# Run the test suite
pytest -q
```

## Components

**Ingestion** — CSV/Excel/JSON parsers with auto-detection of bank export formats, canonical schema, deduplication, and PII masking.

**Categorizer** — Assigns transactions to 18 IRS Schedule-C categories. Default is rule-based keyword matching; upgrades to Gemini 1.5 Flash if configured.

**Forecaster** — Computes daily cashflow, projects runway, and estimates when cash runs out. Default is linear trend; upgrades to Prophet if installed.

**Advisor** — Retrieval-augmented Q&A over transactions. Routes questions (runway, affordability, spend by category) to handlers, composes answers with citations to real transaction IDs. Optional Gemini refinement improves fluency.

**Storage** — Pluggable vector embeddings (hashing or sentence-transformers) and persistent note storage (in-memory JSON or ChromaDB).

## Project structure

```
src/fin_flow/
  ingestion/          # parsers, schema, dedup, CLI
  agents/             # categorizer, forecaster, advisor
  storage/            # embeddings, vector stores
  utils/              # PII masking
  pipeline.py         # orchestrator
  advisor_cli.py      # grounded QA CLI
  dashboard/          # streamlit app
tests/
data/samples/         # example exports
```
