# Fin-Flow Agent (CFO-AI)

Autonomous multi-agent AI system that acts as a virtual CFO ingests raw financial data, categorizes expenses, forecasts cash flow, and answers strategic questions.

## Phases 1 + 2 - Ingestion, Categorization, Forecasting (current)

Phase 1 ships the data plumbing: file parsers, canonical schema, deduplication, and PII masking. Phase 2 adds the agentic brain: a tax-aligned Categorizer and a runway Forecaster, both with a zero-dependency default backend and an optional upgrade path (Gemini for categorization, Prophet for forecasting).

### Quick start

```bash
pip install -r requirements.txt

# Phase 1 only - clean and dedupe a bank export
python -m fin_flow.ingestion.cli data/samples/chase_sample.csv --out data/processed/clean.csv

# Phase 1 + 2 - full pipeline: ingest -> categorize -> forecast
python -m fin_flow.pipeline data/samples/chase_sample.csv data/samples/bofa_sample.csv \
    --starting-balance 8000 --horizon 60 --out data/processed --no-llm

# Run the test suite
pytest -q
```

### Phase 2 deliverables

- `CategorizerAgent` - assigns one of 18 IRS Schedule-C-style categories (Income, Rent, Software & Subscriptions, Meals, Travel, Payroll, …). Default backend is a deterministic keyword-scoring engine; if `google-generativeai` is installed and `GEMINI_API_KEY` is set, it transparently switches to Gemini 1.5 Flash and falls back on any error.
- `ForecasterAgent` - builds a contiguous daily-net series, computes mean burn rate, and projects cash position over a configurable horizon. Returns a `Forecast` with `death_date`, `mean_daily_burn`, and a full projection DataFrame. Default backend is linear; upgrades to Prophet automatically if the `prophet` package is installed.
- `pipeline.run_pipeline(...)` - Python API and CLI that chains ingest → dedupe → categorize → forecast and writes `transactions_categorized.csv` + `forecast.csv`.

### Project layout

```
src/fin_flow/
  ingestion/    # CSV / Excel / JSON parsers, schema normalizer, CLI
  agents/       # Categorizer (rules + Gemini), Forecaster (linear + Prophet)
  storage/      # (Phase 3) Supabase + ChromaDB clients
  utils/        # PII masking, logging, hashing
  pipeline.py   # high-level Phase 1+2 orchestrator (Python API + CLI)
tests/          # pytest suites
data/
  samples/      # example bank exports
  processed/    # normalized output
config/         # schema maps per bank
```

### What Phase 1 delivers

- Robust CSV ingestion that tolerates different bank export formats (Chase, Bank of America, Wells Fargo, generic)
- A canonical `Transaction` schema (date, amount, description, source, raw_hash)
- Deduplication via content-hash so re-uploading the same statement is safe
- PII masking utilities ready for Phase 2's LLM calls
- A CLI and a Python API, both tested

Later phases add the RAG-backed Advisor agent, ChromaDB-powered vector memory of business context, the Supabase storage layer, and the Streamlit dashboard (see the technical spec).
