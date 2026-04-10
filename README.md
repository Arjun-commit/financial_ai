# Fin-Flow Agent (CFO-AI)

Autonomous multi-agent AI system that acts as a virtual CFO ingests raw financial data, categorizes expenses, forecasts cash flow, and answers strategic questions.

## Phases 1 + 2 + 3 - Ingestion, Categorization, Forecasting, RAG Advisor (current)

Phase 1 ships the data plumbing: file parsers, canonical schema, deduplication, and PII masking. Phase 2 adds the agentic brain: a tax-aligned Categorizer and a runway Forecaster. Phase 3 adds the RAG layer: a pluggable vector store for business notes plus an Advisor agent that answers strategic questions with citations back to specific transaction IDs. Every phase ships a zero-dependency default backend and an optional upgrade path (Gemini for LLM categorization and rewriting, Prophet for forecasting, sentence-transformers + ChromaDB for vector memory).

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

### Phase 2 deliverables

- `CategorizerAgent` - assigns one of 18 IRS Schedule-C-style categories (Income, Rent, Software & Subscriptions, Meals, Travel, Payroll, …). Default backend is a deterministic keyword-scoring engine; if `google-generativeai` is installed and `GEMINI_API_KEY` is set, it transparently switches to Gemini 1.5 Flash and falls back on any error.
- `ForecasterAgent` - builds a contiguous daily-net series, computes mean burn rate, and projects cash position over a configurable horizon. Returns a `Forecast` with `death_date`, `mean_daily_burn`, and a full projection DataFrame. Default backend is linear; upgrades to Prophet automatically if the `prophet` package is installed.
- `pipeline.run_pipeline(...)` - Python API and CLI that chains ingest → dedupe → categorize → forecast and writes `transactions_categorized.csv` + `forecast.csv`.

### Phase 3 deliverables

- `HashingEmbedder` - deterministic, dependency-free bag-of-words embedder that produces cosine-comparable vectors. Drop-in `SentenceTransformerEmbedder` activates if `sentence-transformers` is installed.
- `InMemoryVectorStore` - tiny persistent vector store (JSON on disk) with upsert, cosine query, and stable content-hash IDs. `ChromaVectorStore` is a drop-in replacement when `chromadb` is available.
- `AdvisorAgent` - routes questions into four intents (runway, affordability, category spend, general retrieval), composes the Forecaster for cashflow math, and returns an `AdvisorAnswer` whose every financial claim is backed by a list of real transaction IDs. Optional Gemini rewrite preserves the numbers and citations but rephrases for fluency.
- `fin_flow.advisor_cli` - two subcommands (`note`, `ask`) for adding business context and asking grounded questions from the terminal.

### Grounding guarantee

Per the spec's QA section, every financial claim the Advisor makes is traceable. `AdvisorAnswer.citations` is a list of `raw_hash` values that exist in the transactions DataFrame passed to `ask(...)`. A dedicated test (`test_advisor.py::test_citations_reference_only_real_transaction_ids`) enforces that no answer can fabricate a transaction ID.

### Project layout

```
src/fin_flow/
  ingestion/    # CSV / Excel / JSON parsers, schema normalizer, CLI
  agents/       # Categorizer, Forecaster, Advisor (grounded RAG)
  storage/      # embeddings (hashing / sentence-transformers) + vector store (in-memory / Chroma)
  utils/        # PII masking, logging, hashing
  pipeline.py   # high-level Phase 1+2 orchestrator (Python API + CLI)
  advisor_cli.py # Phase 3 CLI: add notes, ask grounded questions
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

Remaining phases add the Supabase relational storage layer and the Streamlit dashboard (see the technical spec).
