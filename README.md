# Fin-Flow Agent (CFO-AI)

Autonomous multi-agent AI system that acts as a virtual CFO — ingests raw financial data, categorizes expenses, forecasts cash flow, and answers strategic questions.

## Phase 1 — Ingestion & Cleaning (current)

This phase contains the data plumbing layer: file parsers, schema normalization, deduplication, and PII masking. No LLMs or cloud services are required to run it.

### Quick start

```bash
pip install -r requirements.txt
python -m fin_flow.ingestion.cli data/samples/chase_sample.csv --out data/processed/clean.csv
pytest -q
```

### Project layout

```
src/fin_flow/
  ingestion/    # CSV / Excel / JSON parsers, schema normalizer, CLI
  agents/       # (Phase 2) Categorizer, Forecaster, Advisor
  storage/      # (Phase 2) Supabase + ChromaDB clients
  utils/        # PII masking, logging, hashing
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

Later phases add the Categorizer, Forecaster, Advisor agents, vector memory, and the Streamlit UI (see the technical spec).
