"""Advisor CLI — add business notes and ask grounded questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .agents import AdvisorAgent
from .storage import InMemoryVectorStore


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fin-flow-advisor")
    sub = p.add_subparsers(dest="cmd", required=True)

    note = sub.add_parser("note", help="Add a business note to the vector store.")
    note.add_argument("--text", required=True)
    note.add_argument("--type", default="note")
    note.add_argument("--priority", default="normal")
    note.add_argument("--store", default="data/processed/notes.json")

    ask = sub.add_parser("ask", help="Ask a grounded financial question.")
    ask.add_argument("question", help="The question to ask.")
    ask.add_argument(
        "--transactions",
        required=True,
        help="Path to a categorized transactions CSV (from the pipeline).",
    )
    ask.add_argument("--starting-balance", type=float, default=0.0)
    ask.add_argument("--store", default="data/processed/notes.json")
    ask.add_argument("--no-llm", action="store_true")
    ask.add_argument("--json", action="store_true", help="Emit JSON instead of prose.")

    return p


def _load_store(path: str) -> InMemoryVectorStore:
    return InMemoryVectorStore(persist_path=path)


def _cmd_note(args) -> int:
    store = _load_store(args.store)
    advisor = AdvisorAgent(vector_store=store, prefer_llm=False)
    note_id = advisor.add_note(args.text, type=args.type, priority=args.priority)
    print(f"[ok] added note {note_id} (store now has {len(store)} notes)")
    return 0


def _cmd_ask(args) -> int:
    store = _load_store(args.store)
    advisor = AdvisorAgent(vector_store=store, prefer_llm=not args.no_llm)
    df = pd.read_csv(args.transactions)
    answer = advisor.ask(
        args.question,
        transactions=df,
        starting_balance=args.starting_balance,
    )
    if args.json:
        print(json.dumps(answer.as_dict(), indent=2))
    else:
        print(f"[intent: {answer.intent} | backend: {answer.backend}]")
        print(answer.answer)
        if answer.citations:
            print(f"\nGrounded on {len(answer.citations)} transaction(s). "
                  f"First IDs: {answer.citations[:3]}")
        if answer.retrieved_notes:
            print("\nRetrieved context notes:")
            for h in answer.retrieved_notes:
                print(f"  - [{h.id}] (score={h.score:.2f}) {h.text}")
    return 0


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "note":
        return _cmd_note(args)
    if args.cmd == "ask":
        return _cmd_ask(args)
    print(f"[error] unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(run())
