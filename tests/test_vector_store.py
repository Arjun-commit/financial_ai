from pathlib import Path

from fin_flow.storage import HashingEmbedder, InMemoryVectorStore

REPO = Path(__file__).resolve().parents[1]


def _store() -> InMemoryVectorStore:
    return InMemoryVectorStore(embedder=HashingEmbedder(dim=128))


def test_add_and_query_returns_best_match():
    s = _store()
    s.add("We plan to scale marketing spend by 20% in Q3.", metadata={"type": "strategic_goal"})
    s.add("Hire two engineers by end of Q2.", metadata={"type": "strategic_goal"})
    s.add("Keep burn rate under $8,000 per month.", metadata={"type": "constraint"})

    hits = s.query("marketing plans for next quarter", k=2)
    assert hits
    assert "marketing" in hits[0].text.lower()


def test_upsert_replaces_same_id():
    s = _store()
    nid = s.add("first version", id="note-1")
    assert nid == "note-1"
    s.add("second version", id="note-1")
    assert len(s) == 1
    hits = s.query("second", k=1)
    assert hits[0].text == "second version"


def test_persistence_round_trip():
    # The sandbox doesn't let us unlink files under `data/processed`, so
    # we rely on stable content-hashed ids and add-on-save being idempotent.
    path = REPO / "data" / "processed" / "test_vector_store.json"
    s = InMemoryVectorStore(embedder=HashingEmbedder(dim=64), persist_path=path)
    # Reset in-memory records so the assertion is tight, then re-add.
    s.records = []
    s.add("persist me uniquely for the round trip test", metadata={"type": "memo"})
    assert path.exists()

    s2 = InMemoryVectorStore(embedder=HashingEmbedder(dim=64), persist_path=path)
    texts = {r.text for r in s2.records}
    assert "persist me uniquely for the round trip test" in texts


def test_empty_query_returns_empty():
    s = _store()
    assert s.query("anything", k=3) == []
