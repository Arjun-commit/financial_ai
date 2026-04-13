"""Vector store for business notes and strategic goals."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .embeddings import HashingEmbedder, best_available, cosine


@dataclass
class VectorRecord:
    id: str
    text: str
    metadata: dict
    embedding: list[float] = field(repr=False)


@dataclass
class VectorHit:
    id: str
    text: str
    metadata: dict
    score: float


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class InMemoryVectorStore:
    name = "in-memory"

    def __init__(self, embedder=None, persist_path: Optional[str | Path] = None) -> None:
        self.embedder = embedder or best_available()
        self.records: list[VectorRecord] = []
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self._load()

    def add(
        self,
        text: str,
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("cannot add empty text to vector store")
        rid = id or _stable_id(text)
        self.records = [r for r in self.records if r.id != rid]
        vec = self.embedder.embed(text)
        self.records.append(
            VectorRecord(
                id=rid,
                text=text.strip(),
                metadata=metadata or {},
                embedding=vec,
            )
        )
        if self.persist_path:
            self._save()
        return rid

    def add_many(self, items: Iterable[dict]) -> list[str]:
        out = []
        for item in items:
            out.append(
                self.add(
                    text=item["text"],
                    metadata=item.get("metadata"),
                    id=item.get("id"),
                )
            )
        return out

    def query(self, text: str, k: int = 4) -> list[VectorHit]:
        if not self.records:
            return []
        q = self.embedder.embed(text)
        scored = [
            VectorHit(
                id=r.id,
                text=r.text,
                metadata=r.metadata,
                score=cosine(q, r.embedding),
            )
            for r in self.records
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return [h for h in scored[:k] if h.score > 0.0]

    def __len__(self) -> int:
        return len(self.records)

    def _save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "id": r.id,
                "text": r.text,
                "metadata": r.metadata,
                "embedding": r.embedding,
            }
            for r in self.records
        ]
        with self.persist_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _load(self) -> None:
        try:
            with self.persist_path.open("r", encoding="utf-8") as f:  # type: ignore[union-attr]
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self.records = [
            VectorRecord(
                id=item["id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
                embedding=item["embedding"],
            )
            for item in payload
        ]


class ChromaVectorStore:
    name = "chroma"

    def __init__(
        self,
        collection_name: str = "business_context",
        persist_dir: Optional[str] = None,
        embedder=None,
    ) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as e:
            raise RuntimeError("chromadb not installed") from e

        self.embedder = embedder or best_available()
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(collection_name)

    def add(
        self,
        text: str,
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
    ) -> str:
        rid = id or _stable_id(text)
        vec = self.embedder.embed(text)
        self._collection.upsert(
            ids=[rid],
            embeddings=[vec],
            documents=[text],
            metadatas=[metadata or {}],
        )
        return rid

    def add_many(self, items: Iterable[dict]) -> list[str]:
        return [
            self.add(text=i["text"], metadata=i.get("metadata"), id=i.get("id"))
            for i in items
        ]

    def query(self, text: str, k: int = 4) -> list[VectorHit]:
        q = self.embedder.embed(text)
        res = self._collection.query(query_embeddings=[q], n_results=k)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[0.0] * len(ids)])[0]
        return [
            VectorHit(
                id=i,
                text=d,
                metadata=m or {},
                score=1.0 - float(dist),
            )
            for i, d, m, dist in zip(ids, docs, metas, dists)
        ]

    def __len__(self) -> int:
        return int(self._collection.count())


def best_available_store(
    persist_path: Optional[str | Path] = None,
) -> InMemoryVectorStore | ChromaVectorStore:
    try:
        import chromadb  # type: ignore  # noqa: F401

        return ChromaVectorStore(persist_dir=str(persist_path) if persist_path else None)
    except Exception:  # noqa: BLE001
        return InMemoryVectorStore(persist_path=persist_path)
