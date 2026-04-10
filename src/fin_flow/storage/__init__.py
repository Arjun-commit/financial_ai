"""Storage layer — vector memory and (future) relational client."""

from .embeddings import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    best_available,
    cosine,
)
from .vector_store import (
    InMemoryVectorStore,
    ChromaVectorStore,
    VectorHit,
    VectorRecord,
    best_available_store,
)

__all__ = [
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "best_available",
    "cosine",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "VectorHit",
    "VectorRecord",
    "best_available_store",
]
