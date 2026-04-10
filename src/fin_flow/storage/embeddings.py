"""Text embedding backends.

Default backend is a tiny, dependency-free hashing bag-of-words model that
produces deterministic, cosine-comparable vectors. It is intentionally
simple: it tokenizes on word boundaries, lowercases, and hashes each token
into a fixed-size vector. Cosine similarity between two such vectors
approximates keyword overlap — good enough for retrieval over a few
hundred short business notes without pulling in a 400MB model.

If the optional `sentence-transformers` package is available, the
`SentenceTransformerEmbedder` drop-in replacement uses a real neural
encoder (default: `all-MiniLM-L6-v2`). The Advisor agent picks whichever
backend is available, preferring the neural one when present.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Optional


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class HashingEmbedder:
    """Deterministic hashing bag-of-words embedder.

    `dim` controls the vector dimensionality. 256 is plenty for small
    corpora and keeps memory microscopic.
    """

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _hash(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % self.dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            vec[self._hash(tok)] += 1.0
        # L2 normalize so cosine similarity == dot product
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class SentenceTransformerEmbedder:
    """Optional neural embedder. Activates if `sentence_transformers`
    is importable. Otherwise stays dormant and the agent falls back."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._available = False
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name)
            self._available = True
        except Exception:  # noqa: BLE001
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def embed(self, text: str) -> list[float]:
        if not self._available:
            raise RuntimeError("sentence-transformers not installed")
        vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]
        return [float(x) for x in vec]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        if not self._available:
            raise RuntimeError("sentence-transformers not installed")
        arr = self._model.encode(list(texts), normalize_embeddings=True)  # type: ignore[union-attr]
        return [[float(x) for x in row] for row in arr]


def best_available() -> HashingEmbedder | SentenceTransformerEmbedder:
    st = SentenceTransformerEmbedder()
    if st.available:
        return st
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
