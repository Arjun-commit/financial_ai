"""Text embedding backends."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Optional


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class HashingEmbedder:
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
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class SentenceTransformerEmbedder:
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
