"""Okapi BM25 over Persian-normalized tokens."""

from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path

from .persian import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    def build(self, texts: list[str]) -> None:
        self.doc_freq = Counter()
        self.postings = {}
        self.doc_len = []
        for idx, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_len.append(len(tokens))
            tf = Counter(tokens)
            for term, freq in tf.items():
                self.postings.setdefault(term, []).append((idx, freq))
                self.doc_freq[term] += 1
        self.n_docs = len(texts)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 40) -> list[tuple[int, float]]:
        if not self.n_docs:
            return []
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for idx, freq in postings:
                dl = self.doc_len[idx] or 1
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[idx] = scores.get(idx, 0.0) + idf * freq * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "doc_freq": self.doc_freq,
                    "postings": self.postings,
                    "doc_len": self.doc_len,
                    "avgdl": self.avgdl,
                    "n_docs": self.n_docs,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(state["k1"], state["b"])
        obj.doc_freq = state["doc_freq"]
        obj.postings = state["postings"]
        obj.doc_len = state["doc_len"]
        obj.avgdl = state["avgdl"]
        obj.n_docs = state["n_docs"]
        return obj
