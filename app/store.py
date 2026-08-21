"""Persistent hybrid index: dense vectors + BM25 + chunk metadata."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .bm25 import BM25


class Store:
    def __init__(self, index_dir: Path):
        self.dir = index_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.bm25 = BM25()
        self.documents: dict[str, dict] = {}
        self.embed_model: str = ""
        self._ord_index: dict[tuple[str, int], int] = {}
        self._lock = threading.RLock()
        self.load()
        self._reindex_ordinals()

    @property
    def chunks_path(self) -> Path:
        return self.dir / "chunks.json"

    @property
    def emb_path(self) -> Path:
        return self.dir / "embeddings.npy"

    @property
    def bm25_path(self) -> Path:
        return self.dir / "bm25.pkl"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    def load(self) -> None:
        if self.chunks_path.exists():
            self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if self.emb_path.exists():
            self.embeddings = np.load(self.emb_path)
        if self.bm25_path.exists():
            try:
                self.bm25 = BM25.load(self.bm25_path)
            except Exception:
                self._rebuild_bm25()
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.documents = meta.get("documents", {})
            self.embed_model = meta.get("embed_model", "")

    def save(self) -> None:
        with self._lock:
            self.chunks_path.write_text(
                json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
            )
            if self.embeddings is not None:
                np.save(self.emb_path, self.embeddings)
            self.bm25.save(self.bm25_path)
            self.meta_path.write_text(
                json.dumps(
                    {"documents": self.documents, "embed_model": self.embed_model},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _rebuild_bm25(self) -> None:
        self.bm25 = BM25()
        self.bm25.build([c["embed_text"] for c in self.chunks])
        self._reindex_ordinals()

    def _reindex_ordinals(self) -> None:
        self._ord_index = {(c["doc_id"], c["ordinal"]): i for i, c in enumerate(self.chunks)}

    def add_document(self, doc_meta: dict, chunks: list[dict], embeddings: np.ndarray) -> None:
        with self._lock:
            doc_id = doc_meta["id"]
            if doc_id in self.documents:
                self.remove_document(doc_id, save=False)
            self.chunks.extend(chunks)
            if self.embeddings is None or len(self.embeddings) == 0:
                self.embeddings = embeddings.astype(np.float32)
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
            self.documents[doc_id] = doc_meta
            self._rebuild_bm25()
            self.save()

    def remove_document(self, doc_id: str, save: bool = True) -> bool:
        with self._lock:
            if doc_id not in self.documents:
                return False
            keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
            self.chunks = [self.chunks[i] for i in keep]
            if self.embeddings is not None and len(self.embeddings):
                self.embeddings = (
                    self.embeddings[keep] if keep else np.zeros((0, self.embeddings.shape[1]), np.float32)
                )
            self.documents.pop(doc_id, None)
            self._rebuild_bm25()
            if save:
                self.save()
            return True

    def clear(self) -> None:
        with self._lock:
            self.chunks = []
            self.embeddings = None
            self.documents = {}
            self.bm25 = BM25()
            for p in (self.chunks_path, self.emb_path, self.bm25_path, self.meta_path):
                if p.exists():
                    p.unlink()

    def search_dense(self, qvec: np.ndarray, top_k: int, doc_ids: set[str] | None = None):
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        sims = self.embeddings @ qvec
        if doc_ids:
            mask = np.array([c["doc_id"] in doc_ids for c in self.chunks])
            sims = np.where(mask, sims, -1e9)
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx if sims[i] > -1e8]

    def search_sparse(self, query: str, top_k: int, doc_ids: set[str] | None = None):
        hits = self.bm25.search(query, top_k * (3 if doc_ids else 1))
        if doc_ids:
            hits = [(i, s) for i, s in hits if self.chunks[i]["doc_id"] in doc_ids][:top_k]
        return hits

    def neighbors(self, idx: int, window: int) -> list[int]:
        chunk = self.chunks[idx]
        doc_id, ordinal = chunk["doc_id"], chunk["ordinal"]
        out = []
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            j = self._ord_index.get((doc_id, ordinal + delta))
            if j is not None:
                out.append(j)
        return out

    @property
    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embed_model": self.embed_model,
        }
