"""Hybrid retrieval: multi-query -> dense + BM25 -> RRF -> cross-encoder rerank."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .embedder import Embedder, Reranker
from .llm import BaseLLM
from .persian import normalize, swap_lam_alef
from .store import Store

RRF_K = 60

_EXPAND_PROMPT = """تو یک دستیار بازنویسی پرسش برای موتور جستجو هستی.
پرسش کاربر را به {n} شکل دیگر بازنویسی کن تا احتمال یافتن متن مرتبط در یک کتاب بیشتر شود.
از مترادف‌ها و اصطلاحات تخصصی مرتبط استفاده کن. هر بازنویسی را در یک خط جداگانه بنویس.
فقط خود بازنویسی‌ها را بنویس، بدون شماره‌گذاری و بدون توضیح.

پرسش: {q}"""


@dataclass
class Hit:
    index: int
    chunk: dict
    fused: float = 0.0
    dense: float = 0.0
    sparse: float = 0.0
    rerank: float | None = None
    is_neighbor: bool = False


@dataclass
class RetrievalResult:
    hits: list[Hit] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    reranked: bool = False
    abstain: bool = False
    top_score: float | None = None
    top_dense: float | None = None


class Retriever:
    def __init__(self, cfg: Config, store: Store, embedder: Embedder, reranker: Reranker | None):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def expand(self, query: str, llm: BaseLLM) -> list[str]:
        if not self.cfg.multi_query or self.cfg.multi_query_count <= 0:
            return []
        try:
            raw = llm.complete(
                [{"role": "user", "content": _EXPAND_PROMPT.format(n=self.cfg.multi_query_count, q=query)}],
                temperature=0.3,
                max_tokens=160,
            )
        except Exception:
            return []
        out = []
        for line in raw.split("\n"):
            line = re.sub(r"^\s*[-*\d.)：:]+\s*", "", line).strip()
            if 3 < len(line) < 300 and line != query:
                out.append(line)
        return out[: self.cfg.multi_query_count]

    def retrieve(self, query: str, llm: BaseLLM | None = None, doc_ids: set[str] | None = None) -> RetrievalResult:
        query = normalize(query)
        if not self.store.chunks:
            return RetrievalResult(queries=[query])

        queries = [query]
        if llm is not None:
            queries += self.expand(query, llm)

        # a corpus whose producer reversed «لا» must be matched in its own spelling,
        # otherwise both retrieval and reranking silently under-score every passage
        rerank_query = query
        if any(d.get("lam_alef_broken") for d in self.store.documents.values()):
            variant = swap_lam_alef(query)
            if variant != query:
                queries.append(variant)
                rerank_query = variant

        fused: dict[int, float] = {}
        dense_scores: dict[int, float] = {}
        sparse_scores: dict[int, float] = {}

        for qi, q in enumerate(queries):
            weight = 1.0 if qi == 0 else 0.6
            qvec = self.embedder.encode_query(q)
            for rank, (idx, score) in enumerate(
                self.store.search_dense(qvec, self.cfg.dense_top_k, doc_ids)
            ):
                fused[idx] = fused.get(idx, 0.0) + weight / (RRF_K + rank + 1)
                dense_scores[idx] = max(dense_scores.get(idx, -1.0), score)
            for rank, (idx, score) in enumerate(
                self.store.search_sparse(q, self.cfg.sparse_top_k, doc_ids)
            ):
                fused[idx] = fused.get(idx, 0.0) + weight / (RRF_K + rank + 1)
                sparse_scores[idx] = max(sparse_scores.get(idx, 0.0), score)

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: self.cfg.fusion_top_k]
        hits = [
            Hit(
                index=i,
                chunk=self.store.chunks[i],
                fused=s,
                dense=dense_scores.get(i, 0.0),
                sparse=sparse_scores.get(i, 0.0),
            )
            for i, s in ordered
        ]

        reranked = False
        top_score: float | None = None
        top_dense = max((h.dense for h in hits), default=0.0)

        if self.cfg.use_reranker and self.reranker is not None and hits:
            head, tail = hits[: self.cfg.rerank_candidates], hits[self.cfg.rerank_candidates :]
            scores = self.reranker.score(rerank_query, [h.chunk["text"] for h in head])
            if self.reranker.available:
                for h, s in zip(head, scores):
                    h.rerank = s
                head.sort(key=lambda h: h.rerank, reverse=True)
                top_score = head[0].rerank
                kept = [h for h in head if h.rerank >= self.cfg.min_rerank_score]
                hits = (kept or head[:2]) + tail
                reranked = True

        if top_score is not None:
            no_answer = top_score < self.cfg.abstain_below
        else:
            no_answer = top_dense < self.cfg.abstain_dense_below
        if no_answer:
            return RetrievalResult(
                hits=[],
                queries=queries,
                reranked=reranked,
                abstain=True,
                top_score=top_score,
                top_dense=top_dense,
            )

        top = hits[: self.cfg.rerank_top_n]
        if self.cfg.neighbor_window > 0:
            seen = {h.index for h in top}
            extra: list[Hit] = []
            for h in top:
                for j in self.store.neighbors(h.index, self.cfg.neighbor_window):
                    if j not in seen:
                        seen.add(j)
                        extra.append(Hit(index=j, chunk=self.store.chunks[j], is_neighbor=True))
            top = _merge_in_reading_order(top, extra, self.cfg.neighbor_window)

        return RetrievalResult(
            hits=top, queries=queries, reranked=reranked, top_score=top_score, top_dense=top_dense
        )


def _merge_in_reading_order(primary: list[Hit], extra: list[Hit], window: int = 1) -> list[Hit]:
    """Keep primary ranking, but attach neighbors right after their anchor for readable context."""
    by_key = {(h.chunk["doc_id"], h.chunk["ordinal"]): h for h in extra}
    out: list[Hit] = []
    used: set[tuple[str, int]] = set()
    for h in primary:
        doc, ordn = h.chunk["doc_id"], h.chunk["ordinal"]
        group = [h]
        for delta in range(-window, window + 1):
            key = (doc, ordn + delta)
            if delta and key in by_key and key not in used:
                used.add(key)
                group.append(by_key[key])
        group.sort(key=lambda x: x.chunk["ordinal"])
        out.extend(group)
    return out
