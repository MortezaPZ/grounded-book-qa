"""Application engine: ingestion pipeline and shared singletons."""

from __future__ import annotations

import hashlib
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from . import answer as answer_mod
from .chunker import chunk_document
from .config import Config, INDEX_DIR, UPLOAD_DIR, ensure_dirs
from .embedder import Embedder, Reranker
from .extract import SUPPORTED, extract
from .llm import build_llm
from .retriever import Retriever
from .store import Store


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued | extracting | chunking | embedding | done | error | cancelled
    cancelled: bool = False
    progress: float = 0.0
    message: str = ""
    doc_id: str = ""
    pages: int = 0
    chunks: int = 0
    warnings: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "cancelled": self.cancelled,
            "progress": round(self.progress, 3),
            "message": self.message,
            "doc_id": self.doc_id,
            "pages": self.pages,
            "chunks": self.chunks,
            "warnings": self.warnings,
        }


class Engine:
    def __init__(self, cfg: Config | None = None):
        ensure_dirs()
        self.cfg = cfg or Config.load()
        self.store = Store(INDEX_DIR)
        self.embedder = Embedder(self.cfg.embed_model, self.cfg.device, self.cfg.embed_batch_size)
        self.reranker = Reranker(self.cfg.rerank_model, self.cfg.device) if self.cfg.use_reranker else None
        self.retriever = Retriever(self.cfg, self.store, self.embedder, self.reranker)
        self.jobs: dict[str, Job] = {}
        self._llm = None
        self._llm_lock = threading.Lock()
        self._ingest_lock = threading.Lock()

        if self.store.embed_model and self.store.embed_model != self.cfg.embed_model:
            print(
                f"[warn] index was built with '{self.store.embed_model}' but config uses "
                f"'{self.cfg.embed_model}'. Re-index required for correct results."
            )

    @property
    def llm(self):
        if self._llm is None:
            with self._llm_lock:
                if self._llm is None:
                    self._llm = build_llm(self.cfg)
        return self._llm

    def reset_llm(self) -> None:
        with self._llm_lock:
            self._llm = None

    def warmup(self) -> None:
        try:
            self.embedder.encode_query("سلام")
        except Exception:
            traceback.print_exc()

    def save_upload(self, filename: str, data: bytes) -> Path:
        safe = Path(filename).name.replace("\\", "_")
        digest = hashlib.sha1(data).hexdigest()[:10]
        dest = UPLOAD_DIR / f"{digest}_{safe}"
        dest.write_bytes(data)
        return dest

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status in ("done", "error", "cancelled"):
            return False
        job.cancelled = True
        if job.status == "queued":
            job.status = "cancelled"
            job.message = "لغو شد"
        return True

    def cancel_pending(self) -> int:
        return sum(
            self.cancel(j.id) for j in list(self.jobs.values()) if j.status not in ("done", "error", "cancelled")
        )

    def ingest(self, path: Path, job: Job) -> None:
        with self._ingest_lock:
            try:
                self._ingest(path, job)
            except Cancelled:
                job.status = "cancelled"
                job.message = "لغو شد"
            except Exception as e:
                job.status = "error"
                job.message = f"{e.__class__.__name__}: {e}"
                traceback.print_exc()

    def _ingest(self, path: Path, job: Job) -> None:
        if path.suffix.lower() not in SUPPORTED:
            raise ValueError(f"نوع فایل پشتیبانی نمی‌شود: {path.suffix}")
        if job.cancelled:
            raise Cancelled

        # the upload name carries a hash of the content, so an identical file lands on
        # the same id and re-indexing it would only rebuild what is already there
        doc_id = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]
        existing = self.store.documents.get(doc_id)
        if existing and existing.get("path") == str(path):
            job.doc_id = doc_id
            job.pages = existing.get("pages", 0)
            job.chunks = existing.get("chunks", 0)
            job.status = "done"
            job.progress = 1.0
            job.message = f"از قبل نمایه شده بود ({existing.get('chunks', 0)} قطعه)"
            return

        job.status = "extracting"
        job.message = "استخراج متن"
        job.progress = 0.05
        doc = extract(path, ocr=self.cfg.ocr_enabled, ocr_lang=self.cfg.ocr_lang)
        job.pages = len(doc.pages)

        if doc.char_count < 200:
            job.warnings.append(
                "متنی از این فایل استخراج نشد. اگر PDF اسکن‌شده است، Tesseract با زبان فارسی نصب کنید."
            )
        if doc.scanned_pages:
            job.warnings.append(
                f"{len(doc.scanned_pages)} صفحه بدون متن قابل استخراج بود (احتمالاً تصویری/اسکن)."
            )
        if doc.ocr_pages:
            job.warnings.append(f"{len(doc.ocr_pages)} صفحه با OCR خوانده شد.")
        if doc.lam_alef_broken:
            job.warnings.append(
                "این فایل حرف «لا» را وارونه ذخیره کرده است (مثلاً «بالا» به شکل «باال»). "
                "سیستم پرسش را با هر دو شکل جستجو می‌کند."
            )

        if job.cancelled:
            raise Cancelled
        job.status = "chunking"
        job.message = "قطعه‌بندی"
        job.progress = 0.2
        chunks = chunk_document(
            doc,
            doc_id=doc_id,
            target_tokens=self.cfg.chunk_tokens,
            overlap_tokens=self.cfg.chunk_overlap,
            min_chars=self.cfg.min_chunk_chars,
            count_tokens=self.embedder.count_tokens,
        )
        job.chunks = len(chunks)
        if not chunks:
            raise ValueError("هیچ محتوای متنی برای نمایه‌سازی یافت نشد.")

        job.status = "embedding"
        job.message = f"ساخت بردار برای {len(chunks)} قطعه"

        def on_progress(done: int, total: int) -> None:
            if job.cancelled:
                raise Cancelled
            job.progress = 0.2 + 0.75 * (done / max(total, 1))
            job.message = f"ساخت بردار {done}/{total}"

        vectors = self.embedder.encode_passages([c.embed_text for c in chunks], progress=on_progress)

        meta = {
            "id": doc_id,
            "title": doc.title,
            "filename": path.name,
            "original_name": job.filename,
            "path": str(path),
            "pages": len(doc.pages),
            "chunks": len(chunks),
            "chars": doc.char_count,
            "added_at": time.time(),
            "warnings": job.warnings,
            "lam_alef_broken": doc.lam_alef_broken,
        }
        self.store.embed_model = self.cfg.embed_model
        self.store.add_document(meta, [c.to_dict() for c in chunks], vectors)

        job.doc_id = doc_id
        job.status = "done"
        job.progress = 1.0
        job.message = f"{len(chunks)} قطعه از {len(doc.pages)} صفحه نمایه شد"

    def ask(self, question: str, history: list[dict] | None = None, doc_ids: set[str] | None = None):
        history = history or []
        standalone = (
            answer_mod.condense(question, history, self.llm)
            if history and self.cfg.condense_history
            else question
        )
        result = self.retriever.retrieve(standalone, llm=self.llm, doc_ids=doc_ids)
        sources = answer_mod.build_sources(result, self.cfg.max_context_chars)
        if self.cfg.unload_reranker_before_answer and self.reranker is not None:
            self.reranker.unload()
        return standalone, result, sources

    @property
    def status(self) -> dict:
        ok, msg = self.llm.health()
        return {
            "llm_ready": ok,
            "llm": msg,
            "backend": self.llm.name,
            **self.store.stats,
            "embed_model": self.cfg.embed_model,
            "index_embed_model": self.store.embed_model,
            "stale_index": bool(self.store.embed_model and self.store.embed_model != self.cfg.embed_model),
            "rerank_model": self.cfg.rerank_model if self.cfg.use_reranker else None,
        }
