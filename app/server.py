"""FastAPI server: upload, index, and grounded chat over the user's own documents."""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import answer as answer_mod
from .config import ROOT
from .engine import Engine, Job

WEB_DIR = ROOT / "web"

engine = Engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=engine.warmup, daemon=True).start()
    yield


app = FastAPI(title="Grounded Book QA", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    doc_ids: list[str] | None = None


class ConfigPatch(BaseModel):
    llm_backend: str | None = None
    ollama_model: str | None = None
    ollama_url: str | None = None
    openai_base_url: str | None = None
    openai_model: str | None = None
    openai_api_key: str | None = None
    use_reranker: bool | None = None
    multi_query: bool | None = None
    rerank_top_n: int | None = None
    temperature: float | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    return engine.status


@app.get("/api/documents")
def documents() -> dict:
    docs = sorted(engine.store.documents.values(), key=lambda d: d.get("added_at", 0), reverse=True)
    return {"documents": docs}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not engine.store.remove_document(doc_id):
        raise HTTPException(404, "document not found")
    return {"ok": True, **engine.store.stats}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    jobs = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        path = engine.save_upload(f.filename or "file", data)
        job = Job(id=uuid.uuid4().hex[:10], filename=f.filename or path.name)
        engine.jobs[job.id] = job
        threading.Thread(target=engine.ingest, args=(path, job), daemon=True).start()
        jobs.append(job.to_dict())
    if not jobs:
        raise HTTPException(400, "no files received")
    return {"jobs": jobs}


@app.get("/api/jobs")
def jobs() -> dict:
    return {"jobs": [j.to_dict() for j in engine.jobs.values()]}


@app.post("/api/jobs/cancel-all")
def cancel_all_jobs() -> dict:
    return {"cancelled": engine.cancel_pending()}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if not engine.cancel(job_id):
        raise HTTPException(404, "job not found or already finished")
    return {"ok": True}


@app.post("/api/jobs/clear")
def clear_finished_jobs() -> dict:
    keep = {k: v for k, v in engine.jobs.items() if v.status not in ("done", "error", "cancelled")}
    removed = len(engine.jobs) - len(keep)
    engine.jobs = keep
    return {"removed": removed}


@app.post("/api/config")
def patch_config(patch: ConfigPatch) -> dict:
    changed = patch.model_dump(exclude_none=True)
    for k, v in changed.items():
        setattr(engine.cfg, k, v)
    engine.cfg.save()
    if any(k.startswith(("llm", "ollama", "openai")) for k in changed):
        engine.reset_llm()
    return {"ok": True, "status": engine.status}


def _sse(event: str, payload: dict) -> str:
    return f"data: {json.dumps({'type': event, **payload}, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "empty question")

    def gen():
        try:
            if not engine.store.chunks:
                yield _sse("error", {"message": "هنوز هیچ سندی بارگذاری نشده است."})
                return
            yield _sse("stage", {"stage": "retrieving"})
            doc_ids = set(req.doc_ids) if req.doc_ids else None
            standalone, result, sources = engine.ask(question, req.history, doc_ids)
            yield _sse(
                "sources",
                {
                    "standalone": standalone,
                    "queries": result.queries,
                    "reranked": result.reranked,
                    "abstain": result.abstain,
                    "top_score": result.top_score,
                    "top_dense": result.top_dense,
                    "sources": [s.to_dict() for s in sources],
                },
            )
            if not sources:
                yield _sse("token", {"text": answer_mod.NO_ANSWER_FA})
                yield _sse("done", {"cited": []})
                return

            yield _sse("stage", {"stage": "generating"})
            buf = []
            for piece in answer_mod.stream_answer(standalone, sources, engine.llm, engine.cfg):
                buf.append(piece)
                yield _sse("token", {"text": piece})
            full = answer_mod.finalize("".join(buf), sources)
            cited = sorted(answer_mod.cited_labels(full, sources), key=lambda x: int(x[1:]))
            yield _sse("done", {"cited": cited, "answer": full})
        except Exception as e:  # noqa: BLE001
            yield _sse("error", {"message": f"{e.__class__.__name__}: {e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/api/reset")
def reset() -> dict:
    engine.store.clear()
    engine.jobs.clear()
    return {"ok": True}


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
