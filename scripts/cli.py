"""CLI for ingesting files and asking questions without the web UI."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import answer as answer_mod  # noqa: E402
from app.engine import Engine, Job  # noqa: E402


def cmd_ingest(engine: Engine, args) -> None:
    paths: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        paths.extend(sorted(p.rglob("*")) if p.is_dir() else [p])
    for p in paths:
        if not p.is_file() or p.suffix.lower() not in {".pdf", ".docx", ".txt", ".md", ".htm", ".html"}:
            continue
        job = Job(id=uuid.uuid4().hex[:8], filename=p.name)
        print(f"\n>> {p.name}")
        engine.ingest(p, job)
        print(f"   {job.status}: {job.message}")
        for w in job.warnings:
            print(f"   ! {w}")


def cmd_ask(engine: Engine, args) -> None:
    question = " ".join(args.question)
    standalone, result, sources = engine.ask(question)
    if result.queries[1:]:
        print("expanded:", " | ".join(result.queries[1:]))
    print(f"retrieved {len(sources)} chunks (reranked={result.reranked})\n")
    for s in sources:
        print(f"  [{s.label}] «{s.doc_title}» p{s.page_start}-{s.page_end} score={s.score:.3f}")
        print(f"      {s.text[:160].replace(chr(10), ' ')}…")
    print("\n" + "-" * 60 + "\n")
    buf = []
    for piece in answer_mod.stream_answer(standalone, sources, engine.llm, engine.cfg):
        buf.append(piece)
        print(piece, end="", flush=True)
    print("\n")
    cited = answer_mod.cited_labels("".join(buf))
    print("cited:", ", ".join(sorted(cited, key=lambda x: int(x[1:]))) or "none")


def cmd_search(engine: Engine, args) -> None:
    question = " ".join(args.question)
    result = engine.retriever.retrieve(question, llm=None)
    for h in result.hits:
        c = h.chunk
        tag = "~" if h.is_neighbor else "*"
        score = h.rerank if h.rerank is not None else h.fused
        print(f"{tag} p{c['page_start']:<4} score={score:>8.3f}  {c['text'][:110]}")


def cmd_status(engine: Engine, args) -> None:
    for k, v in engine.status.items():
        print(f"{k:14} {v}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest"); p.add_argument("paths", nargs="+"); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("ask"); p.add_argument("question", nargs="+"); p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("search"); p.add_argument("question", nargs="+"); p.set_defaults(fn=cmd_search)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("reset"); p.set_defaults(fn=lambda e, a: (e.store.clear(), print("cleared")))

    args = ap.parse_args()
    engine = Engine()
    args.fn(engine, args)


if __name__ == "__main__":
    main()
