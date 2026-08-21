"""Retrieval quality check against a small labelled question set.

questions.json format:
[
  {"q": "اثر شلاقی چیست؟", "must_contain": ["تشدید", "واریانس"], "pages": [12, 13]},
  ...
]
Either `must_contain` (substring in any retrieved chunk) or `pages` may be given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine import Engine  # noqa: E402
from app.persian import fold_for_match  # noqa: E402


def hit(case: dict, chunks: list[dict]) -> bool:
    if case.get("pages"):
        pages = set(case["pages"])
        if any(pages & set(range(c["page_start"], c["page_end"] + 1)) for c in chunks):
            return True
    needles = case.get("must_contain") or []
    if needles:
        blob = fold_for_match(" ".join(c["text"] for c in chunks))
        return all(fold_for_match(n) in blob for n in needles)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", help="path to questions.json")
    ap.add_argument("--no-expand", action="store_true", help="disable LLM query expansion")
    args = ap.parse_args()

    cases = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    engine = Engine()
    llm = None if args.no_expand else engine.llm

    hits = 0
    total_ms = 0.0
    for case in cases:
        t0 = time.time()
        result = engine.retriever.retrieve(case["q"], llm=llm)
        ms = (time.time() - t0) * 1000
        total_ms += ms
        chunks = [h.chunk for h in result.hits]
        ok = hit(case, chunks)
        hits += ok
        pages = sorted({c["page_start"] for c in chunks})
        print(f"{'PASS' if ok else 'FAIL'}  {ms:6.0f}ms  pages={pages}  {case['q'][:60]}")

    n = len(cases)
    print(f"\nrecall@{engine.cfg.rerank_top_n}: {hits}/{n} = {hits / max(n, 1):.1%}")
    print(f"avg latency: {total_ms / max(n, 1):.0f} ms")


if __name__ == "__main__":
    main()
