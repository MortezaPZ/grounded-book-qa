"""Structure-aware chunking with contextual headers for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Callable

from .extract import ExtractedDoc

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?؟؛])\s+|\n+")


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    ordinal: int
    page_start: int
    page_end: int
    section: str
    text: str
    embed_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _default_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.35))


def _section_map(doc: ExtractedDoc) -> dict[int, str]:
    """page number -> breadcrumb of enclosing headings (PDF outlines only)."""
    entries = [o for o in doc.outline if o.page > 0]
    if not entries or max(o.page for o in entries) <= 1:
        return {}
    entries.sort(key=lambda o: (o.page, o.level))
    mapping: dict[int, str] = {}
    stack: list[tuple[int, str]] = []
    idx = 0
    max_page = max((p.number for p in doc.pages), default=1)
    for page in range(1, max_page + 1):
        while idx < len(entries) and entries[idx].page <= page:
            e = entries[idx]
            while stack and stack[-1][0] >= e.level:
                stack.pop()
            stack.append((e.level, e.title))
            idx += 1
        if stack:
            mapping[page] = " > ".join(t for _, t in stack[-3:])
    return mapping


def _heading_levels(doc: ExtractedDoc) -> dict[str, int]:
    levels: dict[str, int] = {}
    for o in doc.outline:
        key = o.title.strip()
        if key and len(key) < 200:
            levels.setdefault(key, o.level)
    return levels


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
    out: list[str] = []
    for s in parts:
        if len(s) > 1200:
            out.extend(s[i : i + 1200] for i in range(0, len(s), 1200))
        else:
            out.append(s)
    return out


def chunk_document(
    doc: ExtractedDoc,
    doc_id: str,
    target_tokens: int = 320,
    overlap_tokens: int = 64,
    min_chars: int = 120,
    count_tokens: Callable[[str], int] | None = None,
) -> list[Chunk]:
    count = count_tokens or _default_count
    page_sections = _section_map(doc)
    headings = _heading_levels(doc)

    units: list[tuple[str, int, int, str, bool]] = []  # (text, page, tokens, section, is_heading)
    stack: list[tuple[int, str]] = []
    for page in doc.pages:
        if not page.text.strip():
            continue
        for sent in _sentences(page.text):
            key = sent.strip()
            level = headings.get(key)
            if level is not None:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, key))
            section = " > ".join(t for _, t in stack[-3:]) or page_sections.get(page.number, "")
            units.append((sent, page.number, count(sent), section, level is not None))

    chunks: list[Chunk] = []
    buf: list[tuple[str, int, int, str, bool]] = []
    total = 0

    def flush() -> None:
        nonlocal buf, total
        if not buf:
            return
        text = " ".join(u[0] for u in buf).strip()
        if len(text) < min_chars and chunks:
            prev = chunks[-1]
            prev.text = (prev.text + " " + text).strip()
            prev.page_end = buf[-1][1]
            prev.embed_text = _embed_text(prev.doc_title, prev.section, prev.page_start, prev.text)
            buf, total = [], 0
            return
        p_start, p_end = buf[0][1], buf[-1][1]
        section = buf[0][3]
        chunks.append(
            Chunk(
                id=f"{doc_id}:{len(chunks)}",
                doc_id=doc_id,
                doc_title=doc.title,
                ordinal=len(chunks),
                page_start=p_start,
                page_end=p_end,
                section=section,
                text=text,
                embed_text=_embed_text(doc.title, section, p_start, text),
            )
        )
        buf, total = [], 0

    for unit in units:
        starts_section = unit[4] and total > target_tokens * 0.4
        if buf and (total + unit[2] > target_tokens or starts_section):
            tail: list[tuple[str, int, int, str, bool]] = []
            acc = 0
            if not starts_section:
                for u in reversed(buf):
                    if acc >= overlap_tokens:
                        break
                    tail.insert(0, u)
                    acc += u[2]
            flush()
            buf = list(tail)
            total = acc
        buf.append(unit)
        total += unit[2]
    flush()
    return chunks


def _embed_text(title: str, section: str, page: int, body: str) -> str:
    head = f"کتاب: {title}"
    if section:
        head += f" | بخش: {section}"
    head += f" | صفحه {page}"
    return f"{head}\n{body}"
