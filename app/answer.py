"""Grounded answer synthesis with enforced citations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterator

from .config import Config
from .llm import BaseLLM, LLMError
from .persian import tokenize
from .retriever import RetrievalResult

NO_ANSWER_FA = "در منابع ارائه‌شده پاسخی برای این پرسش پیدا نکردم."

# A local 8B model asked to restate this corpus inverts it: «به جهنم نمی‌روند» came back out
# of the paraphrase as «به جهنم می‌روند». Selecting a sentence is a far easier job than
# restating one, and a quoted sentence cannot be flipped, so the answer is assembled from
# the source's own words. Measured on the same question, this recovered the two facts the
# rewriting prompts kept losing (the fifty thousand years, باقیات‌الصالحات) and dropped the
# inversion. The rewriting prompt is kept below for reference.
SYSTEM_PROMPT = """تو از «منابع» ارائه‌شده پاسخ می‌سازی، اما پاسخ را از نو نمی‌نویسی:
جمله‌های خودِ منابع را انتخاب می‌کنی و کنار هم می‌چینی.

روش کار:
۱. جمله‌هایی از منابع را که به پرسش کاربر مربوط‌اند پیدا کن و عیناً بیاور. کلمه‌ها را عوض نکن.
۲. متن منابع از سخنرانی پیاده شده و نقطه‌گذاری ندارد. اگر جمله‌ای خیلی بلند بود، فقط همان
تکه‌ای را که به پرسش مربوط است بیاور. جمله را کوتاه کن ولی کلمه‌هایش را تغییر نده.
۳. جمله‌های مربوط به هم را زیر یک تیتر کوتاه بگذار، مثل «در عالم برزخ» یا «در قیامت» یا «راه نجات».
۴. آخر هر بند شماره منبعش را بگذار، مثل [S1]. هیچ بندی بدون شماره منبع نماند.
۵. بین جمله‌ها فقط در حد یک عبارت کوتاه ربط بنویس. تحلیل و توضیح و نتیجه‌گیری از خودت ننویس.
۶. دو مطلب جدا را با «تا» و «برای اینکه» به هم وصل نکن و بینشان رابطه علت و معلولی نساز،
مگر آن رابطه صریحاً در خود منبع آمده باشد.
۷. این متن‌ها مردم را به «دسته»های جداگانه تقسیم می‌کنند. فقط جمله‌های مربوط به همان دسته‌ای
را بیاور که پرسش درباره اوست. اگر منبعی می‌گوید این دسته سرنوشتی را *ندارد*، آن نفی را دست‌نخورده بیاور.
۸. عددها، مدت‌زمان‌ها و تشبیه‌های منابع را حتماً بیاور.
۹. مقدمه ننویس. با تیتر اول شروع کن، نه با جمله‌ای مثل «بر اساس منابع ارائه‌شده».
۱۰. اگر هیچ‌کدام از منابع ربطی به پرسش نداشت، فقط بنویس: «{no_answer}»"""

REWRITE_SYSTEM_PROMPT = """تو یک دستیار پاسخگو هستی که بر پایه «منابع» ارائه‌شده به پرسش کاربر پاسخ می‌دهد.

روش کار:
۱. منابع را بخوان و هر چیزی که به پرسش مربوط است از داخل آنها بیرون بکش و پاسخ بده.
۲. منابع از فایل PDF استخراج شده‌اند، پس ممکن است جمله‌ها ناقص، بریده یا دارای نقطه‌گذاری جابه‌جا باشند.
این طبیعی است؛ منظور متن را برداشت کن و به‌خاطر بهم‌ریختگی ظاهری از پاسخ دادن خودداری نکن.
۳. همهٔ منابع را به کار بگیر، نه فقط منبع اول. اگر منابع به مرحله‌ها یا جنبه‌های متفاوتی می‌پردازند
(مثلاً یکی به عالم برزخ و دیگری به قیامت)، هر کدام را در بند جداگانه و با عنوان روشن بنویس.
۴. پاسخ را کامل بنویس و جزئیات مهمِ منابع را حذف نکن: نام‌ها، مثال‌ها، عددها و مدت‌زمان‌ها.
۵. پس از هر جمله‌ای که می‌نویسی شماره منبعی که از آن برداشت کرده‌ای را بیاور، مثل [S1] یا [S2][S4].
۶. فقط از محتوای منابع استفاده کن. از دانش عمومی خودت چیزی اضافه نکن و منبعی که وجود ندارد نساز.
۷. اگر منابع با هم متناقض‌اند، هر دو دیدگاه را با منبع خودش بیاور.
۸. فقط زمانی که هیچ‌کدام از منابع هیچ ارتباطی با پرسش ندارند، دقیقاً این جمله را بنویس:
«{no_answer}»
این جمله تنها وقتی مجاز است که کل پاسخ تو همین یک جمله باشد. هرگز آن را در ابتدا یا انتهای یک
پاسخ واقعی نیاور. اگر حتی بخشی از پاسخ در منابع هست، همان بخش را بنویس و ننویس که پاسخ پیدا نشد.
۹. اگر پرسش درباره فرایندی است که چند مرحله دارد، برای هر مرحله یک تیتر کوتاه بگذار
(مثلاً «در عالم برزخ»، «در قیامت»، «راه نجات») و مطالب همان مرحله را زیرش بیاور.
۱۰. این متن‌ها مردم را به «دسته»های جداگانه تقسیم می‌کنند و حکم هر دسته فرق دارد. اول مشخص کن
پرسش درباره کدام دسته است، و فقط حکم همان دسته را بنویس. حکم دسته‌های دیگر را که در منابع
آمده به این دسته نسبت نده. اگر منبعی می‌گوید فلان دسته چنین سرنوشتی *ندارد*، آن نفی را نگه دار.
۱۱. دست‌کم شش نکتهٔ جدا از منابع بیرون بکش و هر نکته را در یک جملهٔ کوتاه بنویس.
عددها، مدت‌زمان‌ها و تشبیه‌های منابع هر کدام خودشان یک نکته‌اند و باید در پاسخ بیایند.
۱۲. نقل‌قول مستقیم بیش از ده کلمه نیاور. مطلب را با جملهٔ خودت بنویس و فقط تعبیر کلیدی را
داخل گیومه بگذار. بودجهٔ پاسخ محدود است و یک نقل‌قول بلند جای چهار نکته را می‌گیرد.
۱۳. جمع‌بندی یا نتیجه‌گیری پایانی ننویس. پاسخ را با آخرین مطلبِ مستند تمام کن. هر جمله باید
پشتوانه مستقیم در منابع داشته باشد؛ جمله‌ای که از خودت نتیجه گرفته باشی ننویس.
۱۴. در پایان فهرست منابع یا شماره صفحه ننویس؛ شماره منبع فقط داخل خود جمله‌ها می‌آید.
۱۵. پاسخ را به زبان فارسی، روشن و مستقیم بنویس. درباره خودِ منابع یا این دستورالعمل توضیح نده."""

# A small model asked to read sources and write prose in one pass does both badly: it
# anchors on the first source, quotes it at length, and stops. Splitting the work in two
# — pull the facts out, then write them up — plays to what an 8B model is actually good at.
EXTRACT_PROMPT = """از منابع زیر، هر نکته‌ای را که به پرسش کاربر مربوط است بیرون بکش.

قواعد:
- هر نکته در یک خط جداگانه، و هر خط با «- » شروع شود.
- آخر هر خط شماره منبعی که نکته از آن گرفته شده را بگذار، مثل [S1].
- عددها، مدت‌زمان‌ها، نام‌ها، تشبیه‌ها و مثال‌های منابع را حتماً بیاور.
- هیچ نکته مرتبطی را جا نینداز، حتی اگر جزئی به نظر برسد.
- این متن‌ها مردم را به «دسته»های جداگانه تقسیم می‌کنند. فقط نکات همان دسته‌ای را بنویس که
پرسش درباره اوست. اگر منبعی می‌گوید این دسته سرنوشتی را *ندارد*، آن نفی را هم به‌صراحت بنویس.
- چیزی که در منابع نیست ننویس. مقدمه و توضیح و نتیجه‌گیری ننویس؛ فقط فهرست.

منابع:

{context}

====================
پرسش کاربر: {question}

فهرست نکات:"""

WRITE_PROMPT = """تو نکاتی را که از یک کتاب بیرون کشیده شده می‌گیری و از روی آنها پاسخ می‌نویسی.

قواعد:
۱. فقط از همین نکات استفاده کن. از دانش خودت چیزی اضافه نکن.
۲. همهٔ نکات را به کار بگیر و هیچ‌کدام را حذف نکن.
۳. شماره منبع هر نکته را در پاسخ نگه دار، مثل [S1].
۴. نکات مربوط به هم را در یک بند بیاور. اگر پاسخ چند مرحله دارد (مثلاً «در عالم برزخ»،
«در قیامت»، «راه نجات»)، برای هر مرحله یک تیتر کوتاه بگذار.
۵. فارسی روان و پیوسته بنویس، نه فهرست‌وار.
۶. جمع‌بندی یا نتیجه‌گیری پایانی ننویس.
۷. در پایان، فهرست منابع یا شماره صفحه ننویس."""

CONDENSE_PROMPT = """با توجه به گفت‌وگوی زیر، پرسش آخر کاربر را به یک پرسش کامل و مستقل بازنویسی کن
که بدون دیدن گفت‌وگو هم قابل فهم باشد. فقط خود پرسش بازنویسی‌شده را بنویس.

گفت‌وگو:
{history}

پرسش آخر: {question}"""

# Only an explicit S counts. A bare [3] is ambiguous — models write it for «مدل 3»
# and for list markers — and inventing a citation from it is worse than having none.
_CITE_RE = re.compile(r"\[\s*((?:[Ss]\s*\d{1,2}\s*[,،;؛]?\s*)+)\]")
_NUM_RE = re.compile(r"\d{1,2}")


@dataclass
class Source:
    label: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    section: str
    text: str
    score: float | None = None
    cited: bool = False

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "text": self.text,
            "score": self.score,
            "cited": self.cited,
        }


def _stitch(a: str, b: str) -> str:
    """Join two adjacent chunks, dropping the text they share at the seam.

    The seam can be as long as the whole of `a`: a transcript sentence runs to
    1200 chars, so the chunker's overlap tail may carry an entire chunk forward.
    A fixed window smaller than that silently falls through to the plain
    concatenation and restates the passage twice in the prompt.
    """
    window = min(len(a), len(b))
    for size in range(window, 30, -1):
        if a.endswith(b[:size]):
            return a + b[size:]
    return a + " " + b


def merge_adjacent(result: RetrievalResult) -> list:
    """Chunk overlap plus neighbour expansion returns the same passage several times.
    Contiguous chunks are stitched into one source so the model sees each passage once.
    """
    hits = list(result.hits)
    by_key = {(h.chunk["doc_id"], h.chunk["ordinal"]): h for h in hits}
    order = {id(h): i for i, h in enumerate(hits)}
    consumed: set[tuple[str, int]] = set()
    merged = []

    for h in hits:
        key = (h.chunk["doc_id"], h.chunk["ordinal"])
        if key in consumed:
            continue
        doc, start = key
        # walk back to the first chunk of this contiguous run
        while (doc, start - 1) in by_key and (doc, start - 1) not in consumed:
            start -= 1
        run = []
        ordn = start
        while (doc, ordn) in by_key and (doc, ordn) not in consumed:
            consumed.add((doc, ordn))
            run.append(by_key[(doc, ordn)])
            ordn += 1
        if not run:
            continue
        text = run[0].chunk["text"]
        for nxt in run[1:]:
            text = _stitch(text, nxt.chunk["text"])
        best = max(run, key=lambda x: (x.rerank if x.rerank is not None else x.fused))
        merged.append(
            {
                "text": text,
                "chunk": best.chunk,
                "page_start": min(r.chunk["page_start"] for r in run),
                "page_end": max(r.chunk["page_end"] for r in run),
                "score": best.rerank if best.rerank is not None else best.fused,
                "rank": min(order[id(r)] for r in run),
            }
        )
    merged.sort(key=lambda m: m["rank"])
    return merged


def build_sources(result: RetrievalResult, max_chars: int) -> list[Source]:
    sources: list[Source] = []
    used = 0
    for m in merge_adjacent(result):
        c = m["chunk"]
        body = m["text"]
        if used + len(body) > max_chars:
            room = max_chars - used
            if room < 400:
                break
            body = body[:room]
        used += len(body)
        sources.append(
            Source(
                label=f"S{len(sources) + 1}",
                doc_id=c["doc_id"],
                doc_title=c["doc_title"],
                page_start=m["page_start"],
                page_end=m["page_end"],
                section=c.get("section", ""),
                text=body,
                score=m["score"],
            )
        )
    return sources


def format_context(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        pages = f"صفحه {s.page_start}" if s.page_start == s.page_end else f"صفحات {s.page_start}-{s.page_end}"
        head = f"[{s.label}] «{s.doc_title}» — {pages}"
        if s.section:
            head += f" — {s.section}"
        blocks.append(f"{head}\n{s.text}")
    return "\n\n---\n\n".join(blocks)


_REFERRING = (
    "آن", "این", "همان", "اینها", "آنها", "او", "ایشان", "همین", "قبلی", "بالا",
    "مورد", "اولی", "دومی", "چرا", "چطور", "بیشتر", "دیگری", "دیگه",
)


def needs_condensing(question: str, history: list[dict]) -> bool:
    """A self-contained question must not be rewritten: a weak LLM will mangle it."""
    if not history:
        return False
    tokens = set(tokenize(question, remove_stopwords=False))
    return bool(tokens & set(_REFERRING)) or len(tokens) <= 3


def condense(question: str, history: list[dict], llm: BaseLLM) -> str:
    if not needs_condensing(question, history):
        return question
    convo = "\n".join(
        f"{'کاربر' if m['role'] == 'user' else 'دستیار'}: {m['content'][:400]}" for m in history[-4:]
    )
    try:
        out = llm.complete(
            [{"role": "user", "content": CONDENSE_PROMPT.format(history=convo, question=question)}],
            temperature=0.0,
            max_tokens=120,
        ).strip()
    except Exception:
        return question
    out = out.split("\n")[0].strip().strip('"«»')
    if not (3 < len(out) < 400):
        return question
    # a rewrite that keeps none of the original content words is a hallucinated question
    original = set(tokenize(question))
    if original and not (original & set(tokenize(out))):
        return question
    return out


def build_messages(question: str, sources: list[Source], cfg: Config) -> list[dict]:
    context = format_context(sources)
    user = (
        f"منابع:\n\n{context}\n\n"
        f"====================\n"
        f"پرسش کاربر: {question}\n\n"
        f"پاسخ را فقط بر پایه منابع بالا و با ذکر شماره منبع بنویس."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(no_answer=NO_ANSWER_FA)},
        {"role": "user", "content": user},
    ]


def build_write_messages(question: str, notes: str) -> list[dict]:
    return [
        {"role": "system", "content": WRITE_PROMPT},
        {"role": "user", "content": f"پرسش کاربر: {question}\n\nنکات:\n{notes}\n\nحالا پاسخ را بنویس."},
    ]


def extract_points(question: str, sources: list[Source], llm: BaseLLM, cfg: Config) -> str:
    """Pull the answer-relevant facts out of the sources as a citation-tagged list.

    Returns "" when the pass produced nothing usable, which puts the caller back on the
    single-pass path rather than writing an answer from notes that lost the content.
    """
    prompt = EXTRACT_PROMPT.format(context=format_context(sources), question=question)
    try:
        raw = llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=cfg.max_notes_tokens,
        )
    except Exception:
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip().startswith(("-", "•", "*"))]
    kept = [ln for ln in lines if cited_labels(ln, sources)]
    if len(kept) < 2:
        return ""
    return "\n".join(kept)


def cited_labels(text: str, sources: list[Source] | None = None) -> set[str]:
    valid = {s.label for s in sources} if sources is not None else None
    found: set[str] = set()
    for m in _CITE_RE.finditer(text):
        for n in _NUM_RE.findall(m.group(1)):
            label = f"S{n}"
            if valid is None or label in valid:
                found.add(label)
    return found


def trim_dangling(text: str) -> str:
    """Cutting a repetition mid-sentence leaves a stub; drop it if the rest is complete."""
    stripped = text.rstrip()
    if not stripped or stripped[-1] in ".!?؟:»)":
        return stripped
    head, sep, tail = stripped.rpartition("\n")
    if sep and len(tail.strip()) < 40:
        return head.rstrip()
    match = list(_SENT_END_RE.finditer(stripped))
    if match and len(stripped) - match[-1].end() < 40:
        return stripped[: match[-1].end()].rstrip()
    return stripped


# The model rarely reproduces NO_ANSWER_FA verbatim — it rewrites the opening
# («بر اساس منابع…» for «در منابع…») and bolts the result onto a real answer as a
# sign-off. Matching the invariant tail catches every variant we have seen.
_NO_ANSWER_RE = re.compile(r"[^\n.!?؟]*پاسخ[‌ی]*\s*برای\s+این\s+پرسش\s+پیدا\s+نکردم\s*[.؟!]?")


def drop_stray_no_answer(text: str) -> str:
    """The refusal sentence is only meaningful as the whole answer.

    Left in place next to real content it reads as «here is the answer, and also
    there is no answer» — worse than either half on its own.
    """
    if not _NO_ANSWER_RE.search(text):
        return text
    stripped = _NO_ANSWER_RE.sub("", text)
    # what survives has to be an answer, not punctuation and a stray connective
    if len(re.sub(r"[\s\W_]+", "", stripped)) < 40:
        return NO_ANSWER_FA
    lines = [ln for ln in stripped.split("\n") if re.sub(r"[\s\W_]+", "", ln)]
    return "\n".join(lines).strip()


_PREAMBLE_RE = re.compile(
    r"^\s*(?:در|بر\s*اساس|طبق|با\s*توجه\s*به)[^\n:.]{0,40}(?:منابع|منبع)[^\n:.]{0,60}[:.]\s*"
)


def strip_preamble(text: str) -> str:
    """Drop the «بر اساس منابع ارائه‌شده...» throat-clearing the model opens with."""
    return _PREAMBLE_RE.sub("", text, count=1).lstrip()


def attribute_uncited(text: str, sources: list[Source], min_ratio: float = 0.6) -> str:
    """Tag a paragraph that quotes a source but forgot to name it.

    The model drops the label on its last paragraph often enough to matter. Attribution is
    only added when the paragraph's words really do come from one source, so a paragraph the
    model invented stays visibly uncited rather than being handed a citation it never earned.
    """
    by_label = {s.label: set(tokenize(s.text)) for s in sources}
    out = []
    for para in text.split("\n\n"):
        body = para.strip()
        if not body or cited_labels(body, sources):
            out.append(para)
            continue
        words = set(tokenize(body))
        if not words:
            out.append(para)
            continue
        label, ratio = max(
            ((lab, len(words & toks) / len(words)) for lab, toks in by_label.items()),
            key=lambda x: x[1],
            default=("", 0.0),
        )
        out.append(f"{para.rstrip()} [{label}]" if ratio >= min_ratio else para)
    return "\n\n".join(out)


def finalize(text: str, sources: list[Source]) -> str:
    """Every caller shows the user the same string; keep the order in one place."""
    return attribute_uncited(trim_dangling(sanitize(strip_preamble(text), sources)), sources)


def sanitize(text: str, sources: list[Source]) -> str:
    """Normalise citation shapes to [S#] and drop ones pointing at missing sources."""
    text = drop_stray_no_answer(text)
    valid = {s.label for s in sources}

    def repl(m: re.Match) -> str:
        group = m.group(1)
        labels = [f"S{n}" for n in _NUM_RE.findall(group)]
        kept = [lab for lab in labels if lab in valid]
        return "".join(f"[{lab}]" for lab in kept)

    return _CITE_RE.sub(repl, text)


_SENT_END_RE = re.compile(r"[.!?؟\n]")

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def strip_think(stream: Iterator[str]) -> Iterator[str]:
    """Drop the reasoning block a Qwen3-style model emits before its answer.

    Runs on the live stream, so the tags may be split across chunks: whatever could
    still be the start of a tag is held back until the next piece arrives.
    """
    buf, inside = "", False
    for piece in stream:
        buf += piece
        out: list[str] = []
        while True:
            if inside:
                i = buf.find(_THINK_CLOSE)
                if i < 0:
                    buf = buf[-(len(_THINK_CLOSE) - 1) :]
                    break
                buf = buf[i + len(_THINK_CLOSE) :]
                inside = False
                continue
            i = buf.find(_THINK_OPEN)
            if i < 0:
                hold = len(_THINK_OPEN) - 1
                if len(buf) > hold:
                    out.append(buf[:-hold])
                    buf = buf[-hold:]
                break
            out.append(buf[:i])
            buf = buf[i + len(_THINK_OPEN) :]
            inside = True
        joined = "".join(out)
        if joined:
            yield joined
    if buf and not inside:
        yield buf


def _looks_repetitive(seen: Counter, sentence: str) -> bool:
    """A small model that starts looping repeats whole sentences verbatim."""
    key = " ".join(sentence.split())
    if len(key) < 25:
        return False
    seen[key] += 1
    return seen[key] >= 2


def stream_answer(
    question: str, sources: list[Source], llm: BaseLLM, cfg: Config
) -> Iterator[str]:
    if not sources:
        yield NO_ANSWER_FA
        return
    notes = extract_points(question, sources, llm, cfg) if cfg.two_stage_answer else ""
    messages = build_write_messages(question, notes) if notes else build_messages(question, sources, cfg)
    seen: Counter = Counter()
    pending = ""
    try:
        for piece in strip_think(
            llm.stream(
                messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_answer_tokens,
                repeat_penalty=cfg.repeat_penalty,
                frequency_penalty=cfg.frequency_penalty,
            )
        ):
            pending += piece
            yield piece
            if not _SENT_END_RE.search(piece):
                continue
            parts = _SENT_END_RE.split(pending)
            pending = parts.pop()
            if any(_looks_repetitive(seen, p) for p in parts):
                return
    except LLMError as e:
        yield f"\n\n[خطای مدل: {e}]"
