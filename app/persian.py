"""Persian/Arabic text normalization and tokenization for retrieval."""

from __future__ import annotations

import re
import unicodedata

ZWNJ = "‌"

_CHAR_MAP = {
    "ي": "ی",  # arabic yeh -> persian yeh
    "ى": "ی",  # alef maksura -> persian yeh
    "ے": "ی",  # yeh barree
    "ك": "ک",  # arabic kaf -> persian kaf
    "ڪ": "ک",
    "ة": "ه",  # teh marbuta -> heh
    "ۀ": "ه",  # heh with yeh above
    "ؤ": "و",  # waw with hamza
    "ئ": "ی",  # yeh with hamza
    "إ": "ا",  # alef with hamza below
    "أ": "ا",  # alef with hamza above
    "ٱ": "ا",  # alef wasla
    "ڤ": "ف",
    "ـ": "",  # tatweel / kashida
    " ": " ",
    "‎": "",  # LRM
    "‏": "",  # RLM
    "‪": "",
    "‫": "",
    "‬": "",
    "‭": "",
    "‮": "",
    "﻿": "",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}

# tashkeel / harakat and quranic marks
for _cp in list(range(0x064B, 0x0660)) + [0x0670] + list(range(0x06D6, 0x06ED + 1)):
    _CHAR_MAP[chr(_cp)] = ""

_DIGIT_MAP = {}
for _i in range(10):
    _DIGIT_MAP[chr(0x0660 + _i)] = str(_i)  # arabic-indic
    _DIGIT_MAP[chr(0x06F0 + _i)] = str(_i)  # extended arabic-indic (persian)

_TRANS_CHARS = str.maketrans(_CHAR_MAP)
_TRANS_DIGITS = str.maketrans(_DIGIT_MAP)

_WS_RE = re.compile(r"[ \t ]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_ZWNJ_LONE_RE = re.compile(r"[ \t]+" + ZWNJ + r"+[ \t]+")
_ZWNJ_SPACE_RE = re.compile(r"[ \t]*" + ZWNJ + r"+[ \t]*")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

PERSIAN_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_TOKEN_RE = re.compile(r"[؀-ۿ" + ZWNJ + r"A-Za-z0-9]+")

STOPWORDS = set(
    """
و در به از که این را با است برای آن یک شد های می هم یا اما اگر همه نیز
هر ما شما آنها او وی من تو ایشان بود بودن باشد باشند شود شوند کرد کردن کند کنند کنیم
دارد دارند داشت داشتن ندارد نیست نبود هست هستند بی جز چه چون چرا کجا کی چگونه بله خیر
نه آیا فقط باید نباید حتی دیگر دیگری دیگران بین روی زیر بالا پایین پیش پس قبل بعد
مانند مثل طبق برابر درباره بدون علیه سوی سمت نزد کنار میان ضمن بنابراین لذا ولی بلکه
همین همان چنین چنان همچنین همچنان وقتی هنگام هنگامی زمانی جای جایی طور نحو گونه
شده شدن گرفت گرفته داده دهد دهند آورد آورده یافت یافته توان تواند توانند کل تمام تمامی
بسیار خیلی کم بیش بیشتر کمتر اول دوم سوم آخر ابتدا انتها ای ها هایی تر ترین
""".split()
)

ENGLISH_STOPWORDS = set(
    """
the a an and or of to in for on at by with from is are was were be been being as that this
these those it its if not no but so such than then there their they we you he she i
""".split()
)

_SUFFIXES = (
    "هایمان", "هایتان", "هایشان", "هایی", "هایم", "هایت", "هایش", "های", "ها",
    "ترین", "تر", "مان", "تان", "شان",
)


def normalize(text: str) -> str:
    """Light normalization: safe for display, embedding and citation snippets."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL_RE.sub("", text)
    text = text.translate(_TRANS_CHARS)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _ZWNJ_LONE_RE.sub(" ", text)
    text = _ZWNJ_SPACE_RE.sub(ZWNJ, text)
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def fold_for_match(text: str) -> str:
    """Aggressive folding used only for lexical matching."""
    text = normalize(text).translate(_TRANS_DIGITS)
    text = text.replace("آ", "ا")  # alef madda -> alef
    text = swap_lam_alef(text)  # canonical side of the reversed-ligature pair
    text = text.replace(ZWNJ, " ")
    return text.lower()


def swap_lam_alef(text: str) -> str:
    """Mirror the lam-alef ligature. Some PDF producers emit «لا» as «ال»."""
    return text.replace("لا", "ال")


def lam_alef_broken(text: str) -> bool:
    """True when a Persian text shows the reversed-ligature signature.

    Any substantial Persian text contains «لا» (بالا، خلاصه، تلاش، اطلاعات). Its total
    absence alongside plenty of «ال», or a run of double-alef, means the producer
    reversed the ligature.
    """
    if len(PERSIAN_RE.findall(text)) < 1500:
        return False
    if text.count("اا") >= 3:
        return True
    return text.count("لا") == 0 and text.count("ال") >= 5


def stem(token: str) -> str:
    if len(token) <= 3:
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def tokenize(text: str, remove_stopwords: bool = True, do_stem: bool = True) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(fold_for_match(text)):
        tok = tok.strip(ZWNJ)
        if not tok or (len(tok) == 1 and not tok.isdigit()):
            continue
        if remove_stopwords and (tok in STOPWORDS or tok in ENGLISH_STOPWORDS):
            continue
        out.append(stem(tok) if do_stem else tok)
    return out


def is_persian(text: str, threshold: float = 0.2) -> bool:
    if not text:
        return False
    return len(PERSIAN_RE.findall(text)) / max(len(text), 1) >= threshold


_COMMON_FA = ("این", "که", "برای", "است", "های", "شود", "بود", "کرد", "می")


def fix_visual_order(text: str) -> str:
    """Some PDFs store Arabic-script glyphs in visual order; detect and repair per line."""
    if not text.strip():
        return text
    sample = text[:4000]
    fwd = sum(sample.count(w) for w in _COMMON_FA)
    rev = sum(sample[::-1].count(w) for w in _COMMON_FA)
    if rev > fwd * 2 and rev >= 3:
        return "\n".join(line[::-1] for line in text.split("\n"))
    return text
