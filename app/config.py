from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

# the Xet transfer backend stalls on some Windows networks; plain HTTP is reliable
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "config.json"


def find_gguf() -> str:
    if not MODELS_DIR.exists():
        return ""
    files = sorted(MODELS_DIR.rglob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    return str(files[0]) if files else ""


@dataclass
class Config:
    embed_model: str = "intfloat/multilingual-e5-base"
    # bge-reranker-base scored 1/4 on Persian and ranked an unrelated question above a
    # correct answer; v2-m3 scored 4/4 with a ~4000x margin. The 2.3GB is worth it.
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = True

    chunk_tokens: int = 256
    chunk_overlap: int = 64
    min_chunk_chars: int = 120

    dense_top_k: int = 40
    sparse_top_k: int = 40
    fusion_top_k: int = 40
    rerank_candidates: int = 12  # v2-m3 costs ~0.3s per candidate on CPU
    rerank_top_n: int = 4  # small models lose the instruction when given many passages
    # chunk boundaries cut mid-sentence on messy PDF text, so the top hit often starts
    # mid-thought; the neighbour restores the beginning of the sentence
    neighbor_window: int = 1
    # sentence-transformers applies Sigmoid to these rerankers, so both thresholds
    # live in 0..1, not in logit space. Measured on Persian with v2-m3: correct
    # answers scored 0.39-0.92, unrelated questions topped out at 0.0001.
    min_rerank_score: float = 0.001
    # below these the corpus is treated as having no answer at all, and the model is
    # never asked, so it cannot confabulate from irrelevant context. The reranker
    # decides when it ran; the embedder threshold is only the fallback, since e5
    # similarities sit in a narrow band and separate poorly on their own.
    abstain_below: float = 0.002
    abstain_dense_below: float = 0.76

    # each expansion is an extra LLM call; on a local CPU model that costs ~150s per
    # question and measured no recall gain, so it stays off unless a fast LLM is used
    multi_query: bool = False
    multi_query_count: int = 2
    # rewriting a follow-up into a standalone question needs a capable LLM; a small
    # local model mangles it, so only do it when the question actually refers back
    condense_history: bool = True

    llm_backend: str = "auto"  # auto | ollama | openai | llamacpp | transformers
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    llamacpp_model_path: str = ""
    hf_model: str = ""  # optional transformers fallback, e.g. Qwen/Qwen2.5-1.5B-Instruct

    # Extract-then-write, measured on Dorna-8B-q4: strictly worse than one pass. The
    # extraction step invents claims the sources contradict, the writing step then
    # propagates them, and it drops every [S#] on the way, so the answer arrives with no
    # citations at all. Kept behind the flag because it is a real win on a stronger model;
    # on an 8B the second pass has no capacity left to check the first.
    two_stage_answer: bool = False
    max_notes_tokens: int = 700

    max_context_chars: int = 9000
    temperature: float = 0.1
    max_answer_tokens: int = 1000
    # small models fall into repetition loops on structured documents; 1.05 was far
    # too weak to break them and llama.cpp's own default is 1.1
    repeat_penalty: float = 1.18
    frequency_penalty: float = 0.4
    answer_language: str = "fa"

    ocr_enabled: bool = True
    ocr_lang: str = "fas+eng"

    device: str = "cpu"
    embed_batch_size: int = 8
    # the reranker and the LLM never run at the same time; dropping the reranker before
    # generation frees ~2.3GB so a larger, better model fits in RAM
    unload_reranker_before_answer: bool = True

    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        if CONFIG_PATH.exists():
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k, v in raw.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        for k in asdict(cfg):
            env = os.environ.get("RAG_" + k.upper())
            if env is None:
                continue
            cur = getattr(cfg, k)
            if isinstance(cur, bool):
                setattr(cfg, k, env.strip().lower() in ("1", "true", "yes", "on"))
            elif isinstance(cur, int):
                setattr(cfg, k, int(env))
            elif isinstance(cur, float):
                setattr(cfg, k, float(env))
            elif isinstance(cur, str):
                setattr(cfg, k, env)
        return cfg

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def ensure_dirs() -> None:
    for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
