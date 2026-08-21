"""Pluggable local-first LLM backends with streaming."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterator

import httpx

from .config import Config, find_gguf


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"
    model = ""

    def stream(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 1400) -> Iterator[str]:
        raise NotImplementedError

    def complete(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 1400) -> str:
        return "".join(self.stream(messages, temperature, max_tokens))

    def health(self) -> tuple[bool, str]:
        return True, "ok"


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model

    def health(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=3)
            r.raise_for_status()
            tags = [m["name"] for m in r.json().get("models", [])]
            if not tags:
                return False, "Ollama is running but no model is installed"
            if self.model not in tags and self.model.split(":")[0] not in [t.split(":")[0] for t in tags]:
                return False, f"model '{self.model}' not found. installed: {', '.join(tags)}"
            return True, f"ollama: {self.model}"
        except Exception as e:
            return False, f"ollama unreachable at {self.url} ({e.__class__.__name__})"

    def stream(self, messages, temperature=0.1, max_tokens=1400):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 8192,
                "repeat_penalty": 1.05,
            },
        }
        with httpx.stream("POST", f"{self.url}/api/chat", json=payload, timeout=None) as r:
            if r.status_code >= 400:
                raise LLMError(f"ollama error {r.status_code}: {r.read().decode('utf-8', 'ignore')[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    raise LLMError(str(obj["error"]))
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if obj.get("done"):
                    break


class OpenAICompatLLM(BaseLLM):
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def health(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            if r.status_code >= 400:
                return False, f"{self.base_url} returned {r.status_code}"
            return True, f"openai-compatible: {self.model}"
        except Exception as e:
            return False, f"{self.base_url} unreachable ({e.__class__.__name__})"

    def stream(self, messages, temperature=0.1, max_tokens=1400):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with httpx.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers(), timeout=None
        ) as r:
            if r.status_code >= 400:
                raise LLMError(f"llm error {r.status_code}: {r.read().decode('utf-8', 'ignore')[:300]}")
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if piece:
                    yield piece


class LlamaCppLLM(BaseLLM):
    """In-process GGUF backend. Fully offline once the model file is on disk."""

    name = "llamacpp"

    def __init__(self, model_path: str, n_ctx: int = 8192, threads: int = 0):
        self.model_path = model_path
        self.model = Path(model_path).name if model_path else ""
        self.n_ctx = n_ctx
        self.threads = threads or (os.cpu_count() or 4)
        self._llm = None
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    def health(self) -> tuple[bool, str]:
        if not self.model_path or not Path(self.model_path).exists():
            return False, "هیچ فایل GGUF در پوشه models پیدا نشد"
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            return False, "llama-cpp-python نصب نیست"
        return True, f"llama.cpp: {self.model}"

    @property
    def llm(self):
        if self._llm is None:
            with self._load_lock:
                if self._llm is None:
                    from llama_cpp import Llama

                    # verbose=False makes llama-cpp-python redirect stderr at the OS
                    # level, which deadlocks on Windows and hangs the first request
                    self._llm = Llama(
                        model_path=self.model_path,
                        n_ctx=self.n_ctx,
                        n_threads=self.threads,
                        verbose=True,
                    )
        return self._llm

    def stream(self, messages, temperature=0.1, max_tokens=1400, **opts):
        model = self.llm  # load outside the generation lock; the two are not the same lock
        with self._gen_lock:
            for part in model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                repeat_penalty=opts.get("repeat_penalty", 1.15),
                frequency_penalty=opts.get("frequency_penalty", 0.0),
                stream=True,
            ):
                piece = (part["choices"][0].get("delta") or {}).get("content") or ""
                if piece:
                    yield piece


class TransformersLLM(BaseLLM):
    """In-process HF backend so the app works with no extra runtime installed."""

    name = "transformers"

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = model_name
        self.device = device
        self._pipe = None
        self._lock = threading.Lock()

    def health(self) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
        except Exception:
            return False, "transformers is not installed"
        return True, f"transformers: {self.model} (CPU, کند)"

    def _load(self):
        if self._pipe is None:
            with self._lock:
                if self._pipe is None:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    tok = AutoTokenizer.from_pretrained(self.model)
                    mdl = AutoModelForCausalLM.from_pretrained(
                        self.model, dtype=torch.float32, low_cpu_mem_usage=True
                    )
                    mdl.eval()
                    self._pipe = (tok, mdl)
        return self._pipe

    def stream(self, messages, temperature=0.1, max_tokens=1400):
        from threading import Thread

        import torch
        from transformers import TextIteratorStreamer

        tok, mdl = self._load()
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.01,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
        )
        def run():
            with torch.inference_mode():
                mdl.generate(**kwargs)

        Thread(target=run, daemon=True).start()
        for piece in streamer:
            if piece:
                yield piece


class NullLLM(BaseLLM):
    name = "none"

    def __init__(self, reason: str):
        self.reason = reason

    def health(self):
        return False, self.reason

    def stream(self, messages, temperature=0.1, max_tokens=1400):
        raise LLMError(self.reason)


def build_llm(cfg: Config) -> BaseLLM:
    backend = (cfg.llm_backend or "auto").lower()
    candidates: list[BaseLLM] = []
    if backend in ("ollama", "auto"):
        candidates.append(OllamaLLM(cfg.ollama_url, cfg.ollama_model))
    if backend in ("openai", "auto") and (cfg.openai_api_key or backend == "openai"):
        candidates.append(OpenAICompatLLM(cfg.openai_base_url, cfg.openai_model, cfg.openai_api_key))
    if backend in ("llamacpp", "auto"):
        gguf = cfg.llamacpp_model_path or find_gguf()
        if gguf:
            candidates.append(LlamaCppLLM(gguf))
    if backend in ("transformers", "auto") and cfg.hf_model:
        candidates.append(TransformersLLM(cfg.hf_model, cfg.device))

    problems = []
    for c in candidates:
        ok, msg = c.health()
        if ok:
            return c
        problems.append(f"{c.name}: {msg}")
    return NullLLM(
        "هیچ مدل زبانی در دسترس نیست. " + (" | ".join(problems) if problems else "no backend configured")
    )
