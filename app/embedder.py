"""Dense embedding and cross-encoder reranking wrappers."""

from __future__ import annotations

import threading

import numpy as np

from .config import MODELS_DIR

_LOCK = threading.Lock()


def resolve(model_name: str) -> str:
    """Prefer a copy under models/ so the app runs with no network at all."""
    local = MODELS_DIR / model_name.split("/")[-1]
    if (local / "config.json").exists():
        return str(local)
    return model_name


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 8):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._needs_prefix = "e5" in model_name.lower()

    @property
    def model(self):
        if self._model is None:
            with _LOCK:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(resolve(self.model_name), device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(getter())

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.model.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return max(1, int(len(text.split()) * 1.35))

    def encode_passages(self, texts: list[str], progress=None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prepared = [f"passage: {t}" if self._needs_prefix else t for t in texts]
        out: list[np.ndarray] = []
        step = max(self.batch_size * 8, 32)
        for i in range(0, len(prepared), step):
            batch = prepared[i : i + step]
            vecs = self.model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            out.append(vecs.astype(np.float32))
            if progress:
                progress(min(i + step, len(prepared)), len(prepared))
        return np.vstack(out)

    def encode_query(self, text: str) -> np.ndarray:
        prepared = f"query: {text}" if self._needs_prefix else text
        vec = self.model.encode(
            [prepared], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vec.astype(np.float32)[0]


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self.available = True

    @property
    def model(self):
        if self._model is None:
            with _LOCK:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(
                        resolve(self.model_name), device=self.device, max_length=512
                    )
        return self._model

    def unload(self) -> bool:
        """Free the cross-encoder so a larger LLM fits. It reloads on next use."""
        if self._model is None:
            return False
        with _LOCK:
            self._model = None
        import gc

        gc.collect()
        return True

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        try:
            pairs = [(query, p[:2000]) for p in passages]
            scores = self.model.predict(pairs, batch_size=4, show_progress_bar=False)
            return [float(s) for s in scores]
        except Exception:
            self.available = False
            return [0.0] * len(passages)
