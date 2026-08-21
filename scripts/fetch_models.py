"""Download the local models over a flaky connection.

Uses HTTP Range requests so a dropped transfer resumes where it stopped;
huggingface_hub restarts from zero, which never finishes a multi-GB file.
Re-running is cheap: completed files are skipped.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MODELS_DIR, Config  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

HF = "https://huggingface.co"

RERANKER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]
EMBEDDER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
]


def download(url: str, dest: Path, attempts: int = 300, quiet: bool = False) -> Path:
    import httpx

    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    total = 0
    with httpx.Client(follow_redirects=True, timeout=30) as c:
        h = c.head(url)
        if h.status_code == 404:
            raise FileNotFoundError(url)
        h.raise_for_status()
        total = int(h.headers.get("content-length", 0))

    if not quiet:
        print(f"   {dest.name}: {total / 1e6:.0f} MB")

    t0 = time.time()
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if total and have >= total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30, read=60)) as c:
                with c.stream("GET", url, headers=headers) as r:
                    # the server refuses a range past the end, which means we already have it
                    if r.status_code == 416 and have:
                        break
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    mode = "ab" if have and r.status_code == 206 else "wb"
                    if mode == "wb":
                        have = 0
                    last = time.time()
                    with open(part, mode) as f:
                        for chunk in r.iter_bytes(512 * 1024):
                            f.write(chunk)
                            have += len(chunk)
                            if not quiet and total > 50e6 and time.time() - last > 15:
                                mbps = have / 1e6 / max(time.time() - t0, 1)
                                print(
                                    f"      {have / 1e9:.2f}/{total / 1e9:.2f} GB"
                                    f"  ({100 * have / total:.0f}%, {mbps:.1f} MB/s)"
                                )
                                last = time.time()
        except Exception as e:
            print(f"      retry {attempt}: {e.__class__.__name__} — resuming at {have / 1e6:.0f} MB")
            time.sleep(min(2 * attempt, 15))
            continue
        # the stream ended without error, so the file is complete even when the
        # server sent no content-length to compare against
        break

    size = part.stat().st_size if part.exists() else 0
    if total and size != total:
        raise RuntimeError(f"{dest.name}: got {size} of {total} bytes")
    part.replace(dest)
    return dest


def weight_file(repo: str) -> str:
    """Prefer safetensors, fall back to the pytorch bin."""
    import httpx

    with httpx.Client(follow_redirects=True, timeout=20) as c:
        for name in ("model.safetensors", "pytorch_model.bin"):
            if c.head(f"{HF}/{repo}/resolve/main/{name}").status_code == 200:
                return name
    raise RuntimeError(f"no weight file found for {repo}")


def fetch_repo(repo: str, files: list[str], dest_dir: Path) -> Path:
    print(f"\n== {repo}")
    t0 = time.time()
    for name in files:
        try:
            download(f"{HF}/{repo}/resolve/main/{name}", dest_dir / name, quiet=True)
        except FileNotFoundError:
            pass
    download(f"{HF}/{repo}/resolve/main/{weight_file(repo)}", dest_dir / weight_file(repo))
    print(f"   ready in {time.time() - t0:.0f}s -> {dest_dir}")
    return dest_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["embed", "rerank", "llm"], nargs="*")
    ap.add_argument("--rerank-repo", help="override the reranker to fetch")
    ap.add_argument("--gguf-repo", default="Qwen/Qwen2.5-3B-Instruct-GGUF")
    ap.add_argument("--gguf-file", default="qwen2.5-3b-instruct-q4_k_m.gguf")
    args = ap.parse_args()
    want = set(args.only or ["embed", "rerank", "llm"])

    cfg = Config.load()
    if "embed" in want:
        fetch_repo(cfg.embed_model, EMBEDDER_FILES, MODELS_DIR / cfg.embed_model.split("/")[-1])
    if "rerank" in want:
        repo = args.rerank_repo or cfg.rerank_model
        fetch_repo(repo, RERANKER_FILES, MODELS_DIR / repo.split("/")[-1])
    if "llm" in want:
        print(f"\n== {args.gguf_repo}/{args.gguf_file}")
        download(f"{HF}/{args.gguf_repo}/resolve/main/{args.gguf_file}", MODELS_DIR / args.gguf_file)
        print("   ready")

    print("\nall models ready")


if __name__ == "__main__":
    main()
