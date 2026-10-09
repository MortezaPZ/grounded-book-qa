# Grounded Book QA

Ask questions about your own documents and get an answer that cites a page. Retrieval stays on the machine. Nothing is sent to a remote API unless you configure a local OpenAI-compatible server yourself.

The default models and text normalization are aimed at Persian documents. This overview is in English. Questions should be asked in the language of the uploaded files.

## Overview

PDFs and Word files are extracted, chunked, and indexed with a dense embedding plus BM25. A cross-encoder reranks the hits. The answer is constrained to those passages.

## Features

- Local embedding, BM25, and reranking
- Page-level citations
- Optional GGUF model in `models/`, or Ollama, or an OpenAI-compatible local server
- CLI for ingest, search, and ask
- FastAPI server and a web UI
- Optional OCR for scanned PDFs when Tesseract is installed

## Technology Stack

- Python 3.11
- FastAPI
- PyMuPDF and python-docx
- sentence-transformers and PyTorch (CPU)
- NumPy for the dense index

## Architecture

```
extract     PyMuPDF / python-docx, repeated header removal, optional OCR
normalize   Arabic and Persian letter forms, diacritics, light stemming, stop words
chunker     structure-aware chunks from the book outline, with a text header on each chunk
store       dense vectors (NumPy) + BM25 + metadata on disk
retriever   multi-query, dense search + BM25, RRF merge, cross-encoder rerank, neighbor expansion
answer      source-bound prompt, required [S#] citations, fabricated citations stripped
server      FastAPI + SSE and a web UI
```

## Installation

```powershell
.\run.ps1
```

Then open `http://localhost:8000`.

Answer generation, in order:

1. Any `.gguf` file in `models/`. The largest file is selected. On an 8 GB machine, a 3B Q4 model is the practical default. A 7B model is better if you have the RAM. Download example:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-3B-Instruct-GGUF','qwen2.5-3b-instruct-q4_k_m.gguf',local_dir='models')"
```

2. Ollama, if it is installed:

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
```

3. An OpenAI-compatible server (LM Studio, vLLM, llama.cpp server): set `llm_backend` to `openai` and set `openai_base_url`.

Rough 8 GB budget: embedder about 1.1 GB, reranker about 1.1 GB, a 3B Q4 model about 2.2 GB.

## Usage

```powershell
.\.venv\Scripts\python.exe scripts\cli.py ingest "C:\path\to\book.pdf"
.\.venv\Scripts\python.exe scripts\cli.py ask "What is the bullwhip effect?"
.\.venv\Scripts\python.exe scripts\cli.py search "reorder point"
.\.venv\Scripts\python.exe scripts\cli.py status
```

`search` does not need a language model. Use it to judge retrieval on its own.

`config.json` overrides the defaults in `app/config.py`.

| Key | Default | Notes |
| --- | --- | --- |
| `embed_model` | `intfloat/multilingual-e5-base` | Changing it requires a new index |
| `rerank_model` | `BAAI/bge-reranker-base` | `BAAI/bge-reranker-v2-m3` is stronger and heavier |
| `use_reranker` | `true` | Turning it off is much faster |
| `chunk_tokens` / `chunk_overlap` | `320` / `64` | Chunk size |
| `rerank_candidates` | `20` | Candidates before rerank |
| `rerank_top_n` | `8` | Passages sent to the model |
| `multi_query` | `true` | Rewrite the question for retrieval |
| `min_rerank_score` | `-6.0` | Higher rejects more weak hits |

Changing `embed_model` or `chunk_tokens` requires a fresh ingest.

## Testing

There is no unit suite. `scripts/cli.py search` is the local retrieval check.

## Limitations

Scanned PDFs are not indexed unless OCR is enabled: install Tesseract and `fas.traineddata`, `pip install pytesseract pillow`, and set `ocr_enabled` to `true` in `config.json`.

Some PDF generators store one Persian letter pair in reverse order. The loader warns and searches both forms. The passage shown to the user is still the original file text.

The production embedder and reranker are large downloads. A smaller local model is enough to prove the pipeline.

## License

MIT. See [LICENSE](LICENSE).
