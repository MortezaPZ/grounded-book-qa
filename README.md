# Grounded Book QA

A fully local Persian question-answering system for your own documents. Answers are built only from uploaded files, every claim is cited to a page number, and nothing is sent to a remote API.

The rest of this README is in Persian.

پرسش و پاسخ فارسی روی اسناد شخصی. پاسخ فقط از روی فایل‌های بارگذاری‌شده ساخته می‌شود و هر ادعا با شماره صفحه مستند می‌گردد. همه‌چیز به‌صورت محلی اجرا می‌شود.

## اجرا

```powershell
.\run.ps1
```

سپس http://localhost:8000

## مدل زبانی

بازیابی (embedding + BM25 + reranker) کاملاً محلی است. برای تولید پاسخ، به ترتیب اولویت:

**۱. فایل GGUF در پوشه `models` (پیش‌فرض، بدون نیاز به هیچ نصب جانبی)**

هر فایل `.gguf` که در پوشه `models` بگذارید به‌صورت خودکار پیدا و استفاده می‌شود. روی ۸ گیگابایت رم:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-3B-Instruct-GGUF','qwen2.5-3b-instruct-q4_k_m.gguf',local_dir='models')"
```

اگر رم بیشتری در دسترس بود، `Qwen2.5-7B-Instruct-GGUF` با کیفیت فارسی بهتری می‌دهد. تنها کافی است فایل جدید را در `models` بگذارید؛ بزرگ‌ترین فایل انتخاب می‌شود.

**۲. Ollama** — اگر نصب باشد، خودکار ترجیح داده می‌شود:

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
```

**۳. سرور سازگار با OpenAI** (LM Studio، vLLM، llama.cpp server): `llm_backend` را روی `openai` و `openai_base_url` را تنظیم کنید.

بودجه حافظه روی ۸ گیگابایت: embedder حدود ۱.۱، reranker حدود ۱.۱ و مدل ۳ میلیاردی q4 حدود ۲.۲ گیگابایت. مجموع حدود ۴.۴ گیگابایت.

## خط فرمان

```powershell
.\.venv\Scripts\python.exe scripts\cli.py ingest "C:\path\to\book.pdf"
.\.venv\Scripts\python.exe scripts\cli.py ask "اثر شلاقی چیست؟"
.\.venv\Scripts\python.exe scripts\cli.py search "نقطه سفارش مجدد"
.\.venv\Scripts\python.exe scripts\cli.py status
```

`search` بدون مدل زبانی کار می‌کند و برای سنجش کیفیت بازیابی مفید است.

## محدودیت‌های شناخته‌شده

**PDF اسکن‌شده** پشتیبانی نمی‌شود. اگر فایل متن قابل استخراج نداشته باشد، سیستم هنگام بارگذاری هشدار می‌دهد و آن سند عملاً خالی نمایه می‌شود. برای فعال کردن OCR فارسی:

1. نصب Tesseract و افزودن `fas.traineddata`
2. `pip install pytesseract pillow`
3. `ocr_enabled` در `config.json` روی `true` باشد

**ترتیب معکوس «لا»** — بعضی تولیدکننده‌های PDF این حرف را وارونه ذخیره می‌کنند («بالا» به شکل «باال»). سیستم این حالت را هنگام بارگذاری تشخیص می‌دهد، هشدار می‌دهد و پرسش را با هر دو شکل جستجو می‌کند. متن نمایش‌داده‌شده در منابع همچنان همان شکل معیوب فایل اصلی است.

## تنظیمات

`config.json` (در صورت نبود، مقادیر پیش‌فرض `app/config.py` استفاده می‌شود). مهم‌ترین کلیدها:

| کلید | پیش‌فرض | توضیح |
|---|---|---|
| `embed_model` | `intfloat/multilingual-e5-base` | تغییر آن نیازمند نمایه‌سازی مجدد است |
| `rerank_model` | `BAAI/bge-reranker-base` | برای فارسی بهتر ولی سنگین‌تر: `BAAI/bge-reranker-v2-m3` |
| `use_reranker` | `true` | خاموش کردن، سرعت را چند برابر می‌کند |
| `chunk_tokens` / `chunk_overlap` | `320` / `64` | اندازه قطعه |
| `rerank_candidates` | `20` | تعداد نامزدهای بازرتبه‌بندی |
| `rerank_top_n` | `8` | تعداد منابع ارسالی به مدل |
| `multi_query` | `true` | بازنویسی خودکار پرسش برای بازیابی بهتر |
| `min_rerank_score` | `-6.0` | آستانه رد منابع بی‌ربط (بالاتر = سخت‌گیرتر) |

هر تغییری در `embed_model` یا `chunk_tokens` نیازمند اجرای مجدد ingest است.

## معماری

```
extract      PyMuPDF / python-docx، تشخیص ترتیب معکوس RTL، حذف سرصفحه تکراری، OCR اختیاری
persian      یکسان‌سازی ی/ک عربی، حذف اعراب، ZWNJ، ریشه‌یابی سبک، ایست‌واژه‌ها
chunker      قطعه‌بندی ساختارآگاه بر پایه فهرست کتاب + سرآیند متنی روی هر قطعه
store        بردار متراکم (numpy) + نمایه BM25 + فراداده، ذخیره روی دیسک
retriever    چند-پرسشی -> جستجوی متراکم + BM25 -> ادغام RRF -> بازرتبه‌بندی cross-encoder -> بسط همسایه
answer       اعلان مقید به منبع، استناد اجباری [S#]، پاک‌سازی استنادهای جعلی
server       FastAPI + SSE، رابط وب RTL
```

## License

MIT. See [LICENSE](LICENSE).
