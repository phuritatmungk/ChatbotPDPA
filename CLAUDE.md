# CLAUDE.md

Project context for Claude Code sessions on this repo.

## What this is

PDPA chatbot (Thailand's Personal Data Protection Act). Streamlit UI + LangGraph agentic RAG over Qdrant Cloud, using SCB10X Typhoon LLM and BGE reranker.

- **Repo**: https://github.com/phuritatmungk/ChatbotPDPA (branch `main`)
- **Live**: https://chatbotpdpa3119.streamlit.app
- **Local path**: `C:\P041-การพัฒนาแชตบอทสำหรับกฎหมายPDPA`
- **GitHub user**: phuritatmungk
- **Commit author**: `phuritatmungk <chaitat001@gmail.com>`

## Deployment architecture

Streamlit Cloud has ~1GB RAM, which can't load `BAAI/bge-reranker-v2-m3` (~2GB) or run a local LLM. To work around this, **two services run on the user's PC** and are exposed via ngrok tunnels:

```
Streamlit Cloud  ──HTTPS──►  ngrok  ──►  localhost:8080  (llama-server: Typhoon 2.5 Qwen3 4B)
                                       └─►  localhost:8090  (rerank_server.py: BGE v2-m3)

Streamlit Cloud  ──HTTPS──►  Qdrant Cloud (embeddings stored there)
Streamlit Cloud  loads MiniLM-L12 embedder locally (~120MB, fits in 1GB)
```

The embedder runs on Streamlit Cloud. The LLM and reranker run on the user's PC.

## Running the stack locally / for cloud demo

The user must keep their PC on with these processes alive. From the project dir:

```bat
:: Terminal 1 — LLM server (Typhoon)
llm.bat
:: → C:\llama\llama-server.exe -hf scb10x/typhoon2.5-qwen3-4b-gguf:Q4_K_M -c 8192

:: Terminal 2 — Rerank server (BGE)
.venv\Scripts\python.exe rerank_server.py
:: → loads BAAI/bge-reranker-v2-m3 on port 8090

:: Terminal 3 — ngrok with both tunnels (see C:\Users\chait\AppData\Local\ngrok\ngrok.yml)
ngrok start --all
```

ngrok config (`%LOCALAPPDATA%\ngrok\ngrok.yml`) defines two endpoints: `llama` (→8080) and `rerank` (→8090). URLs change every ngrok restart — grab them from `http://localhost:4040/api/tunnels`.

After URLs change, update **Streamlit Cloud Secrets** (Settings → Secrets) with the new `LLM_BASE_URL` and `RAG_RERANK_URL`, then reboot the cloud app.

## Streamlit Cloud Secrets (canonical config)

These are the only env-vars the cloud app needs. `app_llama3.2.py` bridges `st.secrets` → `os.environ` at startup, so all downstream code reads via `os.getenv()`.

```toml
LLM_BASE_URL = "https://<ngrok>.ngrok-free.app/v1"
LLM_MODEL = "scb10x/typhoon2.5-qwen3-4b-gguf:Q4_K_M"
LLM_API_KEY = "not-needed"

RAG_USE_RERANKER = "1"
RAG_RERANK_URL = "https://<ngrok>.ngrok-free.app"   # no /v1 suffix

QDRANT_URL = "..."
QDRANT_API_KEY = "..."
QDRANT_URL2 = "..."
QDRANT_API_KEY2 = "..."
TYPHOON_OCR_API_KEY = "..."          # for ingest_uploader.py OCR path only
RAG_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

Local `.env` and `.streamlit/secrets.toml` mirror these (both gitignored).

## Code structure

| File | Role |
|---|---|
| `app_llama3.2.py` | Streamlit UI; bridges `st.secrets` → `os.environ` before importing modules |
| `src/agentic_rag/crew.py` | LangGraph workflow; reads `LLM_BASE_URL/LLM_MODEL/LLM_API_KEY` |
| `src/agentic_rag/tools/custom_tool.py` | `DocumentSearchTool` (Qdrant + optional remote reranker via `RAG_RERANK_URL`) |
| `src/agentic_rag/tools/security_filter.py` | Profanity / PDPA-relevance guardrail; reads same `LLM_*` env vars |
| `src/agentic_rag/tools/qdrant_storage.py` | Qdrant client + `MyEmbedder` (SentenceTransformer) |
| `src/agentic_rag/tools/chat_history.py` | Per-session conversation memory in Qdrant |
| `rerank_server.py` | Standalone HTTP server (`stdlib http.server`) wrapping `BAAI/bge-reranker-v2-m3` |
| `ingest_uploader.py` | PDF → chunks → Qdrant with source/page metadata |
| `llm.bat` / `llmgen.bat` | Launch local llama-server (chat / generation variants) |
| `run.bat` | Activates venv + runs Streamlit |
| `ingest_with_metadata.bat` | Runs `ingest_uploader.py --path knowledge --with-metadata` |

## Known gotchas

- **ngrok-free URLs change on every restart.** Free tier has no static domain. Each restart → update Streamlit Cloud secrets.
- **`pywin32==311` is gated to Windows** in `requirements.txt` (`; sys_platform == 'win32'`). Don't undo this — it breaks Streamlit Cloud (Linux) builds.
- **Git "dubious ownership"**: project dir is owned by `BUILTIN/Administrators`, not the user. Resolved globally via `safe.directory = *`.
- **Qdrant client warns about version mismatch** (1.15.1 client vs 1.18.0 server). Harmless.
- **`pywin32` was the only Windows-only dep** that broke cloud builds. If adding new deps, watch for Windows-specific wheels.

## OpenTyphoon hosted API (fallback)

If the user can't keep their PC on, switch `LLM_BASE_URL` to `https://api.opentyphoon.ai/v1` and `LLM_MODEL` to `typhoon-v2.5-30b-a3b-instruct` (the only chat model currently exposed there). Use their personal API key in `LLM_API_KEY`. The reranker still needs ngrok unless disabled with `RAG_USE_RERANKER=0`.

## Editing conventions

- Don't commit `.env`, `.streamlit/secrets.toml`, `__pycache__/`, or anything in `venv/`. `.gitignore` covers these.
- Don't push to `main` without testing locally first — Streamlit Cloud auto-deploys on every push.
- Keep commit messages short and imperative. Author identity: `phuritatmungk <chaitat001@gmail.com>`.
- The user runs Windows 11 with Git Bash; use Unix-style paths in shell commands but forward-slash Windows paths (`C:/...`) work in the Bash tool.
