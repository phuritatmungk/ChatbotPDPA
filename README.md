# ChatbotPDPA

Agentic RAG chatbot for Thailand's Personal Data Protection Act (PDPA / พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล).

Built with Streamlit, LangGraph, Qdrant, and the Typhoon Thai LLM (served locally via `llama.cpp`). Documents are ingested with source-citation metadata so the chatbot can cite the originating PDF and page for each answer.

## Architecture

- **UI** — `app_llama3.2.py` (Streamlit)
- **Workflow** — `src/agentic_rag/crew.py` builds a LangGraph state machine (question refine → retrieve → answer → cite)
- **Tools** — `src/agentic_rag/tools/`
  - `custom_tool.py` — `DocumentSearchTool` over Qdrant
  - `qdrant_storage.py` — Qdrant client + embedder wrapper
  - `security_filter.py` — input/output safety filter
  - `chat_history.py` — per-session conversation memory
  - `evaluate_rag.py`, `generate_pdpa_qas.py` — eval utilities
- **LLM** — Typhoon 2.5 (Qwen3 4B) served by `llama-server` on `http://localhost:8080/v1`
- **Embeddings** — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector DB** — Qdrant Cloud
- **PDF extraction** — Poppler (bundled under `poppler/`) + `pdfplumber` / `PyMuPDF` / Typhoon OCR

## Prerequisites

- Python 3.10+ (Windows)
- A Qdrant Cloud account (URL + API key)
- `llama-server.exe` from [llama.cpp](https://github.com/ggerganov/llama.cpp), placed at `C:\llama\llama-server.exe` (or edit `llm.bat` / `llmgen.bat`)
- Optional: a Typhoon OCR API key for OCR-based ingestion

## Setup

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```bat
copy .env.example .env
```

## Usage

### 1. Start the local LLM server

```bat
llm.bat
```

This launches Typhoon 2.5 (Qwen3 4B) on port 8080. For the larger generation model used during dataset creation, use `llmgen.bat` instead.

### 2. Ingest documents

Drop PDF files into `knowledge/`, then:

```bat
ingest_with_metadata.bat
```

This chunks each PDF, attaches source-citation metadata (file name + page), embeds, and uploads to Qdrant.

### 3. Run the chatbot

```bat
run.bat
```

Opens the Streamlit app in your browser. Ask questions in Thai or English; answers include citations back to the source PDFs.

## Folder layout

```
.
├── app_llama3.2.py            # Streamlit entry point
├── ingest_uploader.py         # PDF → Qdrant ingestion
├── src/agentic_rag/
│   ├── crew.py                # LangGraph workflow builder
│   ├── main.py
│   ├── config/                # agents.yaml, tasks.yaml
│   └── tools/                 # search, storage, security, eval
├── knowledge/                 # source PDFs (input to ingestion)
├── eval/                      # evaluation scripts/data
├── results150Questions/       # benchmark outputs
├── assets/                    # UI assets (logo, icons)
├── poppler/                   # bundled Poppler binaries (Windows)
├── llm.bat / llmgen.bat       # launch local LLM servers
├── run.bat                    # launch Streamlit app
└── ingest_with_metadata.bat   # ingest PDFs into Qdrant
```

## Environment variables

See `.env.example`. Required:

- `QDRANT_URL`, `QDRANT_API_KEY` — primary Qdrant collection
- `QDRANT_URL2`, `QDRANT_API_KEY2` — secondary collection (optional)
- `RAG_EMBED_MODEL` — sentence-transformers model id
- `TYPHOON_OCR_API_KEY` — for OCR-based ingestion (optional)
- `MODEL`, `OPENAI_API_KEY` — only needed if pointing at an OpenAI-compatible cloud LLM instead of local `llama-server`
