# Ragora

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/api-fastapi-009688)
![Vector DB](https://img.shields.io/badge/vector-qdrant%20%7C%20faiss-5b4bdb)
![License](https://img.shields.io/badge/license-MIT-green)

A production-style multimodal RAG system that ingests PDFs, images, and tabular data, then answers with grounded citations through a backend-served UI, CLI, and REST API. With one OpenAI API key, Ragora can run generation, embeddings, and image understanding end-to-end.

## Local Run

```powershell
cd C:\Users\ajith\Downloads\model-rag\Ragora-v2
.\scripts\run_local.ps1
```

Then open `http://127.0.0.1:8000`.

If `MMRAG_OPENAI_API_KEY` is set, Ragora automatically switches into OpenAI mode for:
- answer generation
- text embeddings
- image captioning and image-query understanding

---

## Overview

Ragora handles real-world document types end-to-end:

| Input Type | How It's Processed |
|---|---|
| **PDF** | Layout-aware text extraction + table region detection with dedup |
| **Image** | Vision-based captioning + optional OCR |
| **CSV / TSV** | Tabular normalization into structured chunks |

Retrieval combines dense vector search, BM25 lexical search, and Reciprocal Rank Fusion (RRF), with optional reranking for higher precision.

---

## Features

- Hybrid dense + lexical retrieval with RRF
- Backend-served UI at `/` and `/ui`
- Citation-rich answers with file path, modality, page, and excerpt
- Single-key OpenAI mode for end-to-end hosted inference
- Local-first mode with no external API required
- Optional OpenAI, Anthropic, Ollama, or LlamaIndex-backed generation
- Layout-aware PDF parsing and adaptive chunking
- Async ingest jobs for drag-and-drop uploads
- Multi-tenant isolation and optional API-key auth
- FAISS or Qdrant vector storage
- CLI, REST API, Docker, and evaluation harness

---

## Architecture

```
Input Files (PDF, Image, CSV/TSV)
        │
        ▼
   [Ingestion Layer]
   ├── PDF: Text + Tables
   ├── Image: Caption + OCR
   └── CSV/TSV: Tabular Normalization
        │
        ▼
[Structure-Aware Chunking + Metadata]
        │
        ├──────────────────────┐
        ▼                      ▼
 [Embedding Layer]     [Lexical Index (BM25)]
        │
        ▼
[Vector Store: FAISS / Qdrant]
        │
        ├── Dense Retrieval ───┐
        │                      ▼
        └─────────────── [RRF Fusion]
                               │
                               ▼
                  [Optional Cross-Encoder Rerank]
                               │
                               ▼
                       [LLM Synthesis]
                               │
                               ▼
                  Answer + Citations (source, page, excerpt)
```

---

## Quickstart

```bash
git clone https://github.com/ajithhraj/multimodal-rag-system.git
cd multimodal-rag-system

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -e ".[dev,vision]"
cp .env.example .env
```

### One-key OpenAI mode

Set just this in `.env`:

```env
MMRAG_OPENAI_API_KEY=your_key_here
```

Leave `MMRAG_LLM_PROVIDER=auto` and Ragora will use OpenAI for generation, text embeddings, and image understanding automatically.

### Fully local mode

Keep:

```env
MMRAG_LLM_PROVIDER=local
MMRAG_OPENAI_API_KEY=
```

### Ingest and query

```bash
mmrag ingest ./data --tenant acme
mmrag ask "What are the major metrics shown in the latest PDF tables?" --tenant acme
mmrag ask "Find charts similar to this trend" --image ./data/query_chart.png --tenant acme
mmrag serve --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`  
UI: `http://localhost:8000/`

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/` | Serve the Ragora UI |
| `GET` | `/ui` | Alternate UI route |
| `GET` | `/source-file` | Open a cited source file |
| `POST` | `/ingest-paths` | Ingest from server-side file paths |
| `POST` | `/ingest-files` | Upload and ingest files asynchronously |
| `GET` | `/ingest-jobs` | List ingest jobs |
| `GET` | `/ingest-jobs/{job_id}` | Inspect a specific ingest job |
| `DELETE` | `/ingest-jobs/{job_id}` | Remove an ingest job record |
| `POST` | `/query` | Text query |
| `POST` | `/query-stream` | Streaming text query over SSE |
| `POST` | `/query-multimodal` | Multimodal query with optional image |

### Response shape (`/query`)

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": "x1",
      "source_path": "C:/docs/report.pdf",
      "modality": "text",
      "score": 0.92
    }
  ],
  "citations": [
    {
      "chunk_id": "x1",
      "source_path": "C:/docs/report.pdf",
      "modality": "text",
      "page_number": 4,
      "excerpt": "..."
    }
  ],
  "retrieval_mode": "hybrid",
  "corrected": false,
  "grounded": true,
  "latency_ms": 123.4
}
```

---

## Configuration

All major behavior is configured through `.env.example`.

| Variable | Description |
|---|---|
| `MMRAG_LLM_PROVIDER` | `auto`, `local`, `openai`, `anthropic`, `ollama`, or `llamaindex` |
| `MMRAG_OPENAI_API_KEY` | One key for OpenAI generation, embeddings, and image understanding |
| `MMRAG_VECTOR_BACKEND` | `faiss` or `qdrant` |
| `MMRAG_STORAGE_DIR` | Local storage directory |
| `MMRAG_COLLECTION` | Vector store collection name |
| `MMRAG_DEFAULT_TENANT` | Default tenant ID |
| `MMRAG_AUTH_ENABLED` | Enable API key auth |
| `MMRAG_QDRANT_URL` | Remote Qdrant URL |
| `MMRAG_QDRANT_PATH` | Embedded/local Qdrant path |

---

## Development

```bash
ruff check src tests
pytest -q
```

---

## License

MIT
