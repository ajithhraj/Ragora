# Multimodal RAG System

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/api-fastapi-009688)](https://fastapi.tiangolo.com)
[![Vector DB](https://img.shields.io/badge/vector-qdrant%20%7C%20faiss-5b4bdb)](https://qdrant.tech)
[![LLM](https://img.shields.io/badge/llm-openai%20%7C%20anthropic%20%7C%20ollama-orange)](https://github.com/ajithhraj/multimodal-rag-system)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Production-style multimodal Retrieval-Augmented Generation (RAG) system for PDFs, images, and tabular data, with optional session memory for longer-running workflows.

![Architecture](docs/assets/architecture.svg)

## Overview

This project ingests and queries across:

- PDFs with layout-aware text and table extraction
- Images with CLIP embeddings, optional OCR, and caption context
- CSV / TSV files normalized into retrieval-friendly chunks
- Session memory for user preferences, project facts, and recurring instructions

It ships with:

- A reusable Python package (`src` layout)
- A CLI (`mmrag`) for ingestion, querying, evaluation, and serving
- A FastAPI service with async ingest jobs and SSE query streaming
- Multiple vector backends (`faiss`, `qdrant`)
- Multiple LLM backends (`openai`, `anthropic`, `ollama`, `llamaindex`)
- Strict OpenAI-only mode when you want the full pipeline to avoid local fallbacks
- Session memory with decay, reinforcement, and prompt injection inspired by MemOS

## Key Capabilities

- Hybrid retrieval (dense + BM25) with Reciprocal Rank Fusion (RRF)
- Optional cross-encoder reranking for precision
- Adaptive query expansion and corrective retrieval fallback
- Citation-rich responses with modality, source, page, and excerpt
- Multi-tenant data isolation with optional tenant API-key auth
- Per-tenant API rate limiting
- Real token streaming through `/query-stream` (SSE)
- Session-scoped memory with decay, pinning, reinforcement, and explicit forgetting
- Strict OpenAI mode that disables silent local fallback when you want a single-provider path

## What It Does

The system combines two layers:

- `RAG layer`
  Retrieves grounded evidence from indexed files and uses that evidence to answer questions.
- `Memory layer`
  Stores durable session context such as preferences, recurring goals, and project facts, then injects relevant memories back into later prompts.

That means you can ask questions like:

- “Summarize the key risks in this contract”
- “Find similar chart patterns from the uploaded slides”
- “Explain this architecture, but keep it concise for a product demo”

The answer still comes from retrieved sources, but memory can shape how the answer is framed when that context is relevant.

## Quickstart

```bash
git clone https://github.com/ajithhraj/multimodal-rag-system
cd multimodal-rag-system
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,vision,faiss]"
copy .env.example .env
```

```bash
mmrag ingest ./data --tenant acme
mmrag remember "remember I want concise demo-style answers" --tenant acme --session raj-demo --pin
mmrag ask "Summarize key contract risks" --tenant acme --session raj-demo
mmrag ask "Find similar chart patterns" --image ./data/query_chart.png --tenant acme
mmrag serve --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

## API Surface

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health |
| `POST` | `/ingest-paths` | Synchronous ingest from server-side paths |
| `POST` | `/ingest-files` | Async upload ingest; returns job id |
| `GET` | `/ingest-jobs` | List recent async ingest jobs |
| `GET` | `/ingest-jobs/{job_id}` | Check async ingest status |
| `DELETE` | `/ingest-jobs/{job_id}` | Delete job metadata |
| `POST` | `/reset-collection` | Clear a tenant-scoped indexed collection |
| `POST` | `/query` | JSON query response with citations |
| `POST` | `/query-stream` | SSE streaming query tokens |
| `POST` | `/query-multimodal` | Multipart query with optional image |
| `POST` | `/memory/ingest` | Store session memories from a user message |
| `POST` | `/memory/query` | Retrieve ranked session memories |
| `GET` | `/memory/export` | Export full session memory state |
| `GET` | `/memory/stats` | Memory count, pin count, average importance |
| `POST` | `/memory/{id}/reinforce` | Boost and pin one memory |
| `DELETE` | `/memory/{id}` | Forget one memory |

### Async Ingest Example

```bash
curl -X POST http://localhost:8000/ingest-files \
  -H "X-Tenant-ID: acme" \
  -F "files=@report.pdf" \
  -F "files=@data.csv"
```

### Streaming Query Example

```bash
curl -X POST http://localhost:8000/query-stream \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme" \
  -d "{\"question\":\"What encryption standard is required?\",\"session_id\":\"raj-demo\"}" \
  --no-buffer
```

### Multimodal Query Example

```bash
curl -X POST http://localhost:8000/query-multimodal \
  -H "X-Tenant-ID: acme" \
  -F "question=Find similar chart patterns" \
  -F "session_id=raj-demo" \
  -F "image=@./data/query_chart.png"
```

### Memory Example

```bash
curl -X POST http://localhost:8000/memory/ingest \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme" \
  -d "{\"message\":\"remember I prefer concise answers for product demos\",\"session_id\":\"raj-demo\",\"pinned\":true}"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme" \
  -d "{\"question\":\"summarize the risks\",\"session_id\":\"raj-demo\"}"
```

### Reset a Stale Collection

```bash
curl -X POST http://localhost:8000/reset-collection \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme" \
  -d "{\"collection\":\"default\"}"
```

## LLM Providers

Switch generation provider without changing code:

```bash
# OpenAI (default)
MMRAG_LLM_PROVIDER=openai
MMRAG_OPENAI_API_KEY=sk-...
MMRAG_API_ONLY_MODE=true

# Anthropic
MMRAG_LLM_PROVIDER=anthropic
MMRAG_ANTHROPIC_API_KEY=sk-ant-...
MMRAG_ANTHROPIC_MODEL=claude-sonnet-4-5

# Ollama (local)
MMRAG_LLM_PROVIDER=ollama
MMRAG_OLLAMA_BASE_URL=http://localhost:11434
MMRAG_OLLAMA_MODEL=llama3
```

Install optional extras as needed:

```bash
pip install -e ".[anthropic]"
pip install -e ".[ollama]"
```

## Multi-Tenancy and Auth

- Tenant namespace isolation via `tenant-<id>__<collection>`
- Header-based tenant routing (`X-Tenant-ID` by default)
- Optional API-key auth:
  - `MMRAG_AUTH_ENABLED=true`
  - `MMRAG_AUTH_TENANT_API_KEYS=tenant_a:key_a,tenant_b:key_b`

## Session Memory

Session memory is intentionally separate from document retrieval:

- documents remain the ground truth for factual answers
- memory stores conversational context that may matter later
- memories are scoped by `tenant + session_id`
- pinned memories decay more slowly
- weak memories fade over time and can be pruned automatically

Good memory examples:

- “I prefer concise answers”
- “This repo is for an interview demo”
- “Focus on architecture and risks first”
- “Call this project Raj’s RAG”

Poor memory examples:

- one-off document questions
- temporary OCR noise
- long raw chat dumps that should stay in retrieval history instead

## Rate Limiting

Per-tenant request throttling:

```bash
MMRAG_RATE_LIMIT_ENABLED=true
MMRAG_RATE_LIMIT_RPM=60
```

When exceeded, API returns `HTTP 429` with `Retry-After`.

## Evaluation

```bash
mmrag eval ./eval/datasets/starter_eval.jsonl --ingest-path ./data --tenant acme --k-values 1,3,5,10
mmrag eval ./eval/datasets/starter_eval.jsonl --ingest-path ./data --tenant acme --ablation
```

Reports include: `Recall@k`, `MRR`, citation hit-rate, citation precision, average and p95 latency.

## Configuration

Important variables:

- `MMRAG_VECTOR_BACKEND`
- `MMRAG_STORAGE_DIR`
- `MMRAG_COLLECTION`
- `MMRAG_DEFAULT_TENANT`
- `MMRAG_LLM_PROVIDER`
- `MMRAG_API_ONLY_MODE`
- `MMRAG_OPENAI_API_KEY`
- `MMRAG_CHAT_MODEL`
- `MMRAG_ANTHROPIC_API_KEY`
- `MMRAG_ANTHROPIC_MODEL`
- `MMRAG_OLLAMA_BASE_URL`
- `MMRAG_OLLAMA_MODEL`
- `MMRAG_RATE_LIMIT_ENABLED`
- `MMRAG_RATE_LIMIT_RPM`
- `MMRAG_AUTH_ENABLED`
- `MMRAG_AUTH_TENANT_API_KEYS`
- `MMRAG_QDRANT_URL`
- `MMRAG_QDRANT_API_KEY`
- `MMRAG_MEMORY_ENABLED`
- `MMRAG_MEMORY_TOP_K`
- `MMRAG_MEMORY_DECAY_RATE`
- `MMRAG_MEMORY_PRUNE_THRESHOLD`
- `MMRAG_MEMORY_AUTO_STORE_QUERIES`

See `.env.example` for full defaults.

## CLI

```bash
mmrag ingest <path> [--tenant <id>] [--collection <name>]
mmrag ask <question> [--tenant <id>] [--session <id>] [--image <path>] [--retrieval-mode <mode>]
mmrag remember <message> --session <id> [--tenant <id>] [--pin]
mmrag memory <query> --session <id> [--tenant <id>]
mmrag serve [--host 0.0.0.0] [--port 8000]
mmrag eval <dataset-path> [--tenant <id>] [--ablation]
```

## Project Structure

```text
src/multimodal_rag/
  memory/         # session memory extraction, decay, ranking, persistence
  ingestion/      # pdf/image/table parsing and chunking
  embedding/      # text and vision embedders
  storage/        # faiss and qdrant backends
  retrieval/      # bm25, fusion, reranker, query expansion
  generation/     # multi-provider synthesis and streaming
  api/            # FastAPI app and schemas
  eval/           # retrieval benchmark harness
```

## What Else Can We Implement

High-impact next upgrades:

1. Persistent async job queue with Redis + worker process (replace in-memory job store)
2. OpenTelemetry traces/metrics + Prometheus/Grafana dashboards
3. Conversation memory and session-based chat endpoint (`/chat`)
4. Metadata filtering (`doc_type`, `date_range`, `owner`) in retrieval queries
5. Reranker and embedder model registry with runtime switching
6. Web UI for ingestion jobs, streaming answers, and citation inspection
7. CI benchmark gate to block regressions in Recall@k and latency
8. Production deployment templates (Kubernetes, Helm, managed Qdrant)

## Development

```bash
ruff check src tests
pytest -q
```
