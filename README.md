# Multimodal RAG System

Production-ready multimodal Retrieval-Augmented Generation (RAG) for documents, tables, and images.

The system ingests:
- PDFs (text + tables)
- images (captions + optional OCR)
- CSV/TSV table files

and serves:
- a CLI for local workflows
- a FastAPI REST API for integration
- Dockerized runtime for deployment

## Why This Project

Most RAG demos handle plain text only. This project follows production patterns for multimodal retrieval:
- modality-aware indexing
- hybrid retrieval (dense + lexical BM25)
- reciprocal rank fusion (RRF)
- optional cross-encoder reranking
- citation-rich answers with source grounding

## Architecture

1. Ingestion
- Parse PDF text
- Extract PDF tables
- Caption images with a vision model (optional)
- Attach OCR text when available

2. Embedding
- Text/table embedding
- Image semantic embedding (caption + optional CLIP)

3. Storage
- FAISS backend with NumPy fallback
- Qdrant backend (embedded local path or remote endpoint)

4. Retrieval and Generation
- Dense retrieval per modality (text/table/image)
- Sparse lexical retrieval (BM25 with fallback scorer)
- Reciprocal Rank Fusion (RRF)
- Optional cross-encoder reranking
- Grounded answer synthesis via LangChain orchestration

## Quickstart

### Local setup

```bash
cd multimodal-rag-system
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,vision]"
copy .env.example .env
```

### Ingest data

```bash
mmrag ingest ./data
```

### Ask questions

```bash
mmrag ask "What are the major metrics shown in the latest PDF tables?"
```

### Run API

```bash
mmrag serve --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

## Docker

```bash
docker compose up --build
```

## API Endpoints

- `GET /health`
- `POST /ingest-paths`
- `POST /ingest-files`
- `POST /query`

`/query` returns:
- `answer`
- `sources` (retrieved chunks + scores)
- `citations` (source path, modality, page number, excerpt)

## Configuration

Important environment variables:
- `MMRAG_VECTOR_BACKEND` = `faiss` or `qdrant`
- `MMRAG_STORAGE_DIR` = local state dir (default `.rag_store`)
- `MMRAG_COLLECTION` = logical index namespace
- `MMRAG_OPENAI_API_KEY` = enables OpenAI generation and vision captioning
- `MMRAG_RETRIEVAL_TOP_K_PER_MODALITY` = dense candidates per modality
- `MMRAG_RETRIEVAL_TOP_K_LEXICAL` = lexical BM25 candidates
- `MMRAG_RETRIEVAL_RRF_K` = RRF fusion constant
- `MMRAG_RETRIEVAL_ENABLE_RERANKER` = enable/disable cross-encoder rerank
- `MMRAG_RETRIEVAL_RERANKER_MODEL` = cross-encoder model name
- `MMRAG_RETRIEVAL_RERANK_CANDIDATES` = fused candidates before rerank

For Qdrant:
- `MMRAG_QDRANT_URL`
- `MMRAG_QDRANT_API_KEY`
- `MMRAG_QDRANT_PATH` (embedded mode)

## CLI

- `mmrag ingest <path>`
- `mmrag ask <question>`
- `mmrag serve`

Run `mmrag --help` for full command options.

## Development

```bash
ruff check src tests
pytest -q
```

## Project Structure

```text
src/multimodal_rag/
  ingestion/      # loaders, chunking, pdf/image/table extraction
  embedding/      # text + vision embedders
  storage/        # faiss and qdrant backends
  retrieval/      # lexical index, RRF fusion, reranker
  generation/     # answer synthesis
  api/            # FastAPI app and schemas
```
