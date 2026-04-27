# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

**Backend API:**

```bash
uvicorn api.main:app --reload
# Docs at http://localhost:8000/docs
```

**Web UI:**

```bash
streamlit run ui/app.py
# Available at http://localhost:8501
```

**Docker Compose profiles:**

```bash
docker-compose up                    # default: SQLite + Chroma
docker-compose --profile qdrant up   # add Qdrant vector store
docker-compose --profile ollama up   # add Ollama for local LLMs
docker-compose --profile full up     # all services
```

## Linting and Type Checking

```bash
uvx ruff check .                   # lint (E, F, I, B, C4, SIM rules)
uvx black --check .                # format check (line length 120)
uvx mypy .                         # strict type checking
```

Fix before committing — pre-commit hooks enforce all three. No tests exist yet.

## Architecture

This is a **video RAG pipeline**: ingest video from multiple sources → transcribe with Whisper → chunk → embed → store in vector DB → serve semantic search and LLM Q&A via REST API and Streamlit UI.

**Data flow:**

```
Video source → Ingester → 16kHz WAV → WhisperTranscriber → Transcript (timestamped segments)
  → Chunker (60s windows, 15s overlap, with timestamp metadata)
  → Embedder → ChromaDB/Qdrant
  → Retriever (embed query → vector search) → QAChain/Summarizer (LLM with citations)
```

**Key modules:**

- `ingestion/` — `BaseIngester` → `VideoAsset`. Implementations: YouTube (yt-dlp), LocalFile, LiveStream (HLS/RTMP), VideoAPI (Vimeo/Twitch)
- `transcription/whisper_transcriber.py` — LOCAL or OPENAI_API mode; outputs `Transcript` with `TranscriptSegment` (text + timestamps)
- `processing/chunker.py` — produces `VideoChunk` with deep-link URLs and timestamp metadata
- `processing/embedder.py` — LOCAL (sentence-transformers) or OPENAI mode
- `vector_store/` — `BaseVectorStore` abstract class with `ChromaStore` and `QdrantStore` implementations
- `llm/factory.py` — builds OpenAI, Anthropic, or Ollama LLM instance from config
- `rag/retriever.py` — singleton retriever (cached); `rag/qa_chain.py` — Q&A with timestamped citations; `rag/summarizer.py` — map-reduce summarization
- `storage/database.py` — SQLAlchemy `Video` and `IngestJob` models; SQLite (dev) or PostgreSQL (prod)
- `api/routes/` — ingest (async background jobs), videos, query, search
- `config.py` — all settings via Pydantic `BaseSettings`, populated from `.env`

## Configuration

All settings come from `.env` (see `.env.example`). Key variables:

| Variable                 | Options                       | Default                  |
| ------------------------ | ----------------------------- | ------------------------ |
| `LLM_PROVIDER`           | openai / anthropic / ollama   | openai                   |
| `WHISPER_MODE`           | local / openai_api            | local                    |
| `WHISPER_MODEL_SIZE`     | base / small / medium / large | base                     |
| `EMBEDDING_MODE`         | local / openai                | local                    |
| `VECTOR_STORE_TYPE`      | chroma / qdrant               | chroma                   |
| `CHUNK_DURATION_SECONDS` | int                           | 60                       |
| `CHUNK_OVERLAP_SECONDS`  | int                           | 15                       |
| `RETRIEVAL_TOP_K`        | int                           | 5                        |
| `UI_REQUEST_TIMEOUT`     | int                           | 120 (use 600 for Ollama) |

## Code Standards

- All functions must be fully typed; no `Any` without justification
- Functions ≤ 50 lines, max 3 levels of nesting
- Use `logging`, not `print`
- Explicit error handling — no bare `except`
- Black line length: 120
