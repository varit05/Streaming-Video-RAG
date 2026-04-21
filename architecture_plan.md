# Streaming Video-RAG System — Architecture Plan

## What This System Does

A full-featured RAG (Retrieval-Augmented Generation) pipeline built around video content. You can point it at a YouTube URL, a local file, a live stream, or a video API — and it will transcribe, chunk, embed, and index the content. From there, users can ask questions, search across a video library, or get summaries, all grounded in timestamped video segments.

---

## System Layers

### 1. Ingestion Layer

This is where video enters the system. Each source type has its own adapter.

| Adapter | Source | Tool |
|---|---|---|
| `YouTubeIngester` | YouTube, public URLs | `yt-dlp` |
| `LocalFileIngester` | .mp4, .mov, .avi, .mkv | `ffmpeg` |
| `LiveStreamIngester` | HLS, RTMP, live feeds | `ffmpeg` (segment capture) |
| `VideoAPIIngester` | Vimeo, Twitch, custom APIs | `requests` + API SDKs |

All adapters implement a shared `BaseIngester` interface:
- `ingest(source) -> VideoAsset`
- Returns a `VideoAsset` with: `video_id`, `title`, `duration`, `source_url`, `local_audio_path`, `metadata`

Live stream ingestion works by capturing fixed-length segments (e.g., 60s chunks) continuously, processing each one through the pipeline in near-real-time.

---

### 2. Audio Extraction

Once ingested, audio is stripped from video using **ffmpeg**:
- Output: 16kHz mono WAV (optimal for Whisper)
- Stored temporarily in a local `./data/audio/` folder

---

### 3. Transcription

**Whisper** (OpenAI, run locally via `openai-whisper` or via the API) transcribes audio with word- and segment-level timestamps.

Output per video:
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 12.4,
      "text": "Today we're going to talk about...",
      "words": [...]
    }
  ]
}
```

Options:
- **Local (free)**: `openai-whisper` — runs `tiny`, `base`, `medium`, or `large` models
- **API (faster)**: `openai.audio.transcriptions.create()` — costs money but faster

---

### 4. Chunking

Raw Whisper segments are merged into larger, overlapping chunks for embedding:

- **Chunk size**: ~60 seconds of transcript text (or ~300-400 tokens)
- **Overlap**: 15 seconds between adjacent chunks (to avoid losing context at boundaries)
- **Metadata per chunk**:
  - `video_id`, `title`, `source_url`
  - `start_time`, `end_time` (deep-link timestamps)
  - `chapter` (if available from YouTube chapters)
  - `segment_index`

This gives retrieval a "time-aware" quality — every result comes back with a timestamp you can link directly to in the video.

---

### 5. Embedding & Vector Store

Each chunk is embedded and stored in a vector database.

**Embedding models (pick one):**
- `sentence-transformers/all-MiniLM-L6-v2` — fast, local, free
- `text-embedding-3-small` (OpenAI) — higher quality, costs money

**Vector stores (pick one based on scale):**

| Store | Best for | Notes |
|---|---|---|
| **Chroma** | Local dev / small scale | Simple setup, no server needed |
| **Qdrant** | Production / larger scale | Docker-based, fast filtering |
| **FAISS** | Offline / batch use | No persistence out of the box |

Metadata filtering is supported in both Chroma and Qdrant — so you can scope queries to a specific video, time range, or source.

---

### 6. RAG Pipeline

Three main capabilities, each with its own chain:

**Q&A**
- User asks: *"What did the speaker say about neural networks?"*
- Retriever fetches top-k chunks by semantic similarity
- LLM (GPT-4o, Claude, or local Ollama model) answers with citations (timestamps)

**Search**
- Semantic search across all indexed videos
- Returns ranked chunks with `video_id`, `timestamp`, and a snippet
- Supports metadata filters: by video, by date, by source type

**Summarization**
- Map-reduce: each chunk is summarized independently, then combined
- Works well on long videos (1hr+)
- Output: overall summary + per-chapter breakdown

Framework: **LangChain** (simpler setup) or **LlamaIndex** (more built-in video/document tooling). LangChain recommended for this use case.

---

### 7. API Layer (FastAPI)

RESTful API that wraps the entire pipeline.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Submit a video source for ingestion |
| `GET` | `/ingest/{job_id}` | Poll ingestion job status |
| `GET` | `/videos` | List all indexed videos |
| `DELETE` | `/videos/{video_id}` | Remove a video and its embeddings |
| `POST` | `/query` | Ask a question across all (or specific) videos |
| `POST` | `/search` | Semantic search with optional filters |
| `POST` | `/summarize` | Summarize a video by ID |

Ingestion runs as a **background task** (FastAPI `BackgroundTasks` for small scale, or Celery + Redis for production).

---

### 8. Storage

| What | Where |
|---|---|
| Video metadata & job status | SQLite (dev) / PostgreSQL (prod) |
| Audio/video files | Local `./data/` folder (or S3 in prod) |
| Transcripts | JSON files + database |
| Embeddings | Chroma or Qdrant |

---

## Folder Structure

```
streaming-video-rag/
│
├── ingestion/
│   ├── base.py              # Abstract BaseIngester
│   ├── youtube.py           # yt-dlp ingester
│   ├── local_file.py        # Local video file ingester
│   ├── live_stream.py       # HLS/RTMP stream ingester
│   └── video_api.py         # Vimeo, Twitch, etc.
│
├── transcription/
│   ├── whisper_transcriber.py
│   └── utils.py             # Audio extraction helpers
│
├── processing/
│   ├── chunker.py           # Timestamp-aware chunking
│   └── embedder.py          # Embedding generation
│
├── vector_store/
│   ├── base.py              # Abstract VectorStore
│   ├── chroma_store.py
│   └── qdrant_store.py
│
├── rag/
│   ├── retriever.py         # Semantic retrieval
│   ├── qa_chain.py          # Q&A with citations
│   ├── summarizer.py        # Map-reduce summarization
│   └── search.py            # Search interface
│
├── api/
│   ├── main.py              # FastAPI app entry point
│   ├── models.py            # Pydantic request/response models
│   └── routes/
│       ├── ingest.py
│       ├── query.py
│       ├── search.py
│       └── videos.py
│
├── storage/
│   └── database.py          # SQLAlchemy models + session
│
├── data/                    # Runtime data (gitignored)
│   ├── audio/
│   └── transcripts/
│
├── config.py                # Env vars, model choices, paths
├── requirements.txt
├── docker-compose.yml       # Chroma/Qdrant + Redis (optional)
└── .env.example
```

---

## Data Flow

```
Video Source
    │
    ▼
Ingestion Adapter (yt-dlp / ffmpeg / API)
    │
    ▼
Audio Extraction (ffmpeg → 16kHz WAV)
    │
    ▼
Transcription (Whisper → timestamped segments)
    │
    ▼
Chunking (60s windows with 15s overlap + metadata)
    │
    ▼
Embedding (sentence-transformers or OpenAI)
    │
    ▼
Vector Store (Chroma / Qdrant)
    │
    ▼
RAG Pipeline ──────────────────────────────────┐
    ├── Q&A (retrieve → LLM → answer + timestamps)
    ├── Search (retrieve → ranked results)
    └── Summarize (map-reduce → summary)
```

---

## Key Dependencies

```
# Ingestion
yt-dlp
ffmpeg-python
requests

# Transcription
openai-whisper      # or openai (for API-based Whisper)

# Embeddings & RAG
langchain
langchain-openai
langchain-community
sentence-transformers
chromadb            # or qdrant-client

# API
fastapi
uvicorn
pydantic

# Storage
sqlalchemy
alembic

# Optional
celery              # async job queue
redis               # celery broker
scenedetect         # video scene detection
```

---

## Recommended Build Order

1. **Config + project skeleton** — env vars, folder structure, base classes
2. **Ingestion** — start with YouTube + local file (most common cases)
3. **Transcription** — Whisper local, test on a short video
4. **Chunker + embedder** — get chunks into Chroma
5. **Basic Q&A** — LangChain retrieval chain over the vector store
6. **FastAPI API** — wrap everything in endpoints
7. **Search + summarization** — add remaining RAG capabilities
8. **Live stream ingestion** — more complex, add last
9. **Video API connectors** — add as needed per platform
10. **Production hardening** — swap SQLite → Postgres, Chroma → Qdrant, add Celery

---

## Open Questions to Decide Before Coding

- **LLM**: OpenAI GPT-4o, Claude via API, or a local model (Ollama with Llama 3)?
- **Whisper mode**: Local (free, slower) or OpenAI API (faster, costs ~$0.006/min)?
- **Embedding model**: Local sentence-transformers (free) or OpenAI (better quality)?
- **Vector store**: Chroma (simplest) or Qdrant (more scalable)?
- **Do you need a UI?** (Streamlit is quick; React is more polished)
