# Streaming Video RAG

Retrieval-Augmented Generation system for live streaming video content. Extract, transcribe, index and query video content with natural language.

## Overview

This project provides an end-to-end pipeline for processing video content (local files, YouTube videos, live streams) and making it searchable and queryable using LLMs.

### ✨ Features
- 🎬 Multiple video ingestion sources: Local files, YouTube, RTMP Live Streams
- 🎙️ Automatic audio transcription with OpenAI Whisper
- 📄 Smart content chunking & semantic embedding
- 🔍 Vector search with Chroma / Qdrant
- 💬 Natural language Q&A over video content
- 📊 Streamlit web interface
- 🔌 REST API for integration

## 📋 Requirements
- Python 3.10+
- FFmpeg (for audio extraction)
- LLM Provider (OpenAI API Key **OR** Local Ollama installation)

## ⚠️ Local Ollama Constraints & Recommendations

When running with local Ollama (instead of OpenAI API):

✅ **Recommended Maximum Video Length**: 30-45 minutes
❌ **Avoid videos longer than 60 minutes** with default settings

### Known Limitations:
- Streamlit UI has default 120 second timeout for API requests
- Long video transcription + embedding will exceed HTTP timeout limits
- Local LLMs have significantly lower processing throughput
- Whisper transcription on CPU is ~1x realtime speed (1hr video = 1hr processing)

### Workarounds for Long Videos:
1. **Use the API directly** instead of the Web UI for ingestion
2. **Increase timeout values** in `ui/app.py` for local usage
3. **Split long videos** into smaller segments before ingestion
4. **Run ingestion as background process**
5. Use larger Ollama models with enough VRAM (7B+ models recommended)

For production usage with long-form content, OpenAI API is recommended for predictable performance.

## 🚀 Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd Streaming Video-RAG

# 2. Copy environment configuration
cp .env.example .env
# Edit .env file with your API keys and configuration

# 3. Install dependencies
pip install -r requirements.txt
```

## ▶️ Running the Application

### Start Backend API
```bash
uvicorn api.main:app --reload
```
API will be available at `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

### Start Web UI
```bash
streamlit run ui/app.py
```
UI will be available at `http://localhost:8501`

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Video Input    │────▶│  Transcription  │────▶│  Vector Store   │
│  (Ingestion)    │     │   (Whisper)     │     │   (Chroma)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  User Query     │────▶│  Retriever      │────▶│  LLM Response   │
│  (UI / API)     │     │                 │     │   Generation    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Project Structure
```
Streaming Video-RAG/
├── api/                  # REST API endpoints
├── ingestion/            # Video ingestion handlers
├── transcription/        # Audio transcription services
├── processing/           # Chunking & embedding
├── vector_store/         # Vector database implementations
├── rag/                  # RAG pipeline logic
├── storage/              # Database storage
├── ui/                   # Streamlit web interface
└── data/                 # Local storage for media
```

## 🔧 Usage

### 1. Ingest Video Content
```bash
# Add via API
curl -X POST http://localhost:8000/api/ingest/youtube \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=example"}'
```

### 2. Search Content
```bash
curl "http://localhost:8000/api/search?q=your+query"
```

### 3. Ask Questions
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was discussed about AI?"}'
```

## ⚙️ Configuration

Edit `.env` file for configuration:
- `OPENAI_API_KEY` - Your OpenAI API key
- `VECTOR_STORE` - Choose `chroma` or `qdrant`
- `WHISPER_MODEL` - Whisper model size (base/small/medium/large)
- `CHUNK_SIZE` - Text chunk size for embedding
- `UI_REQUEST_TIMEOUT` - HTTP request timeout for UI (increase for local Ollama, default: 120)

## 📝 API Endpoints

| Method | Endpoint               | Description
|--------|------------------------|-------------
| GET    | `/api/videos`          | List all indexed videos
| POST   | `/api/ingest/youtube`  | Ingest YouTube video
| POST   | `/api/ingest/file`     | Upload local video file
| POST   | `/api/ingest/stream`   | Connect to live stream
| GET    | `/api/search`          | Semantic search
| POST   | `/api/query`           | Natural language Q&A

## 🛠️ Development

```bash
# Run all components via Docker
docker-compose up -d

# View logs
docker-compose logs -f
```

## License
MIT License