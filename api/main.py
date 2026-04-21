"""
Streaming Video-RAG — FastAPI application entry point.

Run:
    uvicorn api.main:app --reload
    or
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import settings
from storage.database import init_db
from api.routes import ingest, videos, query, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting Streaming Video-RAG API...")
    settings.ensure_dirs()
    init_db()
    logger.success(
        f"API ready — LLM={settings.llm_provider.value}/{settings.llm_model}, "
        f"Whisper={settings.whisper_mode.value}/{settings.whisper_model_size}, "
        f"Embeddings={settings.embedding_mode.value}, "
        f"VectorStore={settings.vector_store_type.value}"
    )
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Streaming Video-RAG",
    description=(
        "A RAG system for video content. Ingest YouTube URLs, local files, "
        "live streams, or video API sources — then ask questions, search, "
        "and summarize across your video library."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest.router)
app.include_router(videos.router)
app.include_router(query.router)
app.include_router(search.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider.value,
        "llm_model": settings.llm_model if settings.llm_provider.value != "ollama" else settings.ollama_model,
        "whisper_mode": settings.whisper_mode.value,
        "whisper_model": settings.whisper_model_size,
        "embedding_mode": settings.embedding_mode.value,
        "vector_store": settings.vector_store_type.value,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
