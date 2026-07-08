"""
Streaming Video-RAG — FastAPI application entry point.

Run:
    uvicorn api.main:app --reload
    or
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import ingest, query, search, videos
from config import settings
from storage.database import init_db


def _recover_stale_jobs() -> None:
    """
    Mark jobs that were in progress when the server last stopped as errored.
    BackgroundTasks do not survive a restart, so these jobs can never complete.
    """
    from storage.database import IngestJob, SessionLocal

    stale_statuses = ("queued", "ingesting", "transcribing", "indexing")
    db = SessionLocal()
    try:
        stale_jobs = (
            db.query(IngestJob).filter(IngestJob.status.in_(stale_statuses)).all()
        )
        for job in stale_jobs:
            job.status = "error"
            job.error_message = "Job interrupted by server restart — please resubmit"
        if stale_jobs:
            db.commit()
            logger.warning(f"Marked {len(stale_jobs)} stale ingestion job(s) as error")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to recover stale jobs: {e}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting Streaming Video-RAG API...")
    settings.ensure_dirs()
    init_db()
    _recover_stale_jobs()
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
    allow_origins=["*"],  # tighten this in production
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
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider.value,
        "llm_model": (
            settings.llm_model
            if settings.llm_provider.value != "ollama"
            else settings.ollama_model
        ),
        "whisper_mode": settings.whisper_mode.value,
        "whisper_model": settings.whisper_model_size,
        "embedding_mode": settings.embedding_mode.value,
        "vector_store": settings.vector_store_type.value,
    }


if __name__ == "__main__":
    from pathlib import Path

    import uvicorn

    # Only enable SSL if both cert files exist
    ssl_kwargs = {}
    if settings.api_ssl_certfile and settings.api_ssl_keyfile:
        cert_path = Path(settings.api_ssl_certfile)
        key_path = Path(settings.api_ssl_keyfile)
        if cert_path.exists() and key_path.exists():
            ssl_kwargs["ssl_certfile"] = settings.api_ssl_certfile
            ssl_kwargs["ssl_keyfile"] = settings.api_ssl_keyfile
            logger.info(f"SSL enabled — cert={cert_path}, key={key_path}")
        else:
            logger.warning(
                f"SSL cert/key files not found at {cert_path}, {key_path}. "
                "Running without SSL. Generate certs with: "
                "openssl req -x509 -newkey rsa:4096 -days 365 -nodes "
                "-keyout certs/localhost-key.pem -out certs/localhost.pem "
                "-subj '/CN=localhost' -addext 'subjectAltName=DNS:localhost'"
            )

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        http="h11",
        timeout_keep_alive=300,  # 5 minute timeout for long running tasks
        timeout_graceful_shutdown=300,
        **ssl_kwargs,
    )
