"""
/ingest routes — submit and monitor video ingestion jobs.
Ingestion runs as a FastAPI background task to avoid blocking the request.
"""

import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.models import IngestRequest, IngestResponse, JobStatusResponse
from config import settings
from storage.database import IngestJob, Video, get_db

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("", response_model=IngestResponse)
def submit_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Submit a video source for ingestion.
    Returns a job_id immediately; poll GET /ingest/{job_id} for progress.
    """
    job = IngestJob(
        source=request.source,
        source_type=request.source_type or _detect_source_type(request.source),
        status="queued",
        progress_message="Job queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Run ingestion in the background
    background_tasks.add_task(
        _run_ingest_pipeline,
        job_id=job.id,
        source=request.source,
        source_type=job.source_type,
        language=request.language,
        platform=request.platform,
        api_credentials=request.api_credentials,
    )

    logger.info(f"[Ingest] Job {job.id} queued for source: {request.source[:60]}")

    return IngestResponse(
        job_id=job.id,
        status="queued",
        message="Ingestion job submitted. Use GET /ingest/{job_id} to track progress.",
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Poll the status of an ingestion job."""
    job = db.query(IngestJob).filter(IngestJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_data = job.to_dict()
    # Rename 'id' field to 'job_id' to match Pydantic model expectation
    job_data["job_id"] = job_data.pop("id")
    return JobStatusResponse(**job_data)


# ── Background pipeline ──────────────────────────────────────────────────────

# Module-level singletons avoid reloading heavy ML models on every job.
# In a multi-worker deployment each worker holds its own instance, but
# within a worker the model is reused across all background tasks.
_transcriber_singleton = None
_transcriber_lock = threading.Lock()

_embedder_singleton = None
_embedder_lock = threading.Lock()

_vector_store_singleton = None
_vector_store_lock = threading.Lock()


def _get_transcriber():
    global _transcriber_singleton
    if _transcriber_singleton is None:
        with _transcriber_lock:
            if _transcriber_singleton is None:
                from transcription import WhisperTranscriber

                _transcriber_singleton = WhisperTranscriber()
    return _transcriber_singleton


def _get_embedder():
    global _embedder_singleton
    if _embedder_singleton is None:
        with _embedder_lock:
            if _embedder_singleton is None:
                from processing import Embedder

                _embedder_singleton = Embedder()
    return _embedder_singleton


def _get_vector_store():
    global _vector_store_singleton
    if _vector_store_singleton is None:
        with _vector_store_lock:
            if _vector_store_singleton is None:
                from vector_store import get_vector_store as _factory

                _vector_store_singleton = _factory()
    return _vector_store_singleton


def _run_ingest_pipeline(
    job_id: str,
    source: str,
    source_type: str,
    language: str = None,
    platform: str = None,
    api_credentials: dict[str, str] | None = None,
) -> None:
    """
    Full ingestion pipeline run as a background task:
    1. Ingest (download / capture audio)
    2. Transcribe (Whisper)
    3. Chunk
    4. Embed
    5. Store in vector DB
    6. Save transcript + update metadata
    """
    from storage.database import SessionLocal

    db = SessionLocal()

    def update_job(status: str, message: str, video_id: str = None, error: str = None):
        try:
            job = db.query(IngestJob).filter(IngestJob.id == job_id).first()
            if job:
                job.status = status
                job.progress_message = message
                if video_id:
                    job.video_id = video_id
                if error:
                    job.error_message = error
                if status == "done":
                    job.completed_at = datetime.utcnow()
                db.commit()
        except Exception:
            db.rollback()
            logger.warning(f"[Ingest] Job {job_id} — failed to update status to {status!r}")

    try:
        settings.ensure_dirs()

        # ── Step 1: Ingest ────────────────────────────────────────────────────
        update_job("ingesting", "Downloading / capturing audio...")
        ingester = _get_ingester(source_type, platform, api_credentials)
        asset = ingester.ingest(source)

        update_job("transcribing", "Transcribing audio with Whisper...", video_id=asset.video_id)

        # ── Step 2: Save video record ─────────────────────────────────────────
        # Use merge so a concurrent job for the same video_id doesn't raise IntegrityError.
        video = Video(
            id=asset.video_id,
            title=asset.title,
            source_url=asset.source_url,
            source_type=source_type,
            duration_seconds=asset.duration_seconds,
            description=asset.description,
            uploader=asset.uploader,
            upload_date=asset.upload_date,
            status="processing",
        )
        try:
            db.add(video)
            db.commit()
        except IntegrityError:
            db.rollback()
            video = db.query(Video).filter(Video.id == asset.video_id).first()

        # ── Step 3: Transcribe ────────────────────────────────────────────────
        transcriber = _get_transcriber()
        transcript = transcriber.transcribe(asset.local_audio_path, asset.video_id, language)

        # Save transcript to disk
        transcript_path = Path(settings.transcript_dir) / f"{asset.video_id}.json"
        transcript.save(transcript_path)

        update_job("indexing", "Chunking and indexing content...")

        # ── Step 4: Chunk ─────────────────────────────────────────────────────
        from processing import Chunker

        chunker = Chunker()
        chunks = chunker.chunk(transcript, asset)

        if not chunks:
            raise ValueError("No chunks produced from transcript — audio may be silent or too short")

        # ── Step 5: Embed ─────────────────────────────────────────────────────
        embedder = _get_embedder()
        embeddings = embedder.embed_chunks(chunks)

        # ── Step 6: Store ─────────────────────────────────────────────────────
        store = _get_vector_store()
        store.add_chunks(chunks, embeddings)

        # ── Step 7: Update DB ─────────────────────────────────────────────────
        video = db.query(Video).filter(Video.id == asset.video_id).first()
        if video:
            video.status = "indexed"
            video.chunk_count = len(chunks)
            video.language = transcript.language
            video.indexed_at = datetime.utcnow()
            db.commit()

        update_job("done", f"Indexed {len(chunks)} chunks successfully", video_id=asset.video_id)
        logger.success(f"[Ingest] Job {job_id} complete — {len(chunks)} chunks indexed")

    except Exception as e:
        logger.error(f"[Ingest] Job {job_id} failed: {e}")
        update_job("error", "Ingestion failed", error=str(e))

        # Mark video as error if it was created
        try:
            if "asset" in locals() and hasattr(asset, "video_id"):
                video = db.query(Video).filter(Video.id == asset.video_id).first()
                if video:
                    video.status = "error"
                    video.error_message = str(e)
                    db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _detect_source_type(source: str) -> str:
    if source.startswith("http") and ("youtu" in source or "vimeo" in source or "twitch" in source):
        return "youtube"
    if source.startswith("rtmp") or ".m3u8" in source:
        return "live_stream"
    if source.startswith("http"):
        return "youtube"
    return "local_file"


def _get_ingester(source_type: str, platform: str | None = None, credentials: dict[str, str] | None = None):
    from ingestion import LiveStreamIngester, LocalFileIngester, YouTubeIngester
    from ingestion.video_api import get_api_ingester

    if source_type == "youtube":
        return YouTubeIngester(audio_dir=settings.audio_dir)
    elif source_type == "local_file":
        return LocalFileIngester(audio_dir=settings.audio_dir)
    elif source_type == "live_stream":
        return LiveStreamIngester(audio_dir=settings.audio_dir)
    elif source_type == "video_api":
        return get_api_ingester(platform or "vimeo", credentials or {}, audio_dir=settings.audio_dir)
    else:
        return YouTubeIngester(audio_dir=settings.audio_dir)
