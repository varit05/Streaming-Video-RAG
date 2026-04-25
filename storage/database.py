"""
SQLAlchemy database models and session management.
Tracks video metadata and ingestion job state.
"""

import uuid
from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings

# ── Engine + session ─────────────────────────────────────────────────────────

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ── Models ───────────────────────────────────────────────────────────────────


class Video(Base):
    """Stores metadata for each indexed video."""

    __tablename__ = "videos"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    source_url = Column(String, nullable=False)
    source_type = Column(String, nullable=False)  # youtube | local_file | live_stream | video_api
    duration_seconds = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    uploader = Column(String, nullable=True)
    upload_date = Column(String, nullable=True)
    language = Column(String, default="en")
    chunk_count = Column(Integer, default=0)
    status = Column(
        SAEnum("pending", "processing", "indexed", "error", name="video_status"),
        default="pending",
    )
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "duration_seconds": self.duration_seconds,
            "description": self.description,
            "uploader": self.uploader,
            "upload_date": self.upload_date,
            "language": self.language,
            "chunk_count": self.chunk_count,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
        }


class IngestJob(Base):
    """Tracks the status of each ingestion job."""

    __tablename__ = "ingest_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, nullable=True)  # populated once ingestion starts
    source = Column(String, nullable=False)  # original source URL or path
    source_type = Column(String, nullable=False)
    status = Column(
        SAEnum("queued", "ingesting", "transcribing", "indexing", "done", "error", name="job_status"),
        default="queued",
    )
    progress_message = Column(String, nullable=True)  # e.g. "Transcribing audio..."
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "source": self.source,
            "source_type": self.source_type,
            "status": self.status,
            "progress_message": self.progress_message,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────


def init_db() -> None:
    """Create all tables (safe to call multiple times)."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a DB session and close it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
