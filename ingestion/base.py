"""
Abstract base class for all video ingesters.
Every ingester must produce a VideoAsset containing at minimum:
  - a unique video_id
  - a local path to the extracted audio file (WAV)
  - basic metadata (title, duration, source_url, source_type)
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class SourceType(str, Enum):
    YOUTUBE = "youtube"
    LOCAL_FILE = "local_file"
    LIVE_STREAM = "live_stream"
    VIDEO_API = "video_api"


@dataclass
class VideoAsset:
    """Represents a video that has been ingested and is ready for transcription."""

    video_id: str
    title: str
    source_url: str
    source_type: SourceType
    local_audio_path: Path          # Path to extracted 16kHz mono WAV
    duration_seconds: Optional[float] = None
    description: Optional[str] = None
    chapters: list[dict] = field(default_factory=list)  # [{title, start, end}]
    thumbnail_url: Optional[str] = None
    uploader: Optional[str] = None
    upload_date: Optional[str] = None
    extra_metadata: dict = field(default_factory=dict)

    @classmethod
    def generate_id(cls) -> str:
        return str(uuid.uuid4())


class BaseIngester(ABC):
    """
    All ingesters follow this interface:
      1. validate(source)    — check the source is accessible
      2. ingest(source)      — download/capture and extract audio
      3. Returns a VideoAsset ready for the transcription stage
    """

    def __init__(self, audio_dir: str = "./data/audio"):
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def validate(self, source: str) -> bool:
        """Return True if this ingester can handle the given source."""
        ...

    @abstractmethod
    def ingest(self, source: str, video_id: Optional[str] = None) -> VideoAsset:
        """
        Download or capture the source, extract audio to a WAV file,
        and return a populated VideoAsset.
        """
        ...

    def _audio_path(self, video_id: str) -> Path:
        return self.audio_dir / f"{video_id}.wav"
