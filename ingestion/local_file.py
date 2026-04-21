"""
Local video file ingester.
Accepts any video format that ffmpeg can read (.mp4, .mov, .avi, .mkv, .webm, etc.)
and extracts 16kHz mono WAV audio for transcription.
"""

import subprocess
from pathlib import Path
from typing import Optional

import ffmpeg
from loguru import logger

from .base import BaseIngester, VideoAsset, SourceType


SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".m4v", ".flv", ".wmv", ".ts", ".mts",
    ".mp3", ".m4a", ".aac", ".ogg", ".wav", ".flac",  # audio-only files too
}


class LocalFileIngester(BaseIngester):
    """
    Ingests a locally stored video or audio file.
    Uses ffmpeg-python to probe metadata and extract audio.
    """

    def validate(self, source: str) -> bool:
        path = Path(source)
        return path.exists() and path.suffix.lower() in SUPPORTED_EXTENSIONS

    def ingest(self, source: str, video_id: Optional[str] = None) -> VideoAsset:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        if video_id is None:
            video_id = VideoAsset.generate_id()

        logger.info(f"[LocalFile] Ingesting: {source_path.name}")

        # ── Probe metadata ────────────────────────────────────────────────────
        metadata = self._probe(source_path)
        audio_path = self._audio_path(video_id)

        # ── Extract audio ─────────────────────────────────────────────────────
        self._extract_audio(source_path, audio_path)

        duration = float(metadata.get("format", {}).get("duration", 0)) or None
        title = (
            metadata.get("format", {}).get("tags", {}).get("title")
            or source_path.stem
        )

        logger.success(f"[LocalFile] Done → {audio_path}")

        return VideoAsset(
            video_id=video_id,
            title=title,
            source_url=str(source_path.resolve()),
            source_type=SourceType.LOCAL_FILE,
            local_audio_path=audio_path,
            duration_seconds=duration,
            extra_metadata={
                "original_filename": source_path.name,
                "file_size_bytes": source_path.stat().st_size,
                "format": metadata.get("format", {}).get("format_name"),
            },
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _probe(self, path: Path) -> dict:
        """Use ffprobe to extract duration and tag metadata."""
        try:
            return ffmpeg.probe(str(path))
        except ffmpeg.Error as e:
            logger.warning(f"[LocalFile] ffprobe failed: {e}")
            return {}

    def _extract_audio(self, source: Path, output: Path) -> None:
        """Extract audio to 16kHz mono WAV using ffmpeg."""
        try:
            (
                ffmpeg
                .input(str(source))
                .output(
                    str(output),
                    ar=16000,   # 16kHz — optimal for Whisper
                    ac=1,       # mono
                    acodec="pcm_s16le",
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"ffmpeg audio extraction failed: {e.stderr.decode()}")
