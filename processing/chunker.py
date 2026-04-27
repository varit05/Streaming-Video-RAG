"""
Timestamp-aware chunker.
Merges Whisper transcript segments into larger overlapping chunks
suitable for embedding and retrieval.

Each chunk knows its exact start/end time, so search results
can link directly to the right moment in the video.
"""

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from config import settings
from ingestion.base import VideoAsset
from transcription.whisper_transcriber import Transcript, TranscriptSegment


@dataclass
class VideoChunk:
    """
    A single embeddable unit of video content.
    Carries enough metadata to reconstruct a deep link to the source video.
    """

    chunk_id: str
    video_id: str
    text: str
    start_time: float  # seconds
    end_time: float  # seconds
    segment_index: int  # position among all chunks for this video
    title: str = ""
    source_url: str = ""
    chapter: Optional[str] = None
    language: str = "en"
    extra_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def start_ts(self) -> str:
        m, s = divmod(int(self.start_time), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def end_ts(self) -> str:
        m, s = divmod(int(self.end_time), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def youtube_deep_link(self) -> Optional[str]:
        """Return a ?t= deep link if the source is a YouTube URL."""
        if "youtu" in self.source_url:
            t = int(self.start_time)
            base = self.source_url.split("&t=")[0].split("?t=")[0]
            sep = "&" if "?" in base else "?"
            return f"{base}{sep}t={t}"
        return None

    def to_metadata_dict(self) -> dict[str, str | float | int]:
        """Returns the flat dict stored alongside the embedding in the vector DB."""
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "title": self.title,
            "source_url": self.source_url,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "segment_index": self.segment_index,
            "chapter": self.chapter or "",
            "language": self.language,
            **{f"extra_{k}": str(v) for k, v in self.extra_metadata.items()},
        }


class Chunker:
    """
    Splits a Transcript into overlapping VideoChunks.

    Strategy:
      - Group consecutive segments until the window exceeds chunk_duration_seconds
      - Start the next chunk chunk_overlap_seconds before the previous one ended
      - Assign chapter labels from VideoAsset.chapters if available
    """

    def __init__(
        self,
        chunk_duration: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_duration = chunk_duration or settings.chunk_duration_seconds
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap_seconds

    def chunk(self, transcript: Transcript, asset: VideoAsset) -> list[VideoChunk]:
        """
        Convert a Transcript into a list of VideoChunks.

        Args:
            transcript: The Whisper transcript with timestamped segments
            asset: The VideoAsset (for title, URL, chapters, metadata)

        Returns:
            List of VideoChunk objects ready for embedding
        """
        if not transcript.segments:
            logger.warning(f"[Chunker] Transcript for {asset.video_id} has no segments")
            return []

        chunks: list[VideoChunk] = []
        segments = transcript.segments
        n = len(segments)
        i = 0
        chunk_index = 0

        while i < n:
            start_i = i
            window_start = segments[i].start
            window_end = window_start + self.chunk_duration
            j = i

            # Collect segments that fall within this window
            window_segments: list[TranscriptSegment] = []
            while j < n and segments[j].start < window_end:
                window_segments.append(segments[j])
                j += 1

            if not window_segments:
                i += 1
                continue

            chunk_text = " ".join(s.text.strip() for s in window_segments)
            actual_start = window_segments[0].start
            actual_end = window_segments[-1].end

            chapter = self._get_chapter(actual_start, asset.chapters)

            chunk = VideoChunk(
                chunk_id=f"{asset.video_id}_{chunk_index:04d}",
                video_id=asset.video_id,
                text=chunk_text,
                start_time=actual_start,
                end_time=actual_end,
                segment_index=chunk_index,
                title=asset.title,
                source_url=asset.source_url,
                chapter=chapter,
                language=transcript.language,
                extra_metadata=asset.extra_metadata,
            )
            chunks.append(chunk)
            chunk_index += 1

            # Move forward to where the overlap begins
            overlap_start = actual_end - self.chunk_overlap
            while i < n and segments[i].start < overlap_start:
                i += 1

            # Safety: always advance at least one segment to guarantee termination.
            # This prevents infinite loops on the last chunk when overlap_start
            # falls before the first segment of the current window.
            if i <= start_i:
                i = start_i + 1

        logger.info(f"[Chunker] {asset.video_id} → {len(chunks)} chunks")
        return chunks

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_chapter(self, time: float, chapters: list[dict[str, str | float]]) -> Optional[str]:
        """Return the chapter title for the given timestamp, if chapters exist."""
        for ch in chapters:
            start = float(ch.get("start", 0))
            end = float(ch.get("end", float("inf")))
            if start <= time < end:
                title = ch.get("title")
                return str(title) if title is not None else None
        return None
