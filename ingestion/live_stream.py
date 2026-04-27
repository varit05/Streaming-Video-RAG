"""
Live stream ingester for HLS (.m3u8) and RTMP streams.
Captures the stream in fixed-length segments and yields VideoAsset objects
so each segment can be processed through the pipeline independently.
This enables near-real-time indexing of live content.
"""

from pathlib import Path
from typing import Callable, Generator, Optional

import ffmpeg
from loguru import logger

from config import settings

from .base import BaseIngester, SourceType, VideoAsset


class LiveStreamIngester(BaseIngester):
    """
    Captures a live stream in fixed segments (default: 60 seconds each).
    Each segment is yielded as a VideoAsset so the caller can process
    them through the transcription + RAG pipeline incrementally.

    Usage:
        ingester = LiveStreamIngester()
        for asset in ingester.ingest_stream("https://example.com/stream.m3u8"):
            # process asset through the pipeline
            pipeline.process(asset)
    """

    def validate(self, source: str) -> bool:
        return (
            source.startswith("rtmp://")
            or source.startswith("rtmps://")
            or ".m3u8" in source
            or source.startswith("http")
            and "stream" in source.lower()
        )

    def ingest(self, source: str, video_id: Optional[str] = None) -> VideoAsset:
        """
        Capture a single fixed-length segment from the stream.
        For continuous ingestion, use ingest_stream() instead.
        """
        if video_id is None:
            video_id = VideoAsset.generate_id()

        segment_seconds = settings.live_stream_segment_seconds
        audio_path = self._audio_path(video_id)

        logger.info(f"[LiveStream] Capturing {segment_seconds}s segment from: {source}")
        self._capture_segment(source, audio_path, duration_seconds=segment_seconds)
        logger.success(f"[LiveStream] Segment captured → {audio_path}")

        return VideoAsset(
            video_id=video_id,
            title=f"Live Stream Segment — {video_id[:8]}",
            source_url=source,
            source_type=SourceType.LIVE_STREAM,
            local_audio_path=audio_path,
            duration_seconds=float(segment_seconds),
            extra_metadata={"segment_type": "live", "capture_duration": segment_seconds},
        )

    def ingest_stream(
        self,
        source: str,
        max_segments: Optional[int] = None,
        on_segment: Optional[Callable[[VideoAsset], None]] = None,
    ) -> Generator[VideoAsset, None, None]:
        """
        Continuously capture segments from a live stream.

        Args:
            source: Stream URL (HLS/RTMP)
            max_segments: Stop after N segments (None = run indefinitely)
            on_segment: Optional callback for each captured segment

        Yields:
            VideoAsset for each captured segment
        """
        segment_count = 0
        logger.info(f"[LiveStream] Starting continuous capture: {source}")

        try:
            while max_segments is None or segment_count < max_segments:
                segment_id = VideoAsset.generate_id()
                asset = self.ingest(source, video_id=segment_id)
                segment_count += 1

                if on_segment:
                    on_segment(asset)

                yield asset

                logger.debug(f"[LiveStream] Segment {segment_count} done, capturing next...")

        except KeyboardInterrupt:
            logger.info("[LiveStream] Capture stopped by user.")
        except Exception as e:
            logger.error(f"[LiveStream] Capture error: {e}")
            raise

    # ── Private helpers ──────────────────────────────────────────────────────

    def _capture_segment(self, source: str, output: Path, duration_seconds: int) -> None:
        """
        Use ffmpeg to read from the stream for `duration_seconds` and save as WAV.
        """
        try:
            (
                ffmpeg.input(source, t=duration_seconds)
                .output(
                    str(output),
                    ar=16000,
                    ac=1,
                    acodec="pcm_s16le",
                )
                .overwrite_output()
                .run(quiet=True, timeout=duration_seconds + 30)
            )
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"ffmpeg stream capture failed: {stderr[:500]}") from e
