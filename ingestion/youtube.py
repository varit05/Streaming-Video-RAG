"""
YouTube / public URL ingester using yt-dlp.
Works with any URL that yt-dlp supports: YouTube, Vimeo public links,
Twitter/X videos, direct video URLs, etc.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger

from .base import BaseIngester, VideoAsset, SourceType


class YouTubeIngester(BaseIngester):
    """
    Ingests video from YouTube or any yt-dlp-compatible URL.
    Downloads only the audio stream (no video download needed for RAG).
    """

    SUPPORTED_DOMAINS = [
        "youtube.com", "youtu.be", "youtube-nocookie.com",
        "vimeo.com",     # public Vimeo links
        "twitch.tv",     # VODs
        "twitter.com", "x.com",
        "reddit.com",
        "dailymotion.com",
    ]

    def validate(self, source: str) -> bool:
        """Accept any http(s) URL — yt-dlp handles validation at runtime."""
        return source.startswith("http://") or source.startswith("https://")

    def ingest(self, source: str, video_id: Optional[str] = None) -> VideoAsset:
        if video_id is None:
            video_id = VideoAsset.generate_id()

        logger.info(f"[YouTube] Ingesting: {source}")

        # ── Step 1: fetch metadata (no download) ─────────────────────────────
        metadata = self._fetch_metadata(source)
        audio_path = self._audio_path(video_id)

        # ── Step 2: download audio only ───────────────────────────────────────
        self._download_audio(source, audio_path)

        # ── Step 3: extract chapters if present ───────────────────────────────
        chapters = [
            {
                "title": ch.get("title", f"Chapter {i+1}"),
                "start": ch.get("start_time", 0),
                "end": ch.get("end_time", 0),
            }
            for i, ch in enumerate(metadata.get("chapters") or [])
        ]

        logger.success(f"[YouTube] Done → {audio_path}")

        return VideoAsset(
            video_id=video_id,
            title=metadata.get("title", source),
            source_url=source,
            source_type=SourceType.YOUTUBE,
            local_audio_path=audio_path,
            duration_seconds=metadata.get("duration"),
            description=metadata.get("description"),
            chapters=chapters,
            thumbnail_url=metadata.get("thumbnail"),
            uploader=metadata.get("uploader"),
            upload_date=metadata.get("upload_date"),
            extra_metadata={
                "view_count": metadata.get("view_count"),
                "like_count": metadata.get("like_count"),
                "tags": metadata.get("tags", []),
            },
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _fetch_metadata(self, url: str) -> dict:
        """Run yt-dlp in dump-only mode to get video metadata as JSON."""
        try:
            result = subprocess.run(
                ["yt-dlp", "--dump-json", "--no-playlist", url],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning(f"[YouTube] Metadata fetch warning: {result.stderr[:200]}")
                return {}
            return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"[YouTube] Metadata fetch failed: {e}")
            return {}

    def _download_audio(self, url: str, output_path: Path) -> None:
        """
        Download audio only and convert to 16kHz mono WAV using yt-dlp + ffmpeg.
        This is the format Whisper expects.
        """
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
            "-o", str(output_path.with_suffix("")),  # yt-dlp adds extension
            url,
        ]
        logger.debug(f"[YouTube] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")

        # yt-dlp may add the .wav extension itself; handle both
        if not output_path.exists():
            wav_candidate = output_path.with_suffix("").with_suffix(".wav")
            if wav_candidate.exists():
                wav_candidate.rename(output_path)
            else:
                raise FileNotFoundError(f"Expected audio at {output_path} but not found")
