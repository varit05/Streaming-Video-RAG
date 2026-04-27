"""
Video API ingester — connects to platform APIs (Vimeo, Twitch, etc.)
to fetch video metadata and download content via authenticated endpoints.

Each platform is a subclass of VideoAPIIngester. The factory function
`get_api_ingester(platform, credentials)` returns the right one.
"""

import subprocess
from pathlib import Path
from typing import Optional, Any

import requests
from loguru import logger

from .base import BaseIngester, SourceType, VideoAsset

# ── Base API Ingester ────────────────────────────────────────────────────────


class VideoAPIIngester(BaseIngester):
    """
    Base class for API-authenticated video ingesters.
    Subclasses implement _fetch_metadata() and _get_download_url().
    """

    def __init__(self, credentials: dict[str, str], audio_dir: str = "./data/audio"):
        super().__init__(audio_dir)
        self.credentials = credentials
        self.session = requests.Session()
        self._setup_auth()

    def _setup_auth(self) -> None:
        """Configure authentication headers. Override in subclasses."""
        pass

    def validate(self, source: str) -> bool:
        raise NotImplementedError

    def ingest(self, source: str, video_id: Optional[str] = None) -> VideoAsset:
        raise NotImplementedError


# ── Vimeo ────────────────────────────────────────────────────────────────────


class VimeoIngester(VideoAPIIngester):
    """
    Vimeo API ingester. Requires a Vimeo API access token.
    credentials = {"access_token": "..."}
    """

    API_BASE = "https://api.vimeo.com"

    def _setup_auth(self) -> None:
        token = self.credentials.get("access_token", "")
        self.session.headers.update(
            {
                "Authorization": f"bearer {token}",
                "Accept": "application/vnd.vimeo.*+json;version=3.4",
            }
        )

    def validate(self, source: str) -> bool:
        return "vimeo.com" in source

    def ingest(self, source: str, video_id: Optional[str] = None) -> VideoAsset:
        vimeo_id = self._extract_vimeo_id(source)
        if video_id is None:
            video_id = VideoAsset.generate_id()

        logger.info(f"[Vimeo] Ingesting video {vimeo_id}")

        metadata = self._fetch_metadata(vimeo_id)
        audio_path = self._audio_path(video_id)
        self._download_via_ytdlp(source, audio_path)

        return VideoAsset(
            video_id=video_id,
            title=metadata.get("name", source),
            source_url=source,
            source_type=SourceType.VIDEO_API,
            local_audio_path=audio_path,
            duration_seconds=metadata.get("duration"),
            description=metadata.get("description"),
            uploader=metadata.get("user", {}).get("name"),
            extra_metadata={
                "platform": "vimeo",
                "vimeo_id": vimeo_id,
                "privacy": metadata.get("privacy", {}).get("view"),
            },
        )

    def _extract_vimeo_id(self, url: str) -> str:
        parts = url.rstrip("/").split("/")
        return parts[-1]

    def _fetch_metadata(self, vimeo_id: str) -> dict[str, Any]:
        try:
            resp = self.session.get(f"{self.API_BASE}/videos/{vimeo_id}", timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"[Vimeo] Metadata fetch failed: {e}")
            return {}

    def _download_via_ytdlp(self, url: str, output: Path) -> None:
        """Fall back to yt-dlp for Vimeo download (handles auth via cookies if needed)."""
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--postprocessor-args",
            "ffmpeg:-ar 16000 -ac 1",
            "-o",
            str(output.with_suffix("")),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp (Vimeo) failed: {result.stderr[:300]}")
        if not output.exists():
            candidate = output.with_suffix("").with_suffix(".wav")
            if candidate.exists():
                candidate.rename(output)


# ── Twitch ───────────────────────────────────────────────────────────────────


class TwitchIngester(VideoAPIIngester):
    """
    Twitch API ingester for VODs (past broadcasts).
    credentials = {"client_id": "...", "client_secret": "..."}
    """

    AUTH_URL = "https://id.twitch.tv/oauth2/token"
    API_BASE = "https://api.twitch.tv/helix"

    def _setup_auth(self) -> None:
        # Get app access token
        try:
            resp = requests.post(
                self.AUTH_URL,
                params={
                    "client_id": self.credentials.get("client_id", ""),
                    "client_secret": self.credentials.get("client_secret", ""),
                    "grant_type": "client_credentials",
                },
                timeout=15,
            )
            resp.raise_for_status()
            token = resp.json().get("access_token", "")
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {token}",
                    "Client-Id": self.credentials.get("client_id", ""),
                }
            )
        except requests.RequestException as e:
            logger.warning(f"[Twitch] Auth setup failed: {e}")

    def validate(self, source: str) -> bool:
        return "twitch.tv" in source

    def ingest(self, source: str, video_id: Optional[str] = None) -> VideoAsset:
        twitch_id = self._extract_twitch_vod_id(source)
        if video_id is None:
            video_id = VideoAsset.generate_id()

        logger.info(f"[Twitch] Ingesting VOD {twitch_id}")

        metadata = self._fetch_vod_metadata(twitch_id)
        audio_path = self._audio_path(video_id)
        self._download_via_ytdlp(source, audio_path)

        return VideoAsset(
            video_id=video_id,
            title=metadata.get("title", source),
            source_url=source,
            source_type=SourceType.VIDEO_API,
            local_audio_path=audio_path,
            duration_seconds=self._parse_duration(metadata.get("duration", "")),
            description=metadata.get("description"),
            uploader=metadata.get("user_name"),
            upload_date=metadata.get("created_at", "")[:10],
            extra_metadata={
                "platform": "twitch",
                "twitch_vod_id": twitch_id,
                "view_count": metadata.get("view_count"),
                "game_name": metadata.get("game_name"),
            },
        )

    def _extract_twitch_vod_id(self, url: str) -> str:
        parts = url.rstrip("/").split("/")
        return parts[-1].replace("v", "")

    def _fetch_vod_metadata(self, vod_id: str) -> dict[str, Any]:
        try:
            resp = self.session.get(f"{self.API_BASE}/videos", params={"id": vod_id}, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return data[0] if data else {}
        except requests.RequestException as e:
            logger.warning(f"[Twitch] Metadata fetch failed: {e}")
            return {}

    def _parse_duration(self, duration_str: str) -> Optional[float]:
        """Parse Twitch duration format '1h2m3s' into seconds."""
        if not duration_str:
            return None
        total = 0
        import re

        for value, unit in re.findall(r"(\d+)([hms])", duration_str):
            if unit == "h":
                total += int(value) * 3600
            elif unit == "m":
                total += int(value) * 60
            elif unit == "s":
                total += int(value)
        return float(total) if total else None

    def _download_via_ytdlp(self, url: str, output: Path) -> None:
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--postprocessor-args",
            "ffmpeg:-ar 16000 -ac 1",
            "-o",
            str(output.with_suffix("")),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp (Twitch) failed: {result.stderr[:300]}")
        if not output.exists():
            candidate = output.with_suffix("").with_suffix(".wav")
            if candidate.exists():
                candidate.rename(output)


# ── Factory ──────────────────────────────────────────────────────────────────


def get_api_ingester(platform: str, credentials: dict[str, str], audio_dir: str = "./data/audio") -> VideoAPIIngester:
    """
    Factory: return the right VideoAPIIngester for the given platform name.

    Args:
        platform: "vimeo" | "twitch"
        credentials: dict of API credentials (platform-specific)
        audio_dir: where to store extracted audio

    Returns:
        A ready-to-use VideoAPIIngester subclass
    """
    registry = {
        "vimeo": VimeoIngester,
        "twitch": TwitchIngester,
    }
    cls = registry.get(platform.lower())
    if cls is None:
        raise ValueError(f"Unknown platform '{platform}'. Supported: {list(registry.keys())}")
    return cls(credentials=credentials, audio_dir=audio_dir)
