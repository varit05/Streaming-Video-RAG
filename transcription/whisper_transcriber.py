"""
Whisper-based transcription with two modes:
  - LOCAL:      runs openai-whisper on-device (free, slower)
  - OPENAI_API: calls the OpenAI Whisper API (faster, costs ~$0.006/min)

Controlled by WHISPER_MODE env var (default: local).
Local model size controlled by WHISPER_MODEL_SIZE (default: base).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import WhisperMode, settings

# ── Data models ──────────────────────────────────────────────────────────────


@dataclass
class TranscriptSegment:
    """One time-bounded segment of transcribed speech."""

    start: float  # seconds
    end: float  # seconds
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def format_timestamp(self, t: float) -> str:
        m, s = divmod(int(t), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def start_ts(self) -> str:
        return self.format_timestamp(self.start)

    @property
    def end_ts(self) -> str:
        return self.format_timestamp(self.end)


@dataclass
class Transcript:
    """Full transcript of a video, composed of time-stamped segments."""

    video_id: str
    language: str
    segments: list[TranscriptSegment]
    full_text: str = ""

    def __post_init__(self):
        if not self.full_text:
            self.full_text = " ".join(s.text.strip() for s in self.segments)

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "language": self.language,
            "full_text": self.full_text,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in self.segments],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        logger.debug(f"Transcript saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        data = json.loads(path.read_text())
        segments = [TranscriptSegment(s["start"], s["end"], s["text"]) for s in data["segments"]]
        return cls(
            video_id=data["video_id"],
            language=data["language"],
            segments=segments,
            full_text=data.get("full_text", ""),
        )


# ── Transcriber ──────────────────────────────────────────────────────────────


class WhisperTranscriber:
    """
    Transcribes audio files using Whisper.
    Mode is determined by settings.whisper_mode.
    """

    def __init__(self):
        self.mode = settings.whisper_mode
        self._local_model = None  # lazy-loaded

    def transcribe(self, audio_path: Path, video_id: str, language: Optional[str] = None) -> Transcript:
        """
        Transcribe the audio file at `audio_path`.

        Args:
            audio_path: Path to 16kHz mono WAV file
            video_id: ID to associate with the transcript
            language: ISO 639-1 language code (None = auto-detect)

        Returns:
            Transcript with timestamped segments
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        logger.info(f"[Whisper/{self.mode.value}] Transcribing: {audio_path.name}")

        if self.mode == WhisperMode.LOCAL:
            return self._transcribe_local(audio_path, video_id, language)
        else:
            return self._transcribe_api(audio_path, video_id, language)

    # ── Local Whisper ────────────────────────────────────────────────────────

    def _transcribe_local(self, audio_path: Path, video_id: str, language: Optional[str]) -> Transcript:
        """Run openai-whisper locally."""
        model = self._get_local_model()

        kwargs: dict[str, bool | str] = {"verbose": False}
        if language:
            kwargs["language"] = language

        result = model.transcribe(str(audio_path), **kwargs)
        segments = [
            TranscriptSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg["text"].strip(),
            )
            for seg in result["segments"]
            if seg["text"].strip()
        ]

        detected_language = result.get("language", language or "en")
        logger.success(f"[Whisper/local] Done — {len(segments)} segments, lang={detected_language}")

        return Transcript(video_id=video_id, language=detected_language, segments=segments)

    def _get_local_model(self):
        """Lazy-load the Whisper model (avoids loading it until needed)."""
        if self._local_model is None:
            import whisper

            model_size = settings.whisper_model_size
            logger.info(f"[Whisper/local] Loading model: {model_size}")
            self._local_model = whisper.load_model(model_size)
        return self._local_model

    # ── OpenAI API Whisper ───────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _transcribe_api(self, audio_path: Path, video_id: str, language: Optional[str]) -> Transcript:
        """Call the OpenAI Whisper API."""
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=120.0,  # 2 minute timeout for API calls
            max_retries=2,
        )

        with open(audio_path, "rb") as f:
            kwargs = {
                "model": "whisper-1",
                "file": f,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            }
            if language:
                kwargs["language"] = language

            response = client.audio.transcriptions.create(**kwargs)

        raw_segments = getattr(response, "segments", []) or []
        segments = [
            TranscriptSegment(
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
                text=str(seg.get("text", "")).strip(),
            )
            for seg in raw_segments
            if str(seg.get("text", "")).strip()
        ]

        detected_language = getattr(response, "language", language or "en")
        logger.success(f"[Whisper/api] Done — {len(segments)} segments, lang={detected_language}")

        return Transcript(video_id=video_id, language=detected_language, segments=segments)
