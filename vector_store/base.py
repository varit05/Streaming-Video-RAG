"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from processing.chunker import VideoChunk


@dataclass
class SearchResult:
    """A retrieved chunk with its similarity score."""
    chunk: VideoChunk
    score: float        # higher = more similar (normalized 0-1)

    @property
    def video_id(self) -> str:
        return self.chunk.video_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def start_ts(self) -> str:
        return self.chunk.start_ts

    @property
    def end_ts(self) -> str:
        return self.chunk.end_ts


class BaseVectorStore(ABC):

    @abstractmethod
    def add_chunks(self, chunks: list[VideoChunk], embeddings: list[list[float]]) -> None:
        """Store chunks and their embeddings."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_video_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Return top_k most similar chunks, optionally filtered by video_id."""
        ...

    @abstractmethod
    def delete_video(self, video_id: str) -> int:
        """Remove all chunks for a given video. Returns number deleted."""
        ...

    @abstractmethod
    def count(self, video_id: Optional[str] = None) -> int:
        """Count indexed chunks, optionally scoped to a video."""
        ...
