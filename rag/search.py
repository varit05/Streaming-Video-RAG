"""
Search engine — semantic search across indexed video content.
Returns ranked chunks with timestamps, ready for display.
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from vector_store import SearchResult

from .retriever import get_retriever


@dataclass
class SearchResponse:
    """Structured search result for API responses."""

    query: str
    total_results: int
    results: list[dict[str, object]]


class SearchEngine:
    """
    Semantic search over indexed video content.
    Supports optional filtering by video_id.
    """

    def __init__(self):
        self.retriever = get_retriever()

    def search(
        self,
        query: str,
        top_k: int = 10,
        video_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> SearchResponse:
        """
        Search for video segments relevant to the query.

        Args:
            query: Search string
            top_k: Max number of results
            video_id: If set, search only within this video
            min_score: Filter out results below this similarity score

        Returns:
            SearchResponse with ranked results
        """
        logger.info(f"[Search] Query='{query[:60]}', top_k={top_k}")

        raw_results = self.retriever.retrieve(query, top_k=top_k, video_id=video_id)

        # Apply score threshold
        filtered = [r for r in raw_results if r.score >= min_score]

        results = [self._format_result(r, i + 1) for i, r in enumerate(filtered)]

        return SearchResponse(
            query=query,
            total_results=len(results),
            results=results,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _format_result(self, result: SearchResult, rank: int) -> dict[str, object]:
        chunk = result.chunk
        return {
            "rank": rank,
            "score": round(result.score, 4),
            "video_id": chunk.video_id,
            "title": chunk.title,
            "source_url": chunk.source_url,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "start_ts": chunk.start_ts,
            "end_ts": chunk.end_ts,
            "chapter": chunk.chapter,
            "text": chunk.text,
            "deep_link": chunk.youtube_deep_link(),
        }
