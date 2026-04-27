"""
Video summarizer using a map-reduce approach.
Works well on long videos (1hr+) by summarizing chunks independently
and then combining them into a final summary.
"""

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from llm.factory import get_llm
from vector_store import SearchResult, get_vector_store

MAP_SYSTEM = "You are a concise summarizer of video transcript excerpts."

MAP_TEMPLATE = """\
Summarize the key points from this video transcript excerpt in 2-4 sentences.
Focus on the main ideas and any specific facts, names, or data mentioned.
Do not include timestamps.

Transcript excerpt ({start_ts}–{end_ts}):
{text}
"""

REDUCE_SYSTEM = """\
You are a skilled editor who combines multiple transcript summaries into
a single coherent summary of a full video.
"""

REDUCE_TEMPLATE = """\
Below are summaries of sequential sections of a video titled "{title}".
Combine them into a single, well-structured summary in 3-6 paragraphs.
Preserve key details, facts, and insights. Do not use bullet points.

Section summaries:
{summaries}
"""

CHAPTER_TEMPLATE = """\
For each chapter in this video, write a 1-2 sentence summary based on the content.

Chapters and their transcript content:
{chapter_content}
"""


@dataclass
class VideoSummary:
    """Result of summarizing a video."""

    video_id: str
    title: str
    overall_summary: str
    chapter_summaries: list[dict[str, str]] | None = None  # [{chapter, summary}]
    chunk_count: int = 0


class Summarizer:
    """
    Generates summaries of indexed videos using map-reduce over chunks.
    """

    def __init__(self):
        self.store = get_vector_store()

    def summarize(
        self,
        video_id: str,
        title: str = "",
        include_chapters: bool = True,
    ) -> VideoSummary:
        """
        Summarize a video that has already been indexed.

        Args:
            video_id: The video to summarize
            title: Video title (for the prompt)
            include_chapters: Whether to also generate per-chapter summaries

        Returns:
            VideoSummary with overall summary and optional chapter breakdowns
        """
        logger.info(f"[Summarizer] Starting summary for video {video_id}")

        # ── Fetch all chunks for this video ───────────────────────────────────
        # We retrieve ALL chunks (not just top-k) by using a high limit
        from config import EmbeddingMode, settings

        dim = 1536 if settings.embedding_mode == EmbeddingMode.OPENAI else 384
        dummy_vector = [0.0] * dim
        all_results = self.store.search(dummy_vector, top_k=9999, filter_video_id=video_id)

        # Sort by start time
        all_results.sort(key=lambda r: r.chunk.start_time)

        if not all_results:
            return VideoSummary(
                video_id=video_id,
                title=title,
                overall_summary="No content found for this video.",
            )

        logger.info(f"[Summarizer] Processing {len(all_results)} chunks")

        # ── Map: summarize each chunk ─────────────────────────────────────────
        chunk_summaries = self._map_chunks(all_results)

        # ── Reduce: combine into final summary ────────────────────────────────
        overall = self._reduce(chunk_summaries, title or video_id)

        # ── Chapter summaries (optional) ──────────────────────────────────────
        chapter_summaries = None
        if include_chapters:
            chapters = self._group_by_chapter(all_results)
            if len(chapters) > 1:
                chapter_summaries = self._summarize_chapters(chapters)

        logger.success(f"[Summarizer] Done for {video_id}")

        return VideoSummary(
            video_id=video_id,
            title=title,
            overall_summary=overall,
            chapter_summaries=chapter_summaries,
            chunk_count=len(all_results),
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _map_chunks(self, results: list[SearchResult]) -> list[str]:
        """Summarize each chunk individually (map phase)."""
        llm = get_llm(temperature=0.2)
        summaries = []

        for r in results:
            prompt = MAP_TEMPLATE.format(
                start_ts=r.chunk.start_ts,
                end_ts=r.chunk.end_ts,
                text=r.chunk.text,
            )
            response = llm.invoke([SystemMessage(content=MAP_SYSTEM), HumanMessage(content=prompt)])
            summaries.append(response.content.strip())

        return summaries

    def _reduce(self, summaries: list[str], title: str) -> str:
        """Combine chunk summaries into a final overall summary (reduce phase)."""
        llm = get_llm(temperature=0.3)
        combined = "\n\n".join(f"[Section {i + 1}] {s}" for i, s in enumerate(summaries))
        prompt = REDUCE_TEMPLATE.format(title=title, summaries=combined)
        response = llm.invoke([SystemMessage(content=REDUCE_SYSTEM), HumanMessage(content=prompt)])
        return response.content.strip()

    def _group_by_chapter(self, results: list[SearchResult]) -> dict[str, list[SearchResult]]:
        """Group chunks by their chapter label."""
        grouped = {}
        for r in results:
            key = r.chunk.chapter or "Main Content"
            grouped.setdefault(key, []).append(r)
        return grouped

    def _summarize_chapters(self, chapters: dict[str, list[SearchResult]]) -> list[dict[str, str]]:
        """Generate a brief summary for each chapter."""
        llm = get_llm(temperature=0.2)
        chapter_content = "\n\n".join(
            f"Chapter: {ch}\nContent: {' '.join(r.chunk.text for r in chunks[:3])}" for ch, chunks in chapters.items()
        )
        prompt = CHAPTER_TEMPLATE.format(chapter_content=chapter_content)
        response = llm.invoke([SystemMessage(content=MAP_SYSTEM), HumanMessage(content=prompt)])

        result = []
        for ch in chapters:
            result.append({"chapter": ch, "summary": ""})

        # Parse the response text to extract per-chapter summaries
        lines = response.content.strip().split("\n")
        current_chapter_idx = 0
        buffer = []
        for line in lines:
            matched = False
            for i, ch in enumerate(chapters.keys()):
                if ch.lower() in line.lower():
                    if buffer and current_chapter_idx < len(result):
                        result[current_chapter_idx]["summary"] = " ".join(buffer).strip()
                    current_chapter_idx = i
                    buffer = []
                    matched = True
                    break
            if not matched and line.strip():
                buffer.append(line.strip())
        if buffer and current_chapter_idx < len(result):
            result[current_chapter_idx]["summary"] = " ".join(buffer).strip()

        return result
