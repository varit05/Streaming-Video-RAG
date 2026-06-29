"""
Video summarizer using a map-reduce approach.
Works well on long videos (1hr+) by summarizing chunks independently
and then combining them into a final summary.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from llm.factory import get_llm
from vector_store import SearchResult, get_vector_store

MAP_SYSTEM = "You are a concise summarizer of video transcript excerpts."

MAP_TEMPLATE = """\
Summarize the key points from this video transcript excerpt in 2-4 sentences.
Focus on the main ideas and any specific facts, names, or data mentioned.
Do not include timestamps.

Transcript excerpt ({start_ts}-{end_ts}):
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

    def __init__(self, max_concurrency: int = 5) -> None:
        self.store = get_vector_store()
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def summarize(
        self,
        video_id: str,
        title: str = "",
        include_chapters: bool = True,
    ) -> VideoSummary:
        """
        Synchronous wrapper for summarize_async.
        Note: Will fail if called from within an existing event loop.
        """
        return asyncio.run(self.summarize_async(video_id, title, include_chapters))

    async def summarize_async(
        self,
        video_id: str,
        title: str = "",
        include_chapters: bool = True,
    ) -> VideoSummary:
        """
        Summarize a video that has already been indexed (Asynchronous).

        Args:
            video_id: The video to summarize
            title: Video title (for the prompt)
            include_chapters: Whether to also generate per-chapter summaries

        Returns:
            VideoSummary with overall summary and optional chapter breakdowns
        """
        logger.info(f"[Summarizer] Starting summary for video {video_id}")

        # ── Fetch all chunks for this video ───────────────────────────────────
        from config import EmbeddingMode, settings

        dim = 1536 if settings.embedding_mode == EmbeddingMode.OPENAI else 384
        dummy_vector = [0.0] * dim
        all_results = self.store.search(
            dummy_vector, top_k=9999, filter_video_id=video_id
        )

        # Sort by start time
        all_results.sort(key=lambda r: r.chunk.start_time)

        if not all_results:
            return VideoSummary(
                video_id=video_id,
                title=title,
                overall_summary="No content found for this video.",
            )

        logger.info(f"[Summarizer] Processing {len(all_results)} chunks")

        # ── Map: summarize each chunk in parallel ─────────────────────────────
        chunk_summaries = await self._amap_chunks(all_results)

        if not chunk_summaries:
            return VideoSummary(
                video_id=video_id,
                title=title,
                overall_summary="Failed to summarize individual video segments.",
            )

        # ── Reduce: combine into final summary ────────────────────────────────
        # Use recursive reduce if there are many summaries to avoid context overflow
        overall = await self._recursive_reduce(chunk_summaries, title or video_id)

        # ── Chapter summaries (optional) ──────────────────────────────────────
        chapter_summaries = None
        if include_chapters:
            chapters = self._group_by_chapter(all_results)
            if len(chapters) > 1:
                try:
                    chapter_summaries = await self._asummarize_chapters(chapters)
                except Exception as e:
                    logger.warning(f"[Summarizer] Chapter summary failed: {e}")

        logger.success(f"[Summarizer] Done for {video_id}")

        return VideoSummary(
            video_id=video_id,
            title=title,
            overall_summary=overall,
            chapter_summaries=chapter_summaries,
            chunk_count=len(all_results),
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _call_llm_with_retry(self, messages: list[Any]) -> Any:
        """Call LLM with internal semaphore and retry logic."""
        async with self.semaphore:
            llm = get_llm(temperature=0.2)
            return await llm.ainvoke(messages)

    async def _amap_chunks(self, results: list[SearchResult]) -> list[str]:
        """Summarize each chunk individually in parallel."""
        # Process in batches to control the flow
        batch_size = 5
        all_summaries = []

        for i in range(0, len(results), batch_size):
            batch = results[i : i + batch_size]
            batch_tasks = []
            for r in batch:
                prompt = MAP_TEMPLATE.format(
                    start_ts=r.chunk.start_ts,
                    end_ts=r.chunk.end_ts,
                    text=r.chunk.text,
                )
                messages = [
                    SystemMessage(content=MAP_SYSTEM),
                    HumanMessage(content=prompt),
                ]
                # Wrap each call to handle failure of a single chunk gracefully
                batch_tasks.append(self._safe_call_llm(messages))

            responses = await asyncio.gather(*batch_tasks)
            for resp in responses:
                if resp:
                    content = getattr(resp, "content", str(resp))
                    if not isinstance(content, str):
                        content = str(content)
                    all_summaries.append(content.strip())

        return all_summaries

    async def _safe_call_llm(self, messages: list[Any]) -> Any | None:
        """Call LLM and return None if it fails after all retries."""
        try:
            return await self._call_llm_with_retry(messages)

        except Exception as e:
            logger.error(f"[Summarizer] LLM call failed: {e}")
            return None

    async def _recursive_reduce(self, summaries: list[str], title: str) -> str:
        """Recursively combine summaries until we have a single final one."""
        if len(summaries) <= 15:
            return await self._areduce(summaries, title)

        logger.info(
            f"[Summarizer] Large summary set ({len(summaries)}), using recursive reduce"
        )

        # Split into smaller groups of 10 and reduce each group
        next_level = []
        for i in range(0, len(summaries), 10):
            group = summaries[i : i + 10]
            if len(group) == 1:
                next_level.append(group[0])
            else:
                intermediate = await self._areduce(group, f"{title} (Part {i//10 + 1})")
                next_level.append(intermediate)

        return await self._recursive_reduce(next_level, title)

    async def _areduce(self, summaries: list[str], title: str) -> str:
        """Combine chunk summaries into a final overall summary (async)."""
        combined = "\n\n".join(
            f"[Section {i + 1}] {s}" for i, s in enumerate(summaries)
        )
        prompt = REDUCE_TEMPLATE.format(title=title, summaries=combined)
        messages = [
            SystemMessage(content=REDUCE_SYSTEM),
            HumanMessage(content=prompt),
        ]

        response = await self._call_llm_with_retry(messages)
        content = getattr(response, "content", str(response))
        if not isinstance(content, str):
            content = str(content)
        return content.strip()

    def _group_by_chapter(
        self, results: list[SearchResult]
    ) -> dict[str, list[SearchResult]]:
        """Group chunks by their chapter label."""
        grouped: dict[str, list[SearchResult]] = {}
        for r in results:
            key = r.chunk.chapter or "Main Content"
            grouped.setdefault(key, []).append(r)
        return grouped

    async def _asummarize_chapters(
        self, chapters: dict[str, list[SearchResult]]
    ) -> list[dict[str, str]]:
        """Generate a brief summary for each chapter (async)."""
        chapter_content = "\n\n".join(
            f"Chapter: {ch}\nContent: {' '.join(r.chunk.text for r in chunks[:3])}"
            for ch, chunks in chapters.items()
        )
        prompt = CHAPTER_TEMPLATE.format(chapter_content=chapter_content)
        messages = [
            SystemMessage(content=MAP_SYSTEM),
            HumanMessage(content=prompt),
        ]

        response = await self._call_llm_with_retry(messages)
        result = [{"chapter": ch, "summary": ""} for ch in chapters]

        # Parse the response text to extract per-chapter summaries
        content = getattr(response, "content", str(response))
        if not isinstance(content, str):
            content = str(content)

        lines = content.strip().split("\n")
        current_chapter_idx = 0
        buffer: list[str] = []
        for line in lines:
            matched = False
            for i, ch in enumerate(chapters.keys()):
                if ch.lower() in line.lower():
                    if buffer and current_chapter_idx < len(result):
                        result[current_chapter_idx]["summary"] = " ".join(
                            buffer
                        ).strip()
                    current_chapter_idx = i
                    buffer = []
                    matched = True
                    break
            if not matched and line.strip():
                buffer.append(line.strip())
        if buffer and current_chapter_idx < len(result):
            result[current_chapter_idx]["summary"] = " ".join(buffer).strip()

        return result
