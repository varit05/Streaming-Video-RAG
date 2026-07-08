"""
Q&A chain — answers questions grounded in retrieved video transcript chunks.
Returns the answer text plus citations (video title + timestamp for each source).

Uses the RAGPipeline (multi-query + reranker + MMR) for high-quality retrieval.
"""

from dataclasses import dataclass
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from llm.factory import get_llm
from vector_store import SearchResult

from .pipeline import RAGPipeline

SYSTEM_PROMPT = """\
You are a helpful, precise assistant that answers questions based on video transcript content.
You are given relevant excerpts from one or more video transcripts, each labeled with \
a source video title and timestamp range.

## Rules:
1. Answer ONLY based on the provided transcript excerpts. Do NOT use outside knowledge.
2. If the answer is not fully found in the excerpts, say so clearly and explain what IS available.
3. Be concise but thorough. Cover all relevant information from the transcripts.
4. Cite sources using bracketed numbers: [1], [2], etc. At the end of your answer, \
list the source references with their titles and timestamps.
5. If the question asks for a list, use bullet points. If it asks for an explanation, \
use clear paragraphs.
6. If the question is ambiguous, acknowledge the ambiguity and answer the most likely interpretation.
7. Do NOT invent information beyond what is in the transcripts.

## Output format:
- Answer: [your answer here with [1], [2] citations]
- Sources: [1] "Video Title" @ 00:01:23-00:02:45
"""

QA_PROMPT_TEMPLATE = """\
Context from video transcripts:
{context}

---

Question: {question}

Please answer based on the transcript excerpts above. Cite sources with [1], [2], etc."""


@dataclass
class QAResult:
    """Result of a Q&A query."""

    question: str
    answer: str
    sources: list[SearchResult]
    video_id: str | None = None

    @property
    def source_citations(self) -> list[dict[str, object]]:
        return [
            {
                "index": i + 1,
                "title": r.chunk.title,
                "video_id": r.chunk.video_id,
                "start_ts": r.chunk.start_ts,
                "end_ts": r.chunk.end_ts,
                "source_url": r.chunk.source_url,
                "deep_link": r.chunk.youtube_deep_link(),
                "score": round(r.score, 3),
            }
            for i, r in enumerate(self.sources)
        ]


class QAChain:
    """
    Retrieval-augmented Q&A over indexed video content.
    Uses RAGPipeline (multi-query + reranker + MMR) for retrieval and calls the LLM.
    """

    def __init__(self) -> None:
        self.pipeline = RAGPipeline()

    def ask(
        self,
        question: str,
        video_id: str | None = None,
        top_k: int | None = None,
    ) -> QAResult:
        """
        Answer a question using retrieved video transcript context.

        Args:
            question: The user's question
            video_id: If set, search only within this video
            top_k: Number of chunks to retrieve

        Returns:
            QAResult with answer text and source citations
        """
        logger.info(f"[QA] Question: '{question[:80]}'")

        # ── Retrieve relevant chunks ──────────────────────────────────────────
        results = self.pipeline.retrieve(question, top_k=top_k, video_id=video_id)

        if not results:
            return QAResult(
                question=question,
                answer="I couldn't find relevant content in the indexed videos to answer that question.",
                sources=[],
                video_id=video_id,
            )

        # ── Build context and prompt ──────────────────────────────────────────
        context = self._format_context(results)
        user_message = QA_PROMPT_TEMPLATE.format(context=context, question=question)

        # ── Call LLM ──────────────────────────────────────────────────────────
        llm = get_llm(temperature=0.0)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = cast(str, answer)

        logger.success(f"[QA] Answer generated ({len(answer)} chars)")

        return QAResult(
            question=question,
            answer=answer,
            sources=results,
            video_id=video_id,
        )

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """
        Format retrieved chunks into a context block for the LLM.
        Each chunk is prefixed with its source video title and timestamp.
        """
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f'[{i}] "{r.chunk.title}" @ {r.chunk.start_ts}-{r.chunk.end_ts}\n{r.chunk.text}'
            )
        return "\n\n---\n\n".join(parts)