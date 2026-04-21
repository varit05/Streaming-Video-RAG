"""
Q&A chain — answers questions grounded in retrieved video transcript chunks.
Returns the answer text plus citations (video title + timestamp for each source).
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from llm.factory import get_llm
from vector_store import SearchResult
from .retriever import Retriever


SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on video transcript content.
You are given relevant excerpts from one or more video transcripts, each labeled with \
a source video title and timestamp range.

Rules:
- Answer only based on the provided transcript excerpts.
- If the answer is not found in the excerpts, say so clearly.
- Be concise and direct.
- Reference source numbers (e.g. [1], [2]) when citing specific content.
- Do not invent information beyond what is in the transcripts.
"""

QA_PROMPT_TEMPLATE = """\
Context from video transcripts:
{context}

---

Question: {question}

Please answer based on the transcript excerpts above. If citing a source, reference its number (e.g. [1]).
"""


@dataclass
class QAResult:
    """Result of a Q&A query."""
    question: str
    answer: str
    sources: list[SearchResult]
    video_id: Optional[str] = None

    @property
    def source_citations(self) -> list[dict]:
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
    Retrieves relevant chunks, builds a prompt, and calls the LLM.
    """

    def __init__(self):
        self.retriever = Retriever()

    def ask(
        self,
        question: str,
        video_id: Optional[str] = None,
        top_k: int = None,
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
        results = self.retriever.retrieve(question, top_k=top_k, video_id=video_id)

        if not results:
            return QAResult(
                question=question,
                answer="I couldn't find relevant content in the indexed videos to answer that question.",
                sources=[],
                video_id=video_id,
            )

        # ── Build context and prompt ──────────────────────────────────────────
        context = self.retriever.format_context(results)
        user_message = QA_PROMPT_TEMPLATE.format(context=context, question=question)

        # ── Call LLM ──────────────────────────────────────────────────────────
        llm = get_llm(temperature=0.0)
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        logger.success(f"[QA] Answer generated ({len(answer)} chars)")

        return QAResult(
            question=question,
            answer=answer,
            sources=results,
            video_id=video_id,
        )
