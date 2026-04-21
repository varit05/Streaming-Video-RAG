"""
/query and /summarize routes — RAG-powered Q&A and summarization.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import (
    QueryRequest, QueryResponse, SourceCitation,
    SummarizeRequest, SummarizeResponse, ChapterSummary,
)
from rag import QAChain, Summarizer

router = APIRouter(tags=["RAG"])


@router.post("/query", response_model=QueryResponse)
def query_videos(request: QueryRequest):
    """
    Ask a question and get an answer grounded in indexed video content.
    Returns the answer plus timestamped source citations.
    """
    logger.info(f"[API/query] '{request.question[:60]}'")

    chain = QAChain()
    result = chain.ask(
        question=request.question,
        video_id=request.video_id,
        top_k=request.top_k,
    )

    sources = [SourceCitation(**s) for s in result.source_citations]

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=sources,
    )


@router.post("/summarize", response_model=SummarizeResponse)
def summarize_video(request: SummarizeRequest):
    """
    Generate a summary of an indexed video.
    Uses map-reduce to handle long videos gracefully.
    """
    logger.info(f"[API/summarize] video_id={request.video_id}")

    summarizer = Summarizer()
    result = summarizer.summarize(
        video_id=request.video_id,
        include_chapters=request.include_chapters,
    )

    chapter_summaries = None
    if result.chapter_summaries:
        chapter_summaries = [
            ChapterSummary(chapter=c["chapter"], summary=c["summary"])
            for c in result.chapter_summaries
        ]

    return SummarizeResponse(
        video_id=result.video_id,
        title=result.title,
        overall_summary=result.overall_summary,
        chapter_summaries=chapter_summaries,
        chunk_count=result.chunk_count,
    )
