"""
/query and /summarize routes — RAG-powered Q&A and summarization.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session

from api.models import (
    ChapterSummary,
    QueryRequest,
    QueryResponse,
    SourceCitation,
    SummarizeRequest,
    SummarizeResponse,
)
from rag import QAChain, Summarizer
from storage.database import Video, get_db

router = APIRouter(tags=["RAG"])


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        503: {"description": "LLM service unavailable (e.g. Ollama not running)"},
    },
)
def query_videos(request: QueryRequest) -> QueryResponse:
    """
    Ask a question and get an answer grounded in indexed video content.
    Returns the answer plus timestamped source citations.
    """
    logger.info(f"[API/query] '{request.question[:60]}'")

    chain = QAChain()
    try:
        result = chain.ask(
            question=request.question,
            video_id=request.video_id,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.error(f"[API/query] LLM invocation failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot reach the LLM backend: {exc}. "
                "Check that Ollama is running (`ollama serve`) "
                "or verify your LLM_PROVIDER and API key configuration."
            ),
        ) from exc

    sources = [SourceCitation.model_validate(s) for s in result.source_citations]

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=sources,
    )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={
        404: {"description": "Video not found"},
        503: {"description": "LLM service unavailable (e.g. Ollama not running)"},
    },
)
def summarize_video(
    request: SummarizeRequest, db: Annotated[Session, Depends(get_db)]
) -> SummarizeResponse:
    """
    Generate a summary of an indexed video.
    Uses map-reduce to handle long videos gracefully.
    """
    logger.info(f"[API/summarize] video_id={request.video_id}")

    video = db.query(Video).filter(Video.id == request.video_id).first()
    if not video:
        raise HTTPException(
            status_code=404, detail=f"Video {request.video_id!r} not found"
        )

    summarizer = Summarizer()
    try:
        result = summarizer.summarize(
            video_id=request.video_id,
            title=str(video.title),
            include_chapters=request.include_chapters,
        )
    except Exception as exc:
        # Catch connection errors (e.g. Ollama not running),
        # timeouts, and any other LLM invocation failures.
        logger.error(f"[API/summarize] LLM invocation failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot reach the LLM backend: {exc}. "
                "Check that Ollama is running (`ollama serve`) "
                "or verify your LLM_PROVIDER and API key configuration."
            ),
        ) from exc

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