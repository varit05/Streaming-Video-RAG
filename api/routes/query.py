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


@router.post("/query", response_model=QueryResponse)
def query_videos(request: QueryRequest) -> QueryResponse:
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

    sources = [SourceCitation.model_validate(s) for s in result.source_citations]

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=sources,
    )


@router.post("/summarize", response_model=SummarizeResponse, responses={404: {"description": "Video not found"}})
def summarize_video(request: SummarizeRequest, db: Annotated[Session, Depends(get_db)]) -> SummarizeResponse:
    """
    Generate a summary of an indexed video.
    Uses map-reduce to handle long videos gracefully.
    """
    logger.info(f"[API/summarize] video_id={request.video_id}")

    video = db.query(Video).filter(Video.id == request.video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {request.video_id!r} not found")

    summarizer = Summarizer()
    result = summarizer.summarize(
        video_id=request.video_id,
        title=str(video.title),
        include_chapters=request.include_chapters,
    )

    chapter_summaries = None
    if result.chapter_summaries:
        chapter_summaries = [
            ChapterSummary(chapter=c["chapter"], summary=c["summary"]) for c in result.chapter_summaries
        ]

    return SummarizeResponse(
        video_id=result.video_id,
        title=result.title,
        overall_summary=result.overall_summary,
        chapter_summaries=chapter_summaries,
        chunk_count=result.chunk_count,
    )
