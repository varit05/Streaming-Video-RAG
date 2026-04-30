"""
/search route — semantic search across indexed video content.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import SearchRequest, SearchResponse, SearchResultItem
from rag import SearchEngine

router = APIRouter(tags=["Search"])


@router.post("/search")
def search_videos(request: SearchRequest) -> SearchResponse:
    """
    Semantic search across all indexed videos.
    Returns ranked video segments with timestamps.
    """
    logger.info(f"[API/search] '{request.query[:60]}'")

    try:
        engine = SearchEngine()
        result = engine.search(
            query=request.query,
            top_k=request.top_k,
            video_id=request.video_id,
            min_score=request.min_score,
        )
    except RuntimeError as e:
        logger.error(f"[API/search] Retrieval error: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e

    return SearchResponse(
        query=result.query,
        total_results=result.total_results,
        results=[SearchResultItem.model_validate(r) for r in result.results],
    )
