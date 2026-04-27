"""
/videos routes — list and manage indexed videos.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.models import VideoListResponse, VideoResponse
from storage.database import Video, get_db
from vector_store import get_vector_store

router = APIRouter(prefix="/videos", tags=["Videos"])

DbSession = Annotated[Session, Depends(get_db)]


@router.get("", response_model=VideoListResponse)
def list_videos(
    db: DbSession,
    status: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """List all indexed videos, optionally filtered by status."""
    query = db.query(Video)
    if status:
        query = query.filter(Video.status == status)
    total = query.count()
    videos = query.order_by(Video.created_at.desc()).offset(offset).limit(limit).all()

    return VideoListResponse(
        total=total,
        videos=[VideoResponse(**v.to_dict()) for v in videos],
    )


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(video_id: str, db: DbSession):
    """Get metadata for a specific video."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return VideoResponse(**video.to_dict())


@router.delete("/{video_id}")
def delete_video(video_id: str, db: DbSession):
    """
    Remove a video from the database and delete all its embeddings
    from the vector store.
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    # Remove from vector store
    store = get_vector_store()
    deleted_chunks = store.delete_video(video_id)

    # Remove from DB
    db.delete(video)
    db.commit()

    return {
        "message": f"Video {video_id} deleted",
        "chunks_removed": deleted_chunks,
    }
