"""
Pydantic request/response models for the FastAPI API.
"""

from pydantic import BaseModel, Field

# ── Ingest ───────────────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    source: str = Field(
        ..., description="YouTube URL, local file path, stream URL, or video API URL"
    )
    source_type: str | None = Field(
        None,
        description="Hint for source type: 'youtube' | 'local_file' | 'live_stream' | 'video_api'. Auto-detected if omitted.",
    )
    language: str | None = Field(
        None,
        description="ISO 639-1 language code (e.g. 'en'). Auto-detected if omitted.",
    )
    platform: str | None = Field(
        None, description="For video_api source: 'vimeo' | 'twitch'"
    )
    api_credentials: dict[str, str] | None = Field(
        None, description="API credentials for video_api source"
    )


class IngestResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    video_id: str | None
    status: str
    progress_message: str | None
    error_message: str | None
    created_at: str | None
    completed_at: str | None


# ── Videos ───────────────────────────────────────────────────────────────────


class VideoResponse(BaseModel):
    id: str
    title: str
    source_url: str
    source_type: str
    duration_seconds: float | None
    language: str | None
    chunk_count: int
    status: str
    created_at: str | None
    indexed_at: str | None


class VideoListResponse(BaseModel):
    total: int
    videos: list[VideoResponse]


# ── Q&A ──────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(
        ..., description="Natural language question to answer from video content"
    )
    video_id: str | None = Field(
        None, description="If set, restrict search to this video"
    )
    top_k: int | None = Field(
        None, description="Number of chunks to retrieve (default: from settings)"
    )


class SourceCitation(BaseModel):
    index: int
    title: str
    video_id: str
    start_ts: str
    end_ts: str
    source_url: str
    deep_link: str | None
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceCitation]


# ── Search ───────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., description="Semantic search query")
    video_id: str | None = Field(
        None, description="If set, search only within this video"
    )
    top_k: int = Field(10, description="Maximum number of results")
    min_score: float = Field(0.0, description="Minimum similarity score (0-1)")


class SearchResultItem(BaseModel):
    rank: int
    score: float
    video_id: str
    title: str
    source_url: str
    start_time: float
    end_time: float
    start_ts: str
    end_ts: str
    chapter: str | None
    text: str
    deep_link: str | None


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResultItem]


# ── Summarize ────────────────────────────────────────────────────────────────


class SummarizeRequest(BaseModel):
    video_id: str
    include_chapters: bool = Field(
        True, description="Also generate per-chapter summaries if available"
    )


class ChapterSummary(BaseModel):
    chapter: str
    summary: str


class SummarizeResponse(BaseModel):
    video_id: str
    title: str
    overall_summary: str
    chapter_summaries: list[ChapterSummary] | None
    chunk_count: int
