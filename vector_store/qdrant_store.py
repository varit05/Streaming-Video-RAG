"""
Qdrant vector store — recommended for production.
Requires a running Qdrant server (see docker-compose.yml).
"""

from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import settings
from processing.chunker import VideoChunk
from processing.embedder import Embedder

from .base import BaseVectorStore, SearchResult


class QdrantVectorStore(BaseVectorStore):
    def __init__(self):
        client_kwargs = {
            "url": settings.qdrant_url,
            "timeout": 30,
        }

        if settings.qdrant_api_key:
            client_kwargs["api_key"] = settings.qdrant_api_key

        if settings.qdrant_https:
            client_kwargs["https"] = True

        self._client = QdrantClient(**client_kwargs)
        self._collection = settings.qdrant_collection
        # Infer dimension from embedder
        self._dim = Embedder().dimension
        self._ensure_collection()
        logger.info(f"[Qdrant] Connected — collection '{self._collection}', dim={self._dim}")

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist yet."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            logger.info(f"[Qdrant] Created collection '{self._collection}'")

    def add_chunks(self, chunks: list[VideoChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        points = [
            PointStruct(
                id=self._chunk_id_to_int(c.chunk_id),
                vector=emb,
                payload={**c.to_metadata_dict(), "text": c.text},
            )
            for c, emb in zip(chunks, embeddings)
        ]

        self._client.upsert(collection_name=self._collection, points=points)
        logger.info(f"[Qdrant] Upserted {len(points)} chunks")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_video_id: Optional[str] = None,
    ) -> list[SearchResult]:
        try:
            query_filter = None
            if filter_video_id:
                query_filter = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=filter_video_id))])

            logger.debug(f"[Qdrant] Executing search with top_k={top_k}, filter={filter_video_id}")

            hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

            results = []
            for hit in hits:
                try:
                    payload = hit.payload or {}
                    chunk = VideoChunk(
                        chunk_id=payload.get("chunk_id", str(hit.id)),
                        video_id=payload.get("video_id", ""),
                        text=payload.get("text", ""),
                        start_time=float(payload.get("start_time", 0)),
                        end_time=float(payload.get("end_time", 0)),
                        segment_index=int(payload.get("segment_index", 0)),
                        title=payload.get("title", ""),
                        source_url=payload.get("source_url", ""),
                        chapter=payload.get("chapter") or None,
                        language=payload.get("language", "en"),
                    )
                    results.append(SearchResult(chunk=chunk, score=float(hit.score)))
                except Exception as hit_error:
                    logger.warning(f"[Qdrant] Failed to parse hit id={hit.id}: {hit_error}")
                    continue

            logger.debug(f"[Qdrant] Search completed successfully, found {len(results)} valid results")
            return results

        except Exception as e:
            logger.error(f"[Qdrant] Search failed: {str(e)}")
            logger.error(f"[Qdrant] Collection: {self._collection}, top_k: {top_k}, filter: {filter_video_id}")
            raise

    def delete_video(self, video_id: str) -> int:
        # Count before delete
        count_before = self.count(video_id)
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]),
        )
        logger.info(f"[Qdrant] Deleted ~{count_before} chunks for video {video_id}")
        return count_before

    def count(self, video_id: Optional[str] = None) -> int:
        if video_id:
            result = self._client.count(
                collection_name=self._collection,
                count_filter=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]),
                exact=True,
            )
            return result.count
        return self._client.count(collection_name=self._collection, exact=True).count

    @staticmethod
    def _chunk_id_to_int(chunk_id: str) -> int:
        """
        Qdrant point IDs must be unsigned integers or UUIDs.
        We use a stable hash of the chunk_id string.
        """
        import hashlib

        return int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (2**63)
