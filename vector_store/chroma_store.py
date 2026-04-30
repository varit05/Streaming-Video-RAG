"""
Chroma vector store — recommended for local development.
No server needed; persists to disk automatically.
"""

from collections.abc import Sequence
from typing import Any, cast

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.types import Where
from loguru import logger

from config import settings
from processing.chunker import VideoChunk

from .base import BaseVectorStore, SearchResult

COLLECTION_NAME = "video_rag"


class ChromaVectorStore(BaseVectorStore):
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[Chroma] Connected — collection '{COLLECTION_NAME}', {self._collection.count()} chunks indexed")

    def add_chunks(self, chunks: list[VideoChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_metadata_dict() for c in chunks]

        # Chroma upserts by ID, so re-indexing a video is safe
        self._collection.upsert(
            ids=cast(Sequence[str | None], ids),
            embeddings=cast(Sequence[Sequence[float | int] | None], embeddings),
            documents=cast(Sequence[str | None], documents),
            metadatas=cast(Sequence[dict[str, Any] | None], metadatas),
        )
        logger.info(f"[Chroma] Upserted {len(chunks)} chunks")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_video_id: str | None = None,
    ) -> list[SearchResult]:
        where = cast(Where | None, {"video_id": filter_video_id} if filter_video_id else None)

        results = self._collection.query(
            query_embeddings=cast(Sequence[Sequence[float | int] | None], [query_vector]),
            n_results=min(top_k, self._collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        ids = results["ids"][0] if results["ids"] else []
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i, (doc_id, text, meta, dist) in enumerate(zip(ids, docs, metas, distances, strict=False)):
            if not meta or "video_id" not in meta:
                logger.warning(f"[Chroma] Skipping result {doc_id!r} — missing metadata")
                continue
            # Chroma returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score: 1 - dist/2
            score = max(0.0, 1.0 - dist / 2.0)

            chunk = VideoChunk(
                chunk_id=str(meta.get("chunk_id", doc_id)),
                video_id=str(meta["video_id"]),
                text=text,
                start_time=float(cast(float, meta.get("start_time", 0.0))),
                end_time=float(cast(float, meta.get("end_time", 0.0))),
                segment_index=int(cast(int, meta.get("segment_index", i))),
                title=str(meta.get("title", "")),
                source_url=str(meta.get("source_url", "")),
                chapter=str(meta.get("chapter")) if meta.get("chapter") else None,
                language=str(meta.get("language", "en")),
            )
            search_results.append(SearchResult(chunk=chunk, score=score))

        return sorted(search_results, key=lambda r: r.score, reverse=True)

    def delete_video(self, video_id: str) -> int:
        existing = self._collection.get(where={"video_id": video_id})
        ids_to_delete = existing["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        logger.info(f"[Chroma] Deleted {len(ids_to_delete)} chunks for video {video_id}")
        return len(ids_to_delete)

    def count(self, video_id: str | None = None) -> int:
        if video_id:
            result = self._collection.get(where={"video_id": video_id})
            return len(result["ids"])
        return self._collection.count()
