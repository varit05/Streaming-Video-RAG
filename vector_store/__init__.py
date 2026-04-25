from config import VectorStoreType, settings

from .base import BaseVectorStore, SearchResult
from .chroma_store import ChromaVectorStore
from .qdrant_store import QdrantVectorStore

_vector_store_instance: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    """Factory: return the configured vector store (cached singleton)."""
    global _vector_store_instance
    if _vector_store_instance is None:
        if settings.vector_store_type == VectorStoreType.CHROMA:
            _vector_store_instance = ChromaVectorStore()
        elif settings.vector_store_type == VectorStoreType.QDRANT:
            _vector_store_instance = QdrantVectorStore()
        else:
            raise ValueError(f"Unknown vector store type: {settings.vector_store_type}")
    return _vector_store_instance


__all__ = ["BaseVectorStore", "SearchResult", "ChromaVectorStore", "QdrantVectorStore", "get_vector_store"]
