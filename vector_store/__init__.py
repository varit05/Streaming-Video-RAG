from .base import BaseVectorStore, SearchResult
from .chroma_store import ChromaVectorStore
from .qdrant_store import QdrantVectorStore
from config import settings, VectorStoreType


def get_vector_store() -> BaseVectorStore:
    """Factory: return the configured vector store."""
    if settings.vector_store_type == VectorStoreType.CHROMA:
        return ChromaVectorStore()
    elif settings.vector_store_type == VectorStoreType.QDRANT:
        return QdrantVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {settings.vector_store_type}")


__all__ = ["BaseVectorStore", "SearchResult", "ChromaVectorStore", "QdrantVectorStore", "get_vector_store"]
