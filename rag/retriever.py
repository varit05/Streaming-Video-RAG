"""
Retriever — wraps the vector store + embedder into a clean retrieval interface.
Used by QAChain, Summarizer, and SearchEngine.
"""

from typing import Optional

from loguru import logger

from config import settings
from processing.embedder import Embedder
from vector_store import SearchResult, get_vector_store

_retriever_instance = None


def get_retriever():
    """Return a cached Retriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance


class Retriever:
    """
    Retrieves the most relevant VideoChunks for a query.
    Handles embedding the query and calling the vector store.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.store = get_vector_store()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        video_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Find the most relevant chunks for `query`.

        Args:
            query: Natural language question or search string
            top_k: Number of results to return (defaults to settings.retrieval_top_k)
            video_id: If set, search only within this video

        Returns:
            List of SearchResult sorted by relevance (highest first)
        """
        top_k = top_k or settings.retrieval_top_k
        logger.debug(f"[Retriever] Query='{query[:60]}', top_k={top_k}, video_id={video_id}")

        try:
            query_vector = self.embedder.embed_query(query)
            results = self.store.search(query_vector, top_k=top_k, filter_video_id=video_id)
        except Exception as e:
            import traceback

            error_stack = traceback.format_exc()
            logger.error(f"[Retriever] Retrieval failed with error: {str(e)}")
            logger.error(f"[Retriever] Full error stack:\n{error_stack}")

            # Preserve original exception context
            raise RuntimeError(
                f"Vector store retrieval failed. Error: {str(e)}. "
                f"Type: {type(e).__name__}. Check logs for full stack trace."
            ) from e

        logger.debug(f"[Retriever] Retrieved {len(results)} results successfully")

        # Validate results before returning
        for idx, result in enumerate(results):
            if not hasattr(result, "chunk") or not hasattr(result, "score"):
                logger.warning(f"[Retriever] Invalid result object at index {idx}: {result}")

        return results

    def format_context(self, results: list[SearchResult]) -> str:
        """
        Format retrieved chunks into a context block for the LLM.
        Each chunk is prefixed with its source video title and timestamp.
        """
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f'[{i}] "{r.chunk.title}" @ {r.chunk.start_ts}–{r.chunk.end_ts}\n{r.chunk.text}')
        return "\n\n---\n\n".join(parts)
