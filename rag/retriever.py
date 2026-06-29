"""
Retriever — wraps the vector store + embedder into a clean retrieval interface.
Used by QAChain, Summarizer, and SearchEngine.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import settings
from llm.factory import get_llm
from processing.embedder import Embedder
from vector_store import SearchResult, get_vector_store

_retriever_instance = None

HYDE_SYSTEM_PROMPT = "You are an expert video transcript generator."

HYDE_USER_PROMPT = """\
Given a question, write a single paragraph that would likely appear in a video transcript answering that question.
The paragraph should be in the first person ("I", "we") as if spoken by a presenter.
Focus on being factual and informative. Do not include any meta-talk like "Here is a transcript excerpt".
Just provide the content of the transcript itself.

Question: {query}

Hypothetical Transcript Excerpt:"""


def get_retriever() -> "Retriever":
    """Return a cached Retriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance


class Retriever:
    """
    Retrieves the most relevant VideoChunks for a query.
    Handles embedding the query and calling the vector store.
    Supports HyDE (Hypothetical Document Embeddings) for improved retrieval.
    """

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.store = get_vector_store()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        video_id: str | None = None,
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
        logger.debug(
            f"[Retriever] Query='{query[:60]}', top_k={top_k}, video_id={video_id}, use_hyde={settings.use_hyde}"
        )

        try:
            # ── HyDE Transformation ───────────────────────────────────────────
            search_query = query
            if settings.use_hyde:
                try:
                    search_query = self._generate_hypothetical_document(query)
                    logger.debug(f"[Retriever] HyDE generated: {search_query[:100]}...")
                except Exception as e:
                    logger.warning(
                        f"[Retriever] HyDE failed, falling back to original query: {e}"
                    )

            # ── Embedding & Search ───────────────────────────────────────────
            query_vector = self.embedder.embed_query(search_query)
            results = self.store.search(
                query_vector, top_k=top_k, filter_video_id=video_id
            )
        except Exception as e:
            import traceback

            error_stack = traceback.format_exc()
            logger.error(f"[Retriever] Retrieval failed with error: {e!s}")
            logger.error(f"[Retriever] Full error stack:\n{error_stack}")

            # Preserve original exception context
            raise RuntimeError(
                f"Vector store retrieval failed. Error: {e!s}. "
                f"Type: {type(e).__name__}. Check logs for full stack trace."
            ) from e

        logger.debug(f"[Retriever] Retrieved {len(results)} results successfully")

        # Validate results before returning
        for idx, result in enumerate(results):
            if not hasattr(result, "chunk") or not hasattr(result, "score"):
                logger.warning(
                    f"[Retriever] Invalid result object at index {idx}: {result}"
                )

        return results

    def format_context(self, results: list[SearchResult]) -> str:
        """
        Format retrieved chunks into a context block for the LLM.
        Each chunk is prefixed with its source video title and timestamp.
        """
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f'[{i}] "{r.chunk.title}" @ {r.chunk.start_ts}-{r.chunk.end_ts}\n{r.chunk.text}'
            )
        return "\n\n---\n\n".join(parts)

    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical transcript snippet that would answer the query.
        This is the 'HyDE' technique.
        """
        llm = get_llm(temperature=0.0)
        prompt = HYDE_USER_PROMPT.format(query=query)
        messages = [
            SystemMessage(content=HYDE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        return str(content).strip()
