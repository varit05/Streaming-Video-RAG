"""
RAG pipeline orchestrator — chains together multi-query expansion,
cross-encoder reranking, and MMR diversity for high-quality retrieval.

This sits between the Retriever and QAChain to provide:
  1. Multi-query expansion: generate N rephrasings of the question
  2. Cross-encoder reranking: re-rank vector store results with a more accurate model
  3. MMR (Maximal Marginal Relevance): ensure diversity among top results
"""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import settings
from llm.factory import get_llm
from processing.embedder import Embedder
from vector_store import SearchResult, get_vector_store

# numpy is required for MMR — check availability early
try:
    import numpy as np
except ImportError:
    raise ImportError(
        "numpy is required for MMR diversity. Install it with: pip install numpy"
    )

# ── Multi-query expansion prompts ─────────────────────────────────────────────

MULTI_QUERY_SYSTEM = """\
You are a helpful assistant that rephrases user questions to improve \
semantic search over video transcripts. Generate {count} different versions \
of the given question. Each version should:
- Use different wording and sentence structure
- Cover different angles or aspects of the original question
- Be self-contained and clearly answerable
- Return ONLY the rephrased questions, one per line, numbered 1-{count}"""

MULTI_QUERY_USER = """\
Original question: {question}

Generate {count} rephrased versions of this question for video transcript search:"""

# ── Cross-encoder reranker ────────────────────────────────────────────────────

_RERANKER_INSTANCE = None


def _get_reranker():
    """Lazy-load the cross-encoder reranker model."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        from sentence_transformers import CrossEncoder

        model_name = settings.reranker_model
        logger.info(f"[Reranker] Loading model: {model_name}")
        _RERANKER_INSTANCE = CrossEncoder(model_name)
    return _RERANKER_INSTANCE


# ── MMR (Maximal Marginal Relevance) ──────────────────────────────────────────


def _mmr_select(
    results: list[SearchResult],
    query_embedding: list[float],
    top_k: int,
    lambda_param: float = 0.5,
) -> list[SearchResult]:
    """
    Select diverse results using Maximal Marginal Relevance.

    Args:
        results: Candidate results (already scored by reranker or vector store)
        query_embedding: The original query embedding vector
        top_k: Number of results to return
        lambda_param: 0 = only diversity, 1 = only relevance

    Returns:
        Top-k results with MMR diversity applied
    """
    if len(results) <= top_k:
        return results

    query_vec = np.array(query_embedding, dtype=np.float32)
    # Normalize query vector
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm

    # Build candidate embeddings from chunk texts (re-embed for diversity scoring)
    embedder = Embedder()
    candidate_texts = [r.chunk.text for r in results]
    candidate_embs = np.array(
        embedder.embed_query_batch(candidate_texts), dtype=np.float32
    )

    # Normalize candidate embeddings
    norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    candidate_embs = candidate_embs / norms

    n = len(results)
    selected_indices: list[int] = []
    remaining_indices = list(range(n))

    # Relevance scores from the reranker (or vector store similarity)
    relevance = np.array([r.score for r in results], dtype=np.float32)

    for _ in range(min(top_k, n)):
        if not remaining_indices:
            break

        best_score = -1.0
        best_idx = -1

        for idx in remaining_indices:
            # Relevance component
            rel_score = relevance[idx]

            # Diversity component: max similarity to already selected
            if selected_indices:
                sim_to_selected = max(
                    float(np.dot(candidate_embs[idx], candidate_embs[s]))
                    for s in selected_indices
                )
            else:
                sim_to_selected = 0.0

            # MMR score
            mmr_score = lambda_param * rel_score - (1 - lambda_param) * sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [results[i] for i in selected_indices]


# ── Pipeline orchestrator ─────────────────────────────────────────────────────


class RAGPipeline:
    """
    Orchestrates retrieval with multi-query expansion, cross-encoder reranking,
    and MMR diversity.
    """

    def __init__(self) -> None:
        self.store = get_vector_store()
        self.embedder = Embedder()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        video_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Full retrieval pipeline: multi-query → vector search → rerank → MMR.

        Args:
            query: Natural language question
            top_k: Number of final results to return
            video_id: Optional video filter

        Returns:
            List of SearchResult sorted by relevance (highest first)
        """
        top_k = top_k or settings.retrieval_top_k
        logger.info(f"[RAGPipeline] Query='{query[:80]}', top_k={top_k}")

        # ── Step 1: Generate multiple query variants ──────────────────────────
        queries = [query]
        if settings.use_multi_query:
            try:
                extra_queries = self._generate_multi_queries(query)
                queries.extend(extra_queries)
                logger.debug(f"[RAGPipeline] Multi-query: {len(queries)} variants")
            except Exception as e:
                logger.warning(
                    f"[RAGPipeline] Multi-query failed, using original only: {e}"
                )

        # ── Step 2: Retrieve for each query variant ───────────────────────────
        fetch_k = settings.reranker_top_k if settings.use_reranker else top_k
        all_results: dict[str, SearchResult] = {}

        for q in queries:
            query_vector = self.embedder.embed_query(q)
            results = self.store.search(
                query_vector, top_k=fetch_k, filter_video_id=video_id
            )
            for r in results:
                # Deduplicate by chunk_id, keep highest score
                if (
                    r.chunk.chunk_id not in all_results
                    or r.score > all_results[r.chunk.chunk_id].score
                ):
                    all_results[r.chunk.chunk_id] = r

        merged = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
        logger.debug(f"[RAGPipeline] {len(merged)} unique candidates after merge")

        if not merged:
            return []

        # ── Step 3: Cross-encoder reranking ───────────────────────────────────
        if settings.use_reranker:
            try:
                merged = self._rerank(query, merged)
                logger.debug(f"[RAGPipeline] Reranked {len(merged)} results")
            except Exception as e:
                logger.warning(
                    f"[RAGPipeline] Reranker failed, using vector store scores: {e}"
                )

        # ── Step 4: MMR diversity ─────────────────────────────────────────────
        if settings.use_mmr:
            try:
                query_vector = self.embedder.embed_query(query)
                merged = _mmr_select(
                    merged,
                    query_embedding=query_vector,
                    top_k=top_k,
                    lambda_param=settings.mmr_lambda,
                )
                logger.debug(f"[RAGPipeline] MMR selected {len(merged)} results")
            except Exception as e:
                logger.warning(f"[RAGPipeline] MMR failed, using reranked results: {e}")

        # ── Step 5: Trim to final top_k ───────────────────────────────────────
        final = merged[:top_k]
        logger.success(f"[RAGPipeline] Returning {len(final)} results")
        return final

    # ── Private helpers ────────────────────────────────────────────────────────

    def _generate_multi_queries(self, query: str) -> list[str]:
        """Generate alternative phrasings of the query for better recall."""
        llm = get_llm(temperature=0.3)
        count = settings.multi_query_count
        prompt = MULTI_QUERY_USER.format(question=query, count=count)
        messages = [
            SystemMessage(content=MULTI_QUERY_SYSTEM.format(count=count)),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        content = cast(str, response.content)

        # Parse numbered lines: "1. ..." or "1) ..." or "- ..."
        queries: list[str] = []
        for line in content.strip().split("\n"):
            line = line.strip()
            # Remove leading number/prefix
            for prefix in [f"{i}." for i in range(1, count + 1)]:
                if line.startswith(prefix):
                    line = line[len(prefix) :].strip()
                    break
            if line and len(line) > 5:
                queries.append(line)

        return queries[:count]

    def _rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """
        Re-rank results using a cross-encoder model.
        Cross-encoders are much more accurate than bi-encoder cosine similarity.
        """
        reranker = _get_reranker()
        pairs = [(query, r.chunk.text) for r in results]
        scores = reranker.predict(pairs)

        # Attach new scores and sort
        for r, score in zip(results, scores, strict=False):
            # Cross-encoder returns logits; convert to 0-1 score via sigmoid
            import math

            r.score = 1.0 / (1.0 + math.exp(-float(score)))

        results.sort(key=lambda r: r.score, reverse=True)
        return results
