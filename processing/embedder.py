"""
Embedding generation for VideoChunks.

Two modes controlled by EMBEDDING_MODE env var:
  - LOCAL:  sentence-transformers (runs on-device, free)
  - OPENAI: OpenAI text-embedding API (better quality, costs money)
"""

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import EmbeddingMode, settings

from .chunker import VideoChunk


class Embedder:
    """
    Generates dense vector embeddings for VideoChunks.
    Mode is determined by settings.embedding_mode.
    Lazy-loads the model on first use.
    """

    def __init__(self):
        self.mode = settings.embedding_mode
        self._local_model = None
        self._openai_client = None

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension for the active model."""
        if self.mode == EmbeddingMode.LOCAL:
            return 384  # all-MiniLM-L6-v2
        else:
            # text-embedding-3-small = 1536, text-embedding-3-large = 3072
            if "large" in settings.openai_embedding_model:
                return 3072
            return 1536

    def embed_chunks(self, chunks: list[VideoChunk]) -> list[list[float]]:
        """
        Embed a list of chunks in batch.

        Args:
            chunks: VideoChunk objects to embed

        Returns:
            List of float vectors, one per chunk (same order)
        """
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        logger.info(f"[Embedder/{self.mode.value}] Embedding {len(texts)} chunks...")

        vectors = self._embed_local(texts) if self.mode == EmbeddingMode.LOCAL else self._embed_openai(texts)

        logger.success(f"[Embedder] Done — {len(vectors)} vectors, dim={self.dimension}")
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""
        if self.mode == EmbeddingMode.LOCAL:
            return self._embed_local([query])[0]
        else:
            return self._embed_openai([query])[0]

    # ── Local (sentence-transformers) ────────────────────────────────────────

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = self._get_local_model()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            model_name = settings.local_embedding_model
            logger.info(f"[Embedder/local] Loading model: {model_name}")
            self._local_model = SentenceTransformer(model_name)
        return self._local_model

    def _get_openai_client(self):
        if self._openai_client is None:
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set. Required when EMBEDDING_MODE=openai.")
            from openai import OpenAI

            self._openai_client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=60.0,
                max_retries=2,
            )
        return self._openai_client

    # ── OpenAI embeddings ────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        client = self._get_openai_client()
        model = settings.openai_embedding_model

        # OpenAI API allows up to 2048 inputs per call
        batch_size = 512
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Truncate to avoid token limit (8191 tokens for text-embedding-3-*)
            batch = [t[:8000] for t in batch]
            response = client.embeddings.create(model=model, input=batch)
            batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
