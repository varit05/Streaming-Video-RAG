"""
Central configuration for the Streaming Video-RAG system.
All settings are driven by environment variables (see .env.example).
"""

import enum
from pathlib import Path

from pydantic_settings import BaseSettings


class LLMProvider(enum.StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class WhisperMode(enum.StrEnum):
    LOCAL = "local"
    OPENAI_API = "openai_api"


class EmbeddingMode(enum.StrEnum):
    LOCAL = "local"
    OPENAI = "openai"


class VectorStoreType(enum.StrEnum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o"  # used for openai + anthropic
    ollama_model: str = "llama3"  # used when provider=ollama
    ollama_base_url: str = "http://localhost:11434"

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Whisper transcription ─────────────────────────────────────────────
    whisper_mode: WhisperMode = WhisperMode.LOCAL
    # Local model size: tiny | base | small | medium | large | large-v2
    whisper_model_size: str = "base"

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_mode: EmbeddingMode = EmbeddingMode.LOCAL
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Vector store ──────────────────────────────────────────────────────
    vector_store_type: VectorStoreType = VectorStoreType.CHROMA
    chroma_persist_dir: str = "./data/chroma"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "video_rag"
    qdrant_https: bool = False

    # ── Storage ───────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/video_rag.db"
    data_dir: str = "./data"
    audio_dir: str = "./data/audio"
    transcript_dir: str = "./data/transcripts"

    # ── API ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    api_http2: bool = True
    api_ssl_certfile: str = "./certs/localhost.pem"
    api_ssl_keyfile: str = "./certs/localhost-key.pem"

    # ── RAG tuning ────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    chunk_duration_seconds: int = 60
    chunk_overlap_seconds: int = 15
    use_hyde: bool = False  # HyDE query transformation (adds 1 LLM call per query)

    # ── Re-ranker ─────────────────────────────────────────────────────────
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 10  # Fetch this many from vector store before reranking

    # ── MMR diversity ─────────────────────────────────────────────────────
    use_mmr: bool = True
    mmr_lambda: float = 0.5  # 0 = only diversity, 1 = only relevance

    # ── Multi-query expansion ─────────────────────────────────────────────
    use_multi_query: bool = False  # adds 1 LLM call + N extra searches per query
    multi_query_count: int = 3

    # ── Live stream ───────────────────────────────────────────────────────
    live_stream_segment_seconds: int = 60  # capture window per segment

    # ── UI ────────────────────────────────────────────────────────────────
    ui_api_base_url: str = "http://localhost:8000"
    ui_request_timeout: int = 120

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self) -> None:
        """Create all required data directories if they don't exist."""
        for d in [
            self.data_dir,
            self.audio_dir,
            self.transcript_dir,
            self.chroma_persist_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
