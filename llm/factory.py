"""
LLM factory — returns a LangChain chat model based on settings.llm_provider.

Supported providers:
  - openai    → ChatOpenAI (GPT-4o, GPT-4-turbo, etc.)
  - anthropic → ChatAnthropic (Claude models)
  - ollama    → ChatOllama (local Llama 3, Mistral, etc.)

Usage:
    from llm.factory import get_llm
    llm = get_llm()
    response = llm.invoke("Tell me about neural networks")
"""

from functools import lru_cache
from typing import Any
from pydantic import SecretStr

from loguru import logger

from langchain_core.language_models import BaseChatModel
from config import LLMProvider, settings


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Return a LangChain BaseChatModel for the configured provider.
    Cached so the model is only instantiated once per process.

    Args:
        temperature: Sampling temperature (0 = deterministic, good for RAG)
    """
    provider = settings.llm_provider
    logger.info(f"[LLM] Initializing provider={provider.value}")

    if provider == LLMProvider.OPENAI:
        return _openai_llm(temperature)
    elif provider == LLMProvider.ANTHROPIC:
        return _anthropic_llm(temperature)
    elif provider == LLMProvider.OLLAMA:
        return _ollama_llm(temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm_provider_name() -> str:
    """Human-readable name of the active LLM (for display in UI)."""
    p = settings.llm_provider
    if p == LLMProvider.OPENAI:
        return f"OpenAI / {settings.llm_model}"
    elif p == LLMProvider.ANTHROPIC:
        return f"Anthropic / {settings.llm_model}"
    elif p == LLMProvider.OLLAMA:
        return f"Ollama / {settings.ollama_model}"
    return str(p)


# ── Provider implementations ─────────────────────────────────────────────────


def _openai_llm(temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        api_key=SecretStr(settings.openai_api_key) if settings.openai_api_key else None,
        streaming=True,
    )


def _anthropic_llm(temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # Opus 4.7+ removes temperature/top_p/top_k at the API level (returns 400 if sent).
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.anthropic_api_key,
        "streaming": True,
    }
    if not settings.llm_model.startswith("claude-opus-4-7"):
        kwargs["temperature"] = temperature
    return ChatAnthropic(**kwargs)


def _ollama_llm(temperature: float) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        temperature=temperature,
        base_url=settings.ollama_base_url,
    )
