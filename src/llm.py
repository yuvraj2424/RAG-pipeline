"""
LLM and embedding factory.

LLM selection (in priority order):
  1. Azure AI Foundry (Claude) — when ANTHROPIC_FOUNDRY_API_KEY is set
     MODEL_TIER controls which deployment: "sonnet" | "haiku" | "opus"
  2. OpenAI (GPT-4o) — fallback when Foundry key is absent

Embedding selection (in priority order):
  1. Azure OpenAI — when AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT are set
  2. OpenAI — when OPENAI_API_KEY is set
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.config import (
    # Foundry / Claude
    ANTHROPIC_FOUNDRY_API_KEY,
    ANTHROPIC_FOUNDRY_BASE_URL,
    SONNET_MODEL,
    HAIKU_MODEL,
    OPUS_MODEL,
    MODEL_TIER,
    # OpenAI fallback
    LLM_MODEL,
    # Embeddings
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_EMBEDDING_DEPLOYMENT,
    EMBEDDING_MODEL,
)

# Maps tier name → deployment / model name
_TIER_TO_MODEL: dict[str, str] = {
    "sonnet": SONNET_MODEL,
    "haiku":  HAIKU_MODEL,
    "opus":   OPUS_MODEL,
}


def get_llm(tier: str | None = None) -> BaseChatModel:
    """
    Return a chat LLM for the requested tier.

    Args:
        tier: "sonnet" | "haiku" | "opus" — defaults to MODEL_TIER env var.

    Returns:
        ChatAnthropic (Azure AI Foundry) if ANTHROPIC_FOUNDRY_API_KEY is set,
        otherwise ChatOpenAI (standard OpenAI).
    """
    resolved_tier = tier or MODEL_TIER
    model_name = _TIER_TO_MODEL.get(resolved_tier, SONNET_MODEL)

    if ANTHROPIC_FOUNDRY_API_KEY:
        from langchain_anthropic import ChatAnthropic
        print(f"[LLM] Azure AI Foundry → {model_name} (tier={resolved_tier})")
        return ChatAnthropic(
            model=model_name,
            temperature=0,
            anthropic_api_key=ANTHROPIC_FOUNDRY_API_KEY,
            anthropic_api_url=ANTHROPIC_FOUNDRY_BASE_URL,
        )

    from langchain_openai import ChatOpenAI
    print(f"[LLM] OpenAI → {LLM_MODEL}")
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


def get_embeddings() -> Embeddings:
    """
    Return an embedding model.

    Returns:
        AzureOpenAIEmbeddings if AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT are set,
        otherwise standard OpenAIEmbeddings.
    """
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        from langchain_openai import AzureOpenAIEmbeddings
        print(f"[Embeddings] Azure OpenAI → {AZURE_EMBEDDING_DEPLOYMENT}")
        return AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )

    from langchain_openai import OpenAIEmbeddings
    print(f"[Embeddings] OpenAI → {EMBEDDING_MODEL}")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# Transient errors from Azure AI Foundry that warrant a retry
try:
    from anthropic import OverloadedError as _OverloadedError
    from anthropic import InternalServerError as _InternalServerError
    _RETRY_EXCEPTIONS = (_OverloadedError, _InternalServerError)
except ImportError:
    from anthropic import APIStatusError as _APIStatusError
    _RETRY_EXCEPTIONS = (_APIStatusError,)

# Module-level instances used throughout the codebase.
# `llm` is the raw instance — use it when you need .with_structured_output().
# `llm_with_retry` wraps invoke() with exponential backoff for transient
# 529 "Overloaded" errors from Azure AI Foundry (up to 3 attempts).
llm = get_llm()
llm_with_retry = llm.with_retry(
    retry_if_exception_type=_RETRY_EXCEPTIONS,
    wait_exponential_jitter=True,
    stop_after_attempt=3,
)
embeddings = get_embeddings()
