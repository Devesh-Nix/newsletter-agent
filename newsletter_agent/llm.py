"""Provider-agnostic chat-model factory built on LangChain's `init_chat_model`."""

from __future__ import annotations

import os
from typing import Any, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from newsletter_agent.config import PROVIDER_KEY_ENV, Settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMConfigurationError(RuntimeError):
    """The selected provider cannot be used (missing key or package)."""


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Instantiate the configured chat model with sensible production defaults."""
    provider = settings.resolved_provider
    model = settings.resolved_model

    key_vars = PROVIDER_KEY_ENV.get(provider, ())
    if key_vars and not any(os.getenv(var) for var in key_vars):
        raise LLMConfigurationError(
            f"Provider '{provider}' needs an API key: set {' or '.join(key_vars)} "
            "(in your shell or a .env file), or choose another provider."
        )

    kwargs: dict[str, Any] = {}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    if provider == "ollama":
        kwargs |= {"base_url": settings.ollama_base_url, "num_predict": settings.llm_max_tokens}
    else:
        kwargs |= {
            "max_tokens": settings.llm_max_tokens,
            "timeout": settings.llm_timeout,
            "max_retries": 3,
        }
    if provider == "google_genai" and not os.getenv("GOOGLE_API_KEY"):
        kwargs["google_api_key"] = os.getenv("GEMINI_API_KEY")

    try:
        return init_chat_model(model, model_provider=provider, **kwargs)
    except ImportError as exc:  # provider integration package not installed
        raise LLMConfigurationError(str(exc)) from exc


def structured_output(llm: BaseChatModel, schema: type[SchemaT]) -> Runnable[Any, SchemaT]:
    """`llm.with_structured_output(schema)` with the most reliable method per provider.

    Anthropic models with (adaptive) thinking reject forced tool choice, which is how
    LangChain implements structured output by default, so Claude uses the API's native
    JSON-schema output instead.
    """
    kwargs: dict[str, Any] = {}
    if type(llm).__name__ == "ChatAnthropic":
        kwargs["method"] = "json_schema"
    return llm.with_structured_output(schema, **kwargs).with_retry(stop_after_attempt=2)


def describe_model(llm: BaseChatModel) -> str:
    """Human-readable 'provider:model' label for logs and the UI."""
    name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or "unknown"
    return f"{llm._llm_type}:{name}"
