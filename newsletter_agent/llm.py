"""Provider-agnostic chat-model factory built on LangChain's `init_chat_model`,
plus the resilience layer every model call goes through: pacing, retries and
human-readable errors."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from typing import Any, TypeVar

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from newsletter_agent.config import PROVIDER_KEY_ENV, Settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)

#: Failures worth retrying: provider hiccups and malformed structured output.
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    ModelRateLimitError,
    ModelAPIError,
    ModelConnectionError,
    ModelTimeoutError,
    OutputParserException,
    ValidationError,
)
#: Exponential backoff (seconds). Quotas are usually per minute, so waits grow long.
RETRY_BACKOFF = {"initial": 5.0, "max": 60.0}
RETRY_ATTEMPTS = 4


class LLMConfigurationError(RuntimeError):
    """The selected provider cannot be used (missing key or package)."""


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------


class MinIntervalRateLimiter(BaseRateLimiter):
    """Spaces request starts evenly so a requests-per-minute quota is never exceeded.

    Unlike a token bucket, the first request goes out immediately. Slots are reserved
    under a lock, so concurrent callers (parallel summaries, several web visitors)
    queue up correctly. A 10% margin absorbs network jitter at the minute boundary.
    """

    def __init__(self, requests_per_minute: float) -> None:
        self.interval = 60.0 / requests_per_minute * 1.1
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def _reserve(self, blocking: bool) -> float | None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            if wait and not blocking:
                return None
            self._next_slot = max(now, self._next_slot) + self.interval
            return wait

    def acquire(self, *, blocking: bool = True) -> bool:
        wait = self._reserve(blocking)
        if wait is None:
            return False
        time.sleep(wait)
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        wait = self._reserve(blocking)
        if wait is None:
            return False
        await asyncio.sleep(wait)
        return True


_limiters: dict[tuple[str, str, str, float], MinIntervalRateLimiter] = {}
_limiters_lock = threading.Lock()


def shared_rate_limiter(
    provider: str, model: str, api_key: str | None, requests_per_minute: float
) -> MinIntervalRateLimiter:
    """One limiter per (provider, model, key): quotas belong to the key, so every run and
    every web visitor using the same key must share the same pacing."""
    key_id = hashlib.sha256(api_key.encode()).hexdigest()[:12] if api_key else "env"
    slot = (provider, model, key_id, requests_per_minute)
    with _limiters_lock:
        if slot not in _limiters:
            _limiters[slot] = MinIntervalRateLimiter(requests_per_minute)
        return _limiters[slot]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Instantiate the configured chat model with sensible production defaults."""
    provider = settings.resolved_provider
    model = settings.resolved_model
    # An explicit key (e.g. typed into the web UI) is passed straight to the client and
    # never written to os.environ, which a hosted app shares between all its visitors.
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None

    key_vars = PROVIDER_KEY_ENV.get(provider, ())
    if key_vars and not api_key and not any(os.getenv(var) for var in key_vars):
        raise LLMConfigurationError(
            f"Provider '{provider}' needs an API key: set {' or '.join(key_vars)} "
            "(in your shell or a .env file), or choose another provider."
        )
    if provider == "ollama":
        _check_ollama(settings.ollama_base_url)

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
    if api_key and provider != "ollama":
        kwargs["api_key"] = api_key
    elif provider == "google_genai" and not os.getenv("GOOGLE_API_KEY"):
        kwargs["google_api_key"] = os.getenv("GEMINI_API_KEY")
    if rpm := settings.requests_per_minute:
        kwargs["rate_limiter"] = shared_rate_limiter(provider, model, api_key, rpm)

    try:
        return init_chat_model(model, model_provider=provider, **kwargs)
    except ImportError as exc:  # provider integration package not installed
        raise LLMConfigurationError(str(exc)) from exc


def _check_ollama(base_url: str) -> None:
    """Fail fast with a helpful message instead of erroring on the first LLM call."""
    try:
        httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3).raise_for_status()
    except httpx.HTTPError as exc:
        hosted = ", ".join(var for env_vars in PROVIDER_KEY_ENV.values() for var in env_vars)
        raise LLMConfigurationError(
            f"No LLM available. Set an API key ({hosted}) in your shell or a .env file, "
            f"or start a local Ollama server at {base_url}."
        ) from exc


# ---------------------------------------------------------------------------
# Resilient calls
# ---------------------------------------------------------------------------


def with_backoff(runnable: Runnable) -> Runnable:
    """Retry transient failures (rate limits, 5xx, timeouts, bad JSON) with backoff."""
    return runnable.with_retry(
        retry_if_exception_type=RETRYABLE_ERRORS,
        exponential_jitter_params=RETRY_BACKOFF,
        stop_after_attempt=RETRY_ATTEMPTS,
    )


def structured_output(llm: BaseChatModel, schema: type[SchemaT]) -> Runnable[Any, SchemaT]:
    """`llm.with_structured_output(schema)` with the most reliable method per provider.

    Anthropic models with (adaptive) thinking reject forced tool choice, which is how
    LangChain implements structured output by default, so Claude uses the API's native
    JSON-schema output instead.
    """
    kwargs: dict[str, Any] = {}
    if type(llm).__name__ == "ChatAnthropic":
        kwargs["method"] = "json_schema"
    return with_backoff(llm.with_structured_output(schema, **kwargs))


def explain_llm_error(exc: BaseException) -> str:
    """Turn a provider exception into a short, actionable message for people."""
    text = str(exc)
    if isinstance(exc, ModelRateLimitError):
        if "perday" in text.lower().replace(" ", ""):
            return (
                "The model's daily quota for this API key is used up (free-tier keys have "
                "small daily limits). Try again tomorrow, choose another model, or enable "
                "billing on the key."
            )
        return (
            "The model provider's rate limit was still exceeded after waiting and retrying. "
            "Lower LLM_REQUESTS_PER_MINUTE (the Gemini free tier allows 5 per minute on "
            "Flash models), wait a minute and try again, or use a paid key."
        )
    if isinstance(exc, ModelAuthenticationError | ModelPermissionDeniedError):
        return "The API key was rejected. Check that it is correct and enabled for this model."
    if isinstance(exc, ModelNotFoundError):
        return "The model was not found. Check the model name (LLM_MODEL) for this provider."
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return f"{type(exc).__name__}: {first_line[:300]}"


def describe_model(llm: BaseChatModel) -> str:
    """Human-readable 'provider:model' label for logs and the UI."""
    name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or "unknown"
    return f"{llm._llm_type}:{name}"
