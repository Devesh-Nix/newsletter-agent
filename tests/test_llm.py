import os
import time

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError
from langchain_core.runnables import RunnableLambda

from newsletter_agent import llm as llm_module
from newsletter_agent.config import Settings, detect_provider
from newsletter_agent.llm import (
    LLMConfigurationError,
    MinIntervalRateLimiter,
    create_chat_model,
    explain_llm_error,
    structured_output,
    with_backoff,
)
from newsletter_agent.models import NewsletterPlan

ALL_KEYS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
]


@pytest.fixture
def no_keys(monkeypatch):
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_detect_provider_prefers_first_available_key(no_keys):
    assert detect_provider() == "ollama"
    no_keys.setenv("XAI_API_KEY", "x")
    assert detect_provider() == "xai"
    no_keys.setenv("GEMINI_API_KEY", "g")
    assert detect_provider() == "google_genai"
    no_keys.setenv("ANTHROPIC_API_KEY", "a")
    assert detect_provider() == "anthropic"


def test_resolved_model_uses_provider_default(no_keys):
    assert Settings(llm_provider="anthropic").resolved_model == "claude-opus-5"
    assert Settings(llm_provider="openai", llm_model="custom").resolved_model == "custom"


def test_missing_api_key_is_a_clear_configuration_error(no_keys):
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        create_chat_model(Settings(llm_provider="anthropic"))


def test_unreachable_ollama_is_a_clear_configuration_error(no_keys):
    settings = Settings(llm_provider="ollama", ollama_base_url="http://127.0.0.1:9")
    with pytest.raises(LLMConfigurationError, match="Ollama"):
        create_chat_model(settings)


def test_creates_claude_model_with_configured_limits(no_keys):
    no_keys.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    llm = create_chat_model(Settings(llm_provider="anthropic", llm_max_tokens=1234))
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-opus-5"
    assert llm.max_tokens == 1234


def test_explicit_api_key_is_used_without_touching_the_environment(no_keys):
    llm = create_chat_model(Settings(llm_provider="anthropic", llm_api_key="sk-ant-session"))
    assert llm.anthropic_api_key.get_secret_value() == "sk-ant-session"
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_claude_uses_native_json_schema_structured_output(no_keys, monkeypatch):
    no_keys.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    llm = create_chat_model(Settings(llm_provider="anthropic"))
    seen = {}
    original = ChatAnthropic.with_structured_output

    def spy(self, schema, **kwargs):
        seen.update(kwargs)
        return original(self, schema, **kwargs)

    monkeypatch.setattr(ChatAnthropic, "with_structured_output", spy)
    structured_output(llm, NewsletterPlan)
    assert seen["method"] == "json_schema"


# ---------------------------------------------------------------------------
# Pacing, retries and error messages
# ---------------------------------------------------------------------------


def test_gemini_is_paced_to_the_free_tier_by_default(no_keys):
    no_keys.setenv("GOOGLE_API_KEY", "g")
    assert Settings().requests_per_minute == 5
    assert Settings(llm_requests_per_minute=0).requests_per_minute is None
    assert Settings(llm_requests_per_minute=30).requests_per_minute == 30
    no_keys.setenv("ANTHROPIC_API_KEY", "a")
    assert Settings().requests_per_minute is None  # other providers: no cap unless asked


def test_rate_limiter_is_shared_by_everyone_using_the_same_key(no_keys):
    no_keys.setenv("GOOGLE_API_KEY", "g")
    first = create_chat_model(Settings(llm_provider="google_genai"))
    second = create_chat_model(Settings(llm_provider="google_genai"))
    visitor = create_chat_model(Settings(llm_provider="google_genai", llm_api_key="own-key"))

    assert isinstance(first.rate_limiter, MinIntervalRateLimiter)
    assert first.rate_limiter is second.rate_limiter
    assert visitor.rate_limiter is not first.rate_limiter


def test_min_interval_limiter_spaces_requests_but_starts_immediately():
    limiter = MinIntervalRateLimiter(requests_per_minute=600)  # 0.1 s apart (+10% margin)
    start = time.monotonic()
    limiter.acquire()
    first = time.monotonic() - start
    limiter.acquire()
    limiter.acquire()
    total = time.monotonic() - start

    assert first < 0.05
    assert total >= 2 * limiter.interval * 0.95
    assert not limiter.acquire(blocking=False)


def test_with_backoff_retries_rate_limits(monkeypatch):
    monkeypatch.setattr(llm_module, "RETRY_BACKOFF", {"initial": 0.01, "max": 0.02})
    calls = []

    def flaky(_):
        calls.append(1)
        if len(calls) < 3:
            raise ModelRateLimitError("429 RESOURCE_EXHAUSTED")
        return "ok"

    assert with_backoff(RunnableLambda(flaky)).invoke("x") == "ok"
    assert len(calls) == 3


def test_with_backoff_does_not_retry_bad_credentials(monkeypatch):
    monkeypatch.setattr(llm_module, "RETRY_BACKOFF", {"initial": 0.01, "max": 0.02})
    calls = []

    def unauthorised(_):
        calls.append(1)
        raise ModelAuthenticationError("401")

    with pytest.raises(ModelAuthenticationError):
        with_backoff(RunnableLambda(unauthorised)).invoke("x")
    assert len(calls) == 1


def test_errors_are_explained_in_plain_language():
    per_minute = ModelRateLimitError(
        "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    )
    per_day = ModelRateLimitError("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    assert "LLM_REQUESTS_PER_MINUTE" in explain_llm_error(per_minute)
    assert "daily quota" in explain_llm_error(per_day)
    assert "API key was rejected" in explain_llm_error(ModelAuthenticationError("401"))
    assert explain_llm_error(ValueError("boom\ntrace")) == "ValueError: boom"
