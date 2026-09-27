import pytest
from langchain_anthropic import ChatAnthropic

from newsletter_agent.config import Settings, detect_provider
from newsletter_agent.llm import LLMConfigurationError, create_chat_model, structured_output
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
