"""Runtime configuration, loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai", "google_genai", "xai", "ollama"]

#: Default chat model per provider. Override with LLM_MODEL.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-6-sol",
    "google_genai": "gemini-3.8-flash",
    "xai": "grok-4.7",
    "ollama": "qwen3:8b",
}

#: Environment variable(s) holding the API key for each hosted provider.
PROVIDER_KEY_ENV: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google_genai": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "xai": ("XAI_API_KEY",),
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GOAL = "Create a weekly newsletter on latest AI agent news and send it to our subscribers."


def detect_provider() -> Provider:
    """Pick the first hosted provider with an API key set; fall back to local Ollama."""
    for provider, env_vars in PROVIDER_KEY_ENV.items():
        if any(os.getenv(var) for var in env_vars):
            return provider  # type: ignore[return-value]
    return "ollama"


class Settings(BaseSettings):
    """All tunable knobs of the agent. Every field maps to an upper-case env var."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -----------------------------------------------------------------
    llm_provider: Provider | Literal["auto"] = "auto"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = Field(
        default=None, description="Explicit key for the selected provider (overrides env)."
    )
    llm_temperature: float | None = Field(
        default=None, description="Only sent when set; some models reject sampling params."
    )
    llm_max_tokens: int = 8000
    llm_timeout: float = 180.0
    ollama_base_url: str = "http://localhost:11434"

    # --- Newsletter ----------------------------------------------------------
    newsletter_name: str = "The Agentic Brief"
    sender_name: str = "The Agentic Brief"
    sender_email: str = "newsletter@example.com"
    subscribers_file: Path = PROJECT_ROOT / "data" / "subscribers.json"
    outbox_dir: Path = PROJECT_ROOT / "outbox"

    # --- Agent behaviour -----------------------------------------------------
    lookback_days: int | None = Field(
        default=None, ge=1, le=31, description="Force the news window; None = infer from goal."
    )
    min_articles: int = Field(default=5, ge=1, le=10)
    max_articles: int = Field(default=7, ge=1, le=10)
    max_research_rounds: int = Field(default=3, ge=1, le=6)
    max_revisions: int = Field(default=2, ge=0, le=5)
    quality_threshold: float = Field(default=8.0, ge=1, le=10)
    max_candidates_for_curation: int = 40

    # --- Research tools ------------------------------------------------------
    tavily_api_key: str | None = None
    http_timeout: float = 15.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0 Safari/537.36 NewsletterAgent/1.0"
    )
    article_char_limit: int = 12_000

    @model_validator(mode="after")
    def _check_article_bounds(self) -> Settings:
        if self.min_articles > self.max_articles:
            raise ValueError("min_articles must be <= max_articles")
        return self

    @property
    def default_window_days(self) -> int:
        return self.lookback_days or 7

    @property
    def resolved_provider(self) -> Provider:
        return detect_provider() if self.llm_provider == "auto" else self.llm_provider

    @property
    def resolved_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.resolved_provider]


def load_settings(**overrides: object) -> Settings:
    """Load `.env` into the process environment (provider SDKs read keys from there)
    and build a `Settings` object, applying any explicit overrides on top."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(override=False)  # a .env in the current working directory, if any
    return Settings(**{k: v for k, v in overrides.items() if v is not None})
