"""Typed, env-sourced application settings.

This module is the **only** place in the codebase where a model ID, an LLM base
URL, or a database URL may appear (NFR-0.1). Everything else reads them from
``get_settings()``. That is what makes swapping provider, model, or database
backend a config change rather than a code change.

The defaults below are chosen so that the app imports and starts with nothing
but ``.env.example`` plus a real API key (NFR-0.2).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------
# Defaults. Kept as module constants so the values live in exactly one place
# and .env.example can be diffed against them.
# --------------------------------------------------------------------------

# OpenRouter, OpenAI-compatible. Any other OpenAI-compatible endpoint (Groq,
# Together, a local vLLM, Gemini's compat layer) works by changing this alone.
DEFAULT_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# Free-tier OpenRouter IDs, present on the provider's model list as of
# 2026-09-01. Free-tier IDs rotate and get rate-limited without notice, so
# always run `python scripts/check_models.py` before relying on them.
DEFAULT_TEXT_MODEL = "gemini-2.5-flash"
DEFAULT_VISION_MODEL = "gemini-2.5-pro"
DEFAULT_EXTRACTOR_MODEL = "gemini-2.5-flash-lite"

# SQLite so a clean clone runs with zero setup; swap for
# postgresql+asyncpg://user:pass@host/db to use hosted Postgres.
DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./calorai.db"


class MissingApiKeyError(RuntimeError):
    """Raised when an LLM call is attempted without a usable ``LLM_API_KEY``."""


class SettingsError(RuntimeError):
    """Raised when the environment cannot produce a valid ``Settings``."""


class Settings(BaseSettings):
    """All runtime configuration, sourced from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- provider ---------------------------------------------------------
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_api_key: str = Field(
        ...,
        description="API key for the OpenAI-compatible provider. Required.",
    )

    # --- the three model roles (never one model for everything) -----------
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL

    # --- persistence ------------------------------------------------------
    database_url: str = DEFAULT_DATABASE_URL

    # --- latency / quality knobs -----------------------------------------
    reply_max_tokens: int = 80
    memory_top_k: int = 8
    memory_token_budget: int = 300
    history_turns: int = 6
    image_max_edge: int = 768
    fast_path: bool = False
    bench_default_runs: int = 15

    # --- tracing (optional) ----------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "calorai-meal-logger"

    @property
    def has_api_key(self) -> bool:
        """True when ``LLM_API_KEY`` looks like a real key rather than a placeholder."""
        key = self.llm_api_key.strip()
        return bool(key) and "replace" not in key.lower()

    def require_api_key(self) -> str:
        """Return the API key, or raise a readable error instead of a 401 later."""
        if not self.has_api_key:
            raise MissingApiKeyError(
                "LLM_API_KEY is not set. Copy .env.example to .env and put a real "
                "provider API key in it, or export LLM_API_KEY in your shell."
            )
        return self.llm_api_key.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings accessor.

    Cached because settings are read on every request path and constructing a
    ``BaseSettings`` re-reads the ``.env`` file each time.
    """
    try:
        return Settings()  # type: ignore[call-arg]  # values come from the env
    except ValidationError as exc:
        missing = [
            ".".join(str(p) for p in err["loc"])
            for err in exc.errors()
            if err["type"] == "missing"
        ]
        if missing:
            keys = ", ".join(name.upper() for name in missing)
            raise SettingsError(
                f"Missing required configuration: {keys}. "
                "Copy .env.example to .env and fill it in."
            ) from exc
        raise SettingsError(f"Invalid configuration:\n{exc}") from exc
