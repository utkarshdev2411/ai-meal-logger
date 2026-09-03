"""Typed, env-sourced application settings.

This module is the **only** place in the codebase where a model ID, an LLM base
URL, or a database URL may appear (NFR-0.1). Everything else reads them from
``get_settings()``. That is what makes swapping provider, model, or database
backend a config change rather than a code change.

The defaults below are chosen so that the app imports and starts with nothing
but ``.env.example`` plus a real API key (NFR-0.2).
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from threading import Lock

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------
# Defaults. Kept as module constants so the values live in exactly one place
# and .env.example can be diffed against them.
# --------------------------------------------------------------------------

# OpenRouter, OpenAI-compatible. Any other OpenAI-compatible endpoint (Groq,
# Together, a local vLLM, Gemini's compat layer) works by changing this alone.
DEFAULT_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# One model for all three roles, by explicit choice: gemini-3.1-flash-lite,
# confirmed live (tool calling, image input, JSON extraction) against every
# key in the pool on 2026-09-03. Quota headroom comes from the key pool
# (see next_api_key), not from spreading roles across model tiers.
DEFAULT_TEXT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_VISION_MODEL = "gemini-3.1-flash-lite"
DEFAULT_EXTRACTOR_MODEL = "gemini-3.1-flash-lite"

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
    llm_api_keys_extra: str = Field(
        default="",
        description="Comma-separated additional keys, round-robin-ed with llm_api_key.",
    )

    # --- the three model roles ---------------------------------------------
    # All three point at the same model by design here: this project runs a
    # pool of API keys round-robin (see next_api_key), which buys quota
    # headroom that a single key's per-model daily cap cannot.
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
        """Return the primary API key, or raise a readable error instead of a 401 later."""
        if not self.has_api_key:
            raise MissingApiKeyError(
                "LLM_API_KEY is not set. Copy .env.example to .env and put a real "
                "provider API key in it, or export LLM_API_KEY in your shell."
            )
        return self.llm_api_key.strip()

    @property
    def api_key_pool(self) -> list[str]:
        """The primary key plus every key in ``LLM_API_KEYS_EXTRA``, deduped,
        in order. ``require_api_key()`` still validates the primary key is
        present; this just expands the pool ``next_api_key`` rotates over."""
        primary = self.require_api_key()
        extra = [k.strip() for k in self.llm_api_keys_extra.split(",") if k.strip()]
        pool: list[str] = []
        for key in [primary, *extra]:
            if key not in pool:
                pool.append(key)
        return pool


_key_cycle_lock = Lock()
_key_cycle: itertools.cycle | None = None
_key_cycle_pool: tuple[str, ...] = ()


def next_api_key() -> str:
    """Round-robin across ``Settings.api_key_pool``. A single key behaves
    exactly like calling ``require_api_key()`` every time."""
    global _key_cycle, _key_cycle_pool
    pool = tuple(get_settings().api_key_pool)
    with _key_cycle_lock:
        if pool != _key_cycle_pool:
            _key_cycle = itertools.cycle(pool)
            _key_cycle_pool = pool
        return next(_key_cycle)


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
