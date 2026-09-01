"""T-0.1 — settings load from .env.example; a missing key errors readably."""

from __future__ import annotations

import pytest

from app import config as config_module
from app.config import (
    MissingApiKeyError,
    Settings,
    SettingsError,
    get_settings,
)


def _env_example_settings(repo_root) -> Settings:
    return Settings(_env_file=repo_root / ".env.example")  # type: ignore[call-arg]


def test_settings_load_from_env_example(clean_env, repo_root) -> None:
    """NFR-0.2: .env.example alone is a complete, valid configuration."""
    settings = _env_example_settings(repo_root)

    assert settings.text_model == config_module.DEFAULT_TEXT_MODEL
    assert settings.vision_model == config_module.DEFAULT_VISION_MODEL
    assert settings.extractor_model == config_module.DEFAULT_EXTRACTOR_MODEL
    assert settings.llm_base_url == config_module.DEFAULT_LLM_BASE_URL
    assert settings.database_url == config_module.DEFAULT_DATABASE_URL
    assert settings.reply_max_tokens == 80
    assert settings.memory_top_k == 8
    assert settings.memory_token_budget == 300
    assert settings.history_turns == 6
    assert settings.image_max_edge == 768
    assert settings.fast_path is False
    assert settings.langsmith_tracing is False


def test_env_example_covers_every_key(clean_env, repo_root) -> None:
    """.env.example must document every field, or FR-0.3 has drifted."""
    text = (repo_root / ".env.example").read_text()
    declared = {
        line.split("=", 1)[0].strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    expected = {name.upper() for name in Settings.model_fields}

    assert expected == declared


def test_three_model_roles_are_distinct(clean_env, repo_root) -> None:
    """Never one model for everything."""
    settings = _env_example_settings(repo_root)
    roles = {settings.text_model, settings.vision_model, settings.extractor_model}

    assert len(roles) == 3


def test_missing_api_key_gives_readable_error(clean_env, monkeypatch) -> None:
    """A missing required key must name the key, not dump a pydantic traceback."""
    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})
    get_settings.cache_clear()

    with pytest.raises(SettingsError) as excinfo:
        get_settings()

    message = str(excinfo.value)
    assert "LLM_API_KEY" in message
    assert ".env.example" in message

    get_settings.cache_clear()


def test_placeholder_api_key_is_not_accepted(clean_env, repo_root) -> None:
    """The shipped placeholder must be rejected before it becomes a 401."""
    settings = _env_example_settings(repo_root)

    assert settings.has_api_key is False
    with pytest.raises(MissingApiKeyError) as excinfo:
        settings.require_api_key()
    assert "LLM_API_KEY" in str(excinfo.value)


def test_real_api_key_is_accepted(clean_env, monkeypatch, repo_root) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    settings = _env_example_settings(repo_root)

    assert settings.has_api_key is True
    assert settings.require_api_key() == "test-key-123"


def test_get_settings_is_cached(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    get_settings.cache_clear()

    assert get_settings() is get_settings()

    get_settings.cache_clear()
