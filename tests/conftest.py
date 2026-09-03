"""Shared test fixtures."""


from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every key Settings reads; cleared so tests are hermetic.
CONFIG_ENV_KEYS = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "TEXT_MODEL",
    "VISION_MODEL",
    "EXTRACTOR_MODEL",
    "DATABASE_URL",
    "REPLY_MAX_TOKENS",
    "MEMORY_TOP_K",
    "MEMORY_TOKEN_BUDGET",
    "HISTORY_TURNS",
    "IMAGE_MAX_EDGE",
    "FAST_PATH",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all configuration env vars for the duration of a test."""
    for key in CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def check_models() -> ModuleType:
    """Load scripts/check_models.py as a module (scripts/ is not a package)."""
    path = REPO_ROOT / "scripts" / "check_models.py"
    spec = importlib.util.spec_from_file_location("check_models", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_models"] = module
    spec.loader.exec_module(module)
    return module
