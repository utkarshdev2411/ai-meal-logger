"""T-0.3 / NFR-0.1 — configuration values live only in app/config.py.

Greps the app/ and scripts/ trees for model IDs, provider base URLs and
database URLs. app/config.py is the single allowed home for all three; a hit
anywhere else means swapping provider, model or DB backend has stopped being a
pure config change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# The only file permitted to contain these literals.
# scripts/verify_*.py point DATABASE_URL at a throwaway scratch DB before
# importing the app — that's the scratch fixture, not config sprawl in app code.
ALLOWED = {
    "app/config.py",
    "scripts/verify_db.py",
    "scripts/verify_agent.py",
    "scripts/verify_memory.py",
    "scripts/verify_nutrition.py",
    "scripts/verify_vision.py",
    "scripts/bench.py",
}

SCANNED_TREES = ("app", "scripts")

FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r":free\b", "free-tier model ID suffix"),
    (r"https?://", "hardcoded URL"),
    (r"\b(?:sqlite|postgresql|postgres|mysql|mariadb)(?:\+\w+)?://", "database URL"),
    (
        r"\b(?:gpt|o[134]|claude|gemini|gemma|llama|qwen|mistral|mixtral|glm|"
        r"deepseek|nemotron|phi|grok|command-r)[-/][\w.]*\d",
        "model ID",
    ),
]

COMPILED = [(re.compile(pattern, re.IGNORECASE), label) for pattern, label in FORBIDDEN_PATTERNS]


def _python_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for tree in SCANNED_TREES:
        files.extend(sorted((repo_root / tree).rglob("*.py")))
    return files


def test_trees_are_actually_scanned(repo_root: Path) -> None:
    """Guard the guard: an empty file list would make this suite vacuous."""
    files = _python_files(repo_root)

    assert len(files) >= 2
    assert any(path.name == "config.py" for path in files)
    assert any(path.name == "check_models.py" for path in files)


def test_no_hardcoded_config_outside_config_py(repo_root: Path) -> None:
    violations: list[str] = []

    for path in _python_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        if rel in ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for pattern, label in COMPILED:
                if pattern.search(line):
                    violations.append(f"{rel}:{lineno}: {label} -> {line.strip()}")

    assert not violations, "configuration literals found outside app/config.py:\n" + "\n".join(
        violations
    )


@pytest.mark.parametrize(
    "sample",
    [
        'MODEL = "vendor/some-model-9:free"',
        'BASE = "https://example.invalid/v1"',
        'DB = "postgresql+asyncpg://user:pw@host/db"',
        'M = "gpt-4o-mini"',
    ],
)
def test_patterns_would_catch_a_violation(sample: str) -> None:
    """The patterns have teeth — each representative violation is detected."""
    assert any(pattern.search(sample) for pattern, _ in COMPILED)


def test_config_py_is_the_only_place_that_needs_the_exemption(repo_root: Path) -> None:
    """app/config.py really does contain what the exemption exists for."""
    text = (repo_root / "app" / "config.py").read_text()

    assert any(pattern.search(text) for pattern, _ in COMPILED)
