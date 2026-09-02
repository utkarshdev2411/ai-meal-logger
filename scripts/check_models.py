#!/usr/bin/env python
"""Preflight: prove the three configured model roles actually respond.

Free-tier model IDs get renamed, retired and rate-limited without notice, so a
demo that has not been preflighted is a demo that might not have a text model.
This script pings each role through the configured OpenAI-compatible endpoint,
reports latency and ok/fail per role, and exits non-zero if any role failed.

The vision check sends a real (tiny, generated) PNG as a base64 data URL, so it
exercises the multimodal path rather than just proving the model ID exists.

    python scripts/check_models.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow `python scripts/check_models.py` from a clean clone without installing
# the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from app.config import (  # noqa: E402
    MissingApiKeyError,
    Settings,
    SettingsError,
    get_settings,
)

TIMEOUT_S = 45.0


@dataclass
class Check:
    """Outcome of pinging one model role."""

    role: str
    model: str
    ok: bool
    ms: float
    detail: str = ""


def _test_image_data_url() -> str:
    """A tiny generated PNG as a data URL — enough to exercise the vision path."""
    img = Image.new("RGB", (96, 96), (245, 240, 225))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, 88, 88), fill=(228, 222, 205), outline=(120, 110, 90))
    draw.ellipse((30, 30, 66, 66), fill=(196, 150, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


async def _ping(
    client: httpx.AsyncClient,
    role: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 16,
) -> Check:
    """One chat-completions round trip, timed, never raising."""
    started = time.perf_counter()
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if model.startswith("gemini") and role == "text":
        payload["reasoning_effort"] = "none"
    try:
        response = await client.post("/chat/completions", json=payload)
    except Exception as exc:  # network, timeout, DNS — all just "fail"
        elapsed = (time.perf_counter() - started) * 1000
        return Check(role, model, False, elapsed, f"{type(exc).__name__}: {exc}")

    elapsed = (time.perf_counter() - started) * 1000

    if response.status_code >= 400:
        return Check(role, model, False, elapsed, _error_detail(response))

    try:
        choices = response.json().get("choices") or []
        text = choices[0]["message"]["content"] or ""
    except Exception:
        return Check(role, model, False, elapsed, "malformed response body")

    return Check(role, model, True, elapsed, (text or "").strip()[:60])


def _error_detail(response: httpx.Response) -> str:
    """Best-effort human-readable reason from a provider error response."""
    try:
        body = response.json()
        message = body.get("error", {}).get("message") or body.get("message")
    except Exception:
        message = None
    return f"HTTP {response.status_code}: {message or response.text[:120]}"


async def run_checks(settings: Settings | None = None) -> list[Check]:
    """Ping all three roles concurrently and return their outcomes."""
    settings = settings or get_settings()
    api_key = settings.require_api_key()

    text_messages = [{"role": "user", "content": "Reply with the single word: ok"}]
    vision_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Name the food in one word."},
                {
                    "type": "image_url",
                    "image_url": {"url": _test_image_data_url()},
                },
            ],
        }
    ]
    extractor_messages = [
        {
            "role": "user",
            "content": 'Extract JSON facts from: "i am vegetarian". Reply JSON only.',
        }
    ]

    async with httpx.AsyncClient(
        base_url=settings.llm_base_url.rstrip("/"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=TIMEOUT_S,
    ) as client:
        return list(
            await asyncio.gather(
                _ping(client, "text", settings.text_model, text_messages),
                _ping(client, "vision", settings.vision_model, vision_messages, 24),
                _ping(client, "extractor", settings.extractor_model, extractor_messages, 48),
            )
        )


def report(checks: list[Check]) -> int:
    """Print the result table; return the process exit code."""
    width = max((len(c.model) for c in checks), default=10)
    for check in checks:
        status = "ok  " if check.ok else "FAIL"
        print(f"{check.role:<10} {check.model:<{width}}  {check.ms:>7.0f} ms  {status}  {check.detail}")

    failed = [c for c in checks if not c.ok]
    if failed:
        print(
            f"\n{len(failed)} of {len(checks)} model role(s) failed: "
            + ", ".join(c.role for c in failed)
            + ".\nFree-tier IDs rot and get rate-limited — update the model IDs in .env."
        )
        return 1

    print(f"\nAll {len(checks)} model roles responded.")
    return 0


def main() -> int:
    try:
        settings = get_settings()
        settings.require_api_key()
    except (SettingsError, MissingApiKeyError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print(f"Endpoint: {settings.llm_base_url}\n")
    return report(asyncio.run(run_checks(settings)))


if __name__ == "__main__":
    raise SystemExit(main())
