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


async def _ping_text_native_genai(model: str, api_key: str) -> Check:
    """The real app calls TEXT_MODEL through langchain-google-genai's native
    client, not the OpenAI-compat endpoint (see app/agent/graph.py — needed
    for Gemini 3.x's mandatory thought_signature tool-call round-trip, which
    langchain-openai does not support). Ping through the same client so this
    check proves what the app actually uses, not a different code path."""
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    started = time.perf_counter()
    try:
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, max_output_tokens=16)
        response = await llm.ainvoke([HumanMessage(content="Reply with the single word: ok")])
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        return Check("text", model, False, elapsed, f"{type(exc).__name__}: {exc}")

    elapsed = (time.perf_counter() - started) * 1000
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return Check("text", model, True, elapsed, str(content).strip()[:60])


async def run_checks(settings: Settings | None = None, api_key: str | None = None) -> list[Check]:
    """Ping all three roles concurrently, against one specific key, and
    return their outcomes. Defaults to the primary key when none is given."""
    settings = settings or get_settings()
    api_key = api_key or settings.require_api_key()

    text_check = (
        _ping_text_native_genai(settings.text_model, api_key)
        if settings.text_model.startswith("gemini")
        else None
    )
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
        checks = [
            text_check
            or _ping(client, "text", settings.text_model, [{"role": "user", "content": "Reply with the single word: ok"}]),
            _ping(client, "vision", settings.vision_model, vision_messages, 24),
            _ping(client, "extractor", settings.extractor_model, extractor_messages, 48),
        ]
        return list(await asyncio.gather(*checks))


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


def _mask(key: str) -> str:
    return f"...{key[-6:]}" if len(key) > 10 else key


async def run_pool_checks(settings: Settings) -> dict[str, list[Check]]:
    """Verify every key in the round-robin pool independently, not just the
    primary — a key can auth fine yet 404 on the shared model, or vice versa."""
    results: dict[str, list[Check]] = {}
    for key in settings.api_key_pool:
        results[_mask(key)] = await run_checks(settings, api_key=key)
    return results


def report_pool(results: dict[str, list[Check]]) -> int:
    exit_code = 0
    for label, checks in results.items():
        print(f"key {label}:")
        if report(checks) != 0:
            exit_code = 1
        print()
    if exit_code == 0:
        print(f"All {len(results)} key(s) in the pool are usable.")
    else:
        print("At least one key in the pool failed — see above.")
    return exit_code


def main() -> int:
    try:
        settings = get_settings()
        settings.require_api_key()
    except (SettingsError, MissingApiKeyError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print(f"Endpoint: {settings.llm_base_url}\n")
    return report_pool(asyncio.run(run_pool_checks(settings)))


if __name__ == "__main__":
    raise SystemExit(main())
