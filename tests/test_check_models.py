"""Preflight model check tests."""


from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import Settings


@pytest.fixture
def settings(clean_env, monkeypatch, repo_root) -> Settings:
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    return Settings(_env_file=repo_root / ".env.example")  # type: ignore[call-arg]


def _mock_post(monkeypatch, responder) -> list[dict]:
    """Replace httpx.AsyncClient.post; return the list of captured payloads."""
    captured: list[dict] = []

    async def fake_post(self, url, *, json=None, **kwargs):  # noqa: A002
        captured.append(json or {})
        request = httpx.Request("POST", str(self.base_url) + url)
        return responder(json or {}, request)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return captured


def _ok(payload, request) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={"choices": [{"message": {"content": "ok"}}]},
    )


def _mock_text_native(monkeypatch, check_models, *, ok: bool = True, detail: str = "ok") -> None:
    """TEXT_MODEL routes through a different client (langchain-google-genai,
    not httpx) whenever it's gemini-prefixed — see check_models.py's
    _ping_text_native_genai. Mock that function directly rather than faking
    the SDK's own transport."""

    async def fake(model, api_key):
        return check_models.Check("text", model, ok, 0.0, detail)

    monkeypatch.setattr(check_models, "_ping_text_native_genai", fake)


def test_bogus_model_id_exits_non_zero(check_models, settings, monkeypatch) -> None:
    """A provider 404 on an unknown model must fail the preflight."""

    def responder(payload, request) -> httpx.Response:
        return httpx.Response(
            404,
            request=request,
            json={"error": {"message": f"No endpoints found for {payload['model']}."}},
        )

    _mock_post(monkeypatch, responder)
    bogus = settings.model_copy(update={"text_model": "nonexistent/model-that-rotted:free"})

    checks = asyncio.run(check_models.run_checks(bogus))
    exit_code = check_models.report(checks)

    assert exit_code != 0
    assert all(not c.ok for c in checks)
    assert "404" in checks[0].detail


def test_one_failing_role_fails_the_whole_preflight(
    check_models, settings, monkeypatch
) -> None:
    """Two healthy roles must not mask a third that has gone away."""

    def _is_vision_payload(payload: dict) -> bool:
        content = payload["messages"][0]["content"]
        return isinstance(content, list) and any(p.get("type") == "image_url" for p in content)

    def responder(payload, request) -> httpx.Response:
        if _is_vision_payload(payload):
            return httpx.Response(
                429, request=request, json={"error": {"message": "rate limited"}}
            )
        return _ok(payload, request)

    _mock_post(monkeypatch, responder)
    _mock_text_native(monkeypatch, check_models)

    checks = asyncio.run(check_models.run_checks(settings))

    assert check_models.report(checks) != 0
    by_role = {c.role: c for c in checks}
    assert by_role["text"].ok and by_role["extractor"].ok
    assert not by_role["vision"].ok


def test_all_roles_healthy_exits_zero(check_models, settings, monkeypatch) -> None:
    captured = _mock_post(monkeypatch, _ok)
    _mock_text_native(monkeypatch, check_models)

    checks = asyncio.run(check_models.run_checks(settings))

    assert check_models.report(checks) == 0
    assert {c.role for c in checks} == {"text", "vision", "extractor"}
    assert {p["model"] for p in captured} == {
        settings.vision_model,
        settings.extractor_model,
    }


def test_vision_check_sends_an_actual_image(check_models, settings, monkeypatch) -> None:
    """FR-0.5: the vision role must be exercised with real image content."""
    captured = _mock_post(monkeypatch, _ok)
    _mock_text_native(monkeypatch, check_models)

    asyncio.run(check_models.run_checks(settings))

    vision = next(p for p in captured if p["model"] == settings.vision_model)
    parts = vision["messages"][0]["content"]
    image_parts = [p for p in parts if p["type"] == "image_url"]

    assert len(image_parts) == 1
    url = image_parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert len(url) > 200  # a real encoded PNG, not an empty string


def test_network_failure_is_reported_not_raised(
    check_models, settings, monkeypatch
) -> None:
    def responder(payload, request):
        raise httpx.ConnectTimeout("timed out", request=request)

    _mock_post(monkeypatch, responder)
    _mock_text_native(monkeypatch, check_models, ok=False, detail="ConnectTimeout: timed out")

    checks = asyncio.run(check_models.run_checks(settings))

    assert check_models.report(checks) != 0
    assert all("ConnectTimeout" in c.detail for c in checks)


def test_missing_api_key_exits_non_zero_without_traceback(
    check_models, clean_env, monkeypatch, capsys
) -> None:
    """No API key must be a readable message and a non-zero exit, not a crash."""
    monkeypatch.setattr(
        check_models.Settings,
        "model_config",
        {**check_models.Settings.model_config, "env_file": None},
    )
    check_models.get_settings.cache_clear()

    exit_code = check_models.main()

    assert exit_code != 0
    err = capsys.readouterr().err
    assert "LLM_API_KEY" in err
    assert "Traceback" not in err

    check_models.get_settings.cache_clear()


def test_text_native_genai_failure_is_reported_not_raised(check_models, monkeypatch) -> None:
    class FakeGenAI:
        def __init__(self, **kwargs):
            pass

        async def ainvoke(self, messages):
            raise RuntimeError("boom")

    import langchain_google_genai

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeGenAI)

    check = asyncio.run(check_models._ping_text_native_genai("gemini-3.6-flash", "test-key"))

    assert not check.ok
    assert "RuntimeError" in check.detail
    assert "boom" in check.detail
