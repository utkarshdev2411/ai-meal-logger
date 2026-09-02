import base64
import json

import httpx

from app.config import get_settings
from app.vision.schema import VisionObservation

_TIMEOUT_S = 30.0

# The configured vision model advertises `response_format` but NOT
# `structured_outputs`, so schema enforcement can't be assumed: we ask for
# json_object, then validate ourselves and retry once before giving up.
_SCHEMA_HINT = (
    '{"items": [{"name": "<food>", "portion_estimate": "<e.g. 1 medium bowl>", '
    '"confidence": <0..1>, "alternatives": ["<other guess>"]}], '
    '"plate_context": "<e.g. thali, takeaway box, or null>", '
    '"overall_confidence": <0..1>, "unclear": ["<anything unidentifiable>"]}'
)

_PROMPT = (
    "Identify the food items on this plate. Estimate each portion. Where an item "
    "is genuinely ambiguous, put your best guess in `name` and the other "
    "candidates in `alternatives`, and lower its `confidence`. Put anything you "
    "cannot identify at all in `unclear`. Do not invent items you cannot see.\n\n"
    "Reply with JSON only, matching this exact shape:\n" + _SCHEMA_HINT
)


class VisionExtractionError(Exception):
    pass


async def _request(client: httpx.AsyncClient, model: str, data_url: str, strict: bool) -> str:
    content = [
        {"type": "text", "text": _PROMPT if not strict else _PROMPT + "\n\nReturn ONLY the JSON object, no prose, no markdown fences."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    response = await client.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _parse(raw: str) -> VisionObservation:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.lstrip("`")
        text = text.removeprefix("json").strip()
    return VisionObservation.model_validate(json.loads(text))


async def extract_vision(image_bytes: bytes, mime_type: str) -> VisionObservation:
    settings = get_settings()
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"

    async with httpx.AsyncClient(
        base_url=settings.llm_base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {settings.require_api_key()}"},
        timeout=_TIMEOUT_S,
    ) as client:
        last_error: Exception | None = None
        for strict in (False, True):
            try:
                return _parse(await _request(client, settings.vision_model, data_url, strict))
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc

    raise VisionExtractionError(f"vision model returned unusable JSON: {last_error}")
