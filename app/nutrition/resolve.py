"""Item -> macros (FR-2.3, FR-2.4, FR-2.5).

Order: normalized table lookup -> in-process cache -> one batched LLM call for
whatever's left. Never raises — an item the LLM can't place either still
becomes a `ResolvedItem`, just a low-confidence one.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from app.config import get_settings, next_api_key
from app.nutrition.normalize import normalize
from app.nutrition.table import NUTRITION_TABLE

logger = logging.getLogger(__name__)

NutritionSource = Literal["table", "cache", "model"]

_LLM_TIMEOUT_S = 20.0
_TABLE_CONFIDENCE = 0.95
_FALLBACK_CONFIDENCE = 0.2

# Reasonable defaults for an item neither the table nor the model can place,
# so an unresolvable name degrades to a flagged estimate instead of a crash.
_FALLBACK_ESTIMATE = {
    "unit": "serving",
    "kcal": 250.0,
    "protein_g": 8.0,
    "carbs_g": 30.0,
    "fat_g": 8.0,
    "fiber_g": 2.0,
}


class RawItem(BaseModel):
    name: str
    quantity: float = 1.0
    unit: str | None = None


class ResolvedItem(BaseModel):
    name: str
    canonical_key: str
    quantity: float
    unit: str
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    nutrition_source: NutritionSource
    confidence: float


class LLMNutritionItem(BaseModel):
    """One entry of the batched fallback response, per default unit (pre-multiply)."""

    name: str
    unit: str = "serving"
    kcal: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class LLMNutritionBatch(BaseModel):
    items: list[LLMNutritionItem]


# Cache of LLM-resolved per-unit macros, keyed by the *normalized-or-raw* name
# that missed the table. A simple process-lifetime dict is enough here — FR-2.3
# just asks for an in-process cache, not eviction under memory pressure.
_CACHE: dict[str, LLMNutritionItem] = {}


def _cache_key(raw_name: str) -> str:
    return " ".join(raw_name.strip().lower().split())


def _from_table(key: str, quantity: float, source: NutritionSource, confidence: float, raw_name: str) -> ResolvedItem:
    entry = NUTRITION_TABLE[key]
    return ResolvedItem(
        name=raw_name,
        canonical_key=key,
        quantity=quantity,
        unit=entry["unit"],
        kcal=entry["kcal"] * quantity,
        protein_g=entry["protein_g"] * quantity,
        carbs_g=entry["carbs_g"] * quantity,
        fat_g=entry["fat_g"] * quantity,
        fiber_g=entry["fiber_g"] * quantity,
        nutrition_source=source,
        confidence=confidence,
    )


def _from_llm_item(item: LLMNutritionItem, quantity: float, raw_name: str) -> ResolvedItem:
    return ResolvedItem(
        name=raw_name,
        canonical_key=_cache_key(item.name) or _cache_key(raw_name),
        quantity=quantity,
        unit=item.unit,
        kcal=item.kcal * quantity,
        protein_g=item.protein_g * quantity,
        carbs_g=item.carbs_g * quantity,
        fat_g=item.fat_g * quantity,
        fiber_g=item.fiber_g * quantity,
        nutrition_source="model",
        confidence=item.confidence,
    )


def _fallback_item(raw_name: str, quantity: float) -> ResolvedItem:
    return ResolvedItem(
        name=raw_name,
        canonical_key=_cache_key(raw_name),
        quantity=quantity,
        unit=_FALLBACK_ESTIMATE["unit"],
        kcal=_FALLBACK_ESTIMATE["kcal"] * quantity,
        protein_g=_FALLBACK_ESTIMATE["protein_g"] * quantity,
        carbs_g=_FALLBACK_ESTIMATE["carbs_g"] * quantity,
        fat_g=_FALLBACK_ESTIMATE["fat_g"] * quantity,
        fiber_g=_FALLBACK_ESTIMATE["fiber_g"] * quantity,
        nutrition_source="model",
        confidence=_FALLBACK_CONFIDENCE,
    )


async def _call_llm_batch(names: list[str]) -> dict[str, LLMNutritionItem]:
    """One call for every miss in the batch (FR-2.4). Returns a name -> item map;
    missing/unparseable names are simply absent, and the caller falls back."""
    settings = get_settings()
    api_key = next_api_key()

    prompt = (
        "For each food item below, estimate its nutrition for ONE typical default "
        "serving (not per 100g). Reply with JSON only, matching this exact shape:\n"
        '{"items": [{"name": "<as given>", "unit": "<piece|bowl|cup|glass|serving|g>", '
        '"kcal": <number>, "protein_g": <number>, "carbs_g": <number>, "fat_g": <number>, '
        '"fiber_g": <number>, "confidence": <0..1>}]}\n\n'
        "Items:\n" + "\n".join(f"- {n}" for n in names)
    )

    async with httpx.AsyncClient(
        base_url=settings.llm_base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=_LLM_TIMEOUT_S,
    ) as client:
        response = await client.post(
            "/chat/completions",
            json={
                "model": settings.extractor_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

    batch = LLMNutritionBatch.model_validate(json.loads(content))
    return {_cache_key(item.name): item for item in batch.items}


async def resolve(items: list[RawItem]) -> list[ResolvedItem]:
    resolved: dict[int, ResolvedItem] = {}
    misses: list[tuple[int, RawItem]] = []

    for idx, raw in enumerate(items):
        key = normalize(raw.name)
        if key is not None and key in NUTRITION_TABLE:
            resolved[idx] = _from_table(key, raw.quantity, "table", _TABLE_CONFIDENCE, raw.name)
            continue

        cache_key = _cache_key(raw.name)
        cached = _CACHE.get(cache_key)
        if cached is not None:
            resolved[idx] = _from_llm_item(cached, raw.quantity, raw.name)
            resolved[idx].nutrition_source = "cache"
            continue

        misses.append((idx, raw))

    if misses:
        miss_names = [raw.name for _, raw in misses]
        llm_results: dict[str, LLMNutritionItem] = {}
        try:
            llm_results = await _call_llm_batch(miss_names)
        except Exception:  # network, auth, malformed JSON — never propagate
            logger.warning("nutrition LLM fallback failed for %s", miss_names, exc_info=True)

        for idx, raw in misses:
            item = llm_results.get(_cache_key(raw.name))
            if item is not None:
                _CACHE[_cache_key(raw.name)] = item
                resolved[idx] = _from_llm_item(item, raw.quantity, raw.name)
            else:
                resolved[idx] = _fallback_item(raw.name, raw.quantity)

    return [resolved[i] for i in range(len(items))]
