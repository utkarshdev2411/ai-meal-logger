#!/usr/bin/env python
"""Verification script for the nutrition module."""


from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.nutrition import resolve as resolve_mod  # noqa: E402
from app.nutrition.resolve import RawItem, resolve  # noqa: E402
from app.nutrition.table import NUTRITION_TABLE  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


async def check_table_only() -> None:
    """1: table-only resolution, zero LLM calls, quantity-multiplied macros."""
    calls = {"n": 0}
    original = resolve_mod._call_llm_batch

    async def spy(names: list[str]):
        calls["n"] += 1
        return await original(names)

    resolve_mod._call_llm_batch = spy
    try:
        items = [RawItem(name="parathas", quantity=2), RawItem(name="chai", quantity=1)]
        out = await resolve(items)
    finally:
        resolve_mod._call_llm_batch = original

    per_unit = NUTRITION_TABLE["paratha"]
    paratha, chai = out
    ok = (
        calls["n"] == 0
        and paratha.nutrition_source == "table"
        and paratha.kcal == per_unit["kcal"] * 2
        and paratha.protein_g == per_unit["protein_g"] * 2
        and chai.nutrition_source == "table"
        and chai.kcal == NUTRITION_TABLE["chai"]["kcal"]
    )
    check(
        "table-only resolution: zero LLM calls, quantity-multiplied macros",
        ok,
        f"llm_calls={calls['n']}, paratha.kcal={paratha.kcal}, chai.kcal={chai.kcal}",
    )


async def check_synonyms() -> None:
    """2: plural/synonym variants normalize to the same canonical key & macros."""
    out = await resolve([RawItem(name=n) for n in ("rotis", "roti", "chapati", "chapatis")])
    keys = {r.canonical_key for r in out}
    kcals = {r.kcal for r in out}
    ok = keys == {"roti"} and len(kcals) == 1 and all(r.nutrition_source == "table" for r in out)
    check("plural/synonym variants collapse to one canonical key", ok, f"keys={keys}, kcals={kcals}")


async def check_unresolvable_no_llm() -> None:
    """3: an item nothing can resolve doesn't raise, and comes back low-confidence."""
    async def failing(_names: list[str]):
        raise RuntimeError("simulated: no network / no API key")

    original = resolve_mod._call_llm_batch
    resolve_mod._call_llm_batch = failing
    try:
        out = await resolve([RawItem(name="xyzzy-nonexistent-food-item-9000")])
    except Exception as exc:  # the thing under test — must not happen
        check("unresolvable item never raises", False, f"raised {type(exc).__name__}: {exc}")
        return
    finally:
        resolve_mod._call_llm_batch = original

    item = out[0]
    ok = item.confidence < 0.5 and item.kcal > 0
    check(
        "unresolvable item never raises, returns low-confidence estimate",
        ok,
        f"confidence={item.confidence}, kcal={item.kcal}, source={item.nutrition_source}",
    )


async def check_batched_fallback() -> None:
    """4: N misses in one resolve() call trigger exactly one LLM call, not N."""
    from app.nutrition.resolve import LLMNutritionBatch, LLMNutritionItem

    calls = {"n": 0}

    async def fake(names: list[str]):
        calls["n"] += 1
        return {
            resolve_mod._cache_key(n): LLMNutritionItem(
                name=n, unit="serving", kcal=300, protein_g=10, carbs_g=30, fat_g=10, fiber_g=2, confidence=0.6
            )
            for n in names
        }

    original = resolve_mod._call_llm_batch
    resolve_mod._call_llm_batch = fake
    try:
        misses = ["quinoa power bowl", "dragonfruit smoothie", "kimchi jjigae"]
        out = await resolve([RawItem(name=n) for n in misses])
    finally:
        resolve_mod._call_llm_batch = original

    ok = (
        calls["n"] == 1
        and len(out) == len(misses)
        and all(r.nutrition_source == "model" and r.confidence == 0.6 for r in out)
    )
    check("N misses trigger exactly one batched LLM call", ok, f"llm_calls={calls['n']} for {len(misses)} misses")

    # a name the fake LLM also can't place still degrades gracefully, no exception
    calls["n"] = 0

    async def partial(names: list[str]):
        calls["n"] += 1
        return {}  # nothing placed

    resolve_mod._call_llm_batch = partial
    try:
        out2 = await resolve([RawItem(name="totally unknown dish")])
    finally:
        resolve_mod._call_llm_batch = original
    ok2 = calls["n"] == 1 and out2[0].confidence < 0.5
    check("LLM miss on a name it also can't place still returns an estimate", ok2, f"llm_calls={calls['n']}")


async def main() -> int:
    await check_table_only()
    await check_synonyms()
    await check_unresolvable_no_llm()
    await check_batched_fallback()

    failed = [name for name, ok, _ in results if not ok]
    print()
    if failed:
        print(f"{len(failed)}/{len(results)} checks FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(results)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
