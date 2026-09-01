"""Collapse plurals, synonyms, and common spelling variants to a table key (FR-2.2)."""

from __future__ import annotations

import re

from app.nutrition.table import NUTRITION_TABLE

# alias (already lowercased, singular-ish) -> canonical table key. Entries the
# plural-stripping heuristic below can't reach on its own (different words,
# multi-word phrases, spelling variants).
_SYNONYMS: dict[str, str] = {
    "chapati": "roti",
    "chapatis": "roti",
    "phulka": "roti",
    "phulkas": "roti",
    "tea": "chai",
    "tea with milk": "chai",
    "masala chai": "chai",
    "chana masala": "chana",
    "chole": "chana",
    "chickpea curry": "chana",
    "chickpeas": "chana",
    "dahi": "curd",
    "yoghurt": "yogurt",
    "curd rice": "rice",
    "steamed rice": "rice",
    "chawal": "rice",
    "aloo curry": "aloo_sabzi",
    "aloo ki sabzi": "aloo_sabzi",
    "potato curry": "aloo_sabzi",
    "sabji": "sabzi",
    "veg sabzi": "sabzi",
    "mixed veg": "sabzi",
    "mixed vegetable curry": "sabzi",
    "paneer sabzi": "paneer",
    "cottage cheese": "paneer",
    "palak paneer": "palak_paneer",
    "spinach paneer": "palak_paneer",
    "chicken": "chicken_breast",
    "grilled chicken": "chicken_breast",
    "chicken breast": "chicken_breast",
    "peanut butter": "peanut_butter",
    "pb": "peanut_butter",
    "orange juice": "orange_juice",
    "oj": "orange_juice",
    "boiled egg": "egg",
    "boiled eggs": "egg",
    "fried egg": "egg",
    "fried eggs": "egg",
    "omelette": "egg",
    "omelet": "egg",
    "bread toast": "toast",
    "buttered toast": "toast",
    "medu vada": "vada",
    "black coffee": "coffee",
    "filter coffee": "coffee",
    "green salad": "salad",
    "veg salad": "salad",
    "chaas": "buttermilk",
    "chach": "buttermilk",
    "pav bhaji": "pav_bhaji",
    "khichadi": "khichdi",
    "kichdi": "khichdi",
}

_LEADING_FILLER = re.compile(r"^(a|an|some|one|1)\s+")


def _strip_plural(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def normalize(name: str) -> str | None:
    """Map free-text food name to a `NUTRITION_TABLE` key, or `None` if unknown."""
    cleaned = " ".join(name.strip().lower().split())
    cleaned = _LEADING_FILLER.sub("", cleaned)
    if not cleaned:
        return None

    for candidate in (cleaned, cleaned.replace("-", " "), cleaned.replace("_", " ")):
        if candidate in NUTRITION_TABLE:
            return candidate
        if candidate in _SYNONYMS:
            return _SYNONYMS[candidate]

    singular = _strip_plural(cleaned)
    if singular in NUTRITION_TABLE:
        return singular
    if singular in _SYNONYMS:
        return _SYNONYMS[singular]

    return None
