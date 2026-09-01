"""Hardcoded nutrition table (FR-2.1).

Each entry is per **default unit** (a single serving as most people would say
it, e.g. one roti, one bowl of dal) — `resolve()` multiplies by quantity to
get item totals, per docs/SCHEMA.md §1.3. Values are reasonable household
estimates, not lab-grade — good enough for a calorie-tracking chat, not a
diet-clinic printout.
"""

from __future__ import annotations

from typing import TypedDict


class TableEntry(TypedDict):
    unit: str
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float


# canonical_key -> per-default-unit macros
NUTRITION_TABLE: dict[str, TableEntry] = {
    # --- Indian staples ---
    "roti": {"unit": "piece", "kcal": 71, "protein_g": 3.0, "carbs_g": 15.0, "fat_g": 0.4, "fiber_g": 2.0},
    "paratha": {"unit": "piece", "kcal": 126, "protein_g": 3.0, "carbs_g": 18.0, "fat_g": 5.0, "fiber_g": 2.0},
    "naan": {"unit": "piece", "kcal": 262, "protein_g": 9.0, "carbs_g": 45.0, "fat_g": 5.0, "fiber_g": 2.0},
    "rice": {"unit": "bowl", "kcal": 200, "protein_g": 4.0, "carbs_g": 45.0, "fat_g": 0.4, "fiber_g": 0.6},
    "dal": {"unit": "bowl", "kcal": 180, "protein_g": 9.0, "carbs_g": 27.0, "fat_g": 4.0, "fiber_g": 6.0},
    "biryani": {"unit": "bowl", "kcal": 400, "protein_g": 15.0, "carbs_g": 55.0, "fat_g": 14.0, "fiber_g": 3.0},
    "sabzi": {"unit": "bowl", "kcal": 150, "protein_g": 4.0, "carbs_g": 15.0, "fat_g": 8.0, "fiber_g": 4.0},
    "chai": {"unit": "cup", "kcal": 60, "protein_g": 2.0, "carbs_g": 8.0, "fat_g": 2.0, "fiber_g": 0.0},
    "idli": {"unit": "piece", "kcal": 39, "protein_g": 1.5, "carbs_g": 8.0, "fat_g": 0.2, "fiber_g": 0.5},
    "dosa": {"unit": "piece", "kcal": 133, "protein_g": 3.0, "carbs_g": 20.0, "fat_g": 4.5, "fiber_g": 1.0},
    "poha": {"unit": "bowl", "kcal": 250, "protein_g": 5.0, "carbs_g": 40.0, "fat_g": 8.0, "fiber_g": 3.0},
    "chana": {"unit": "bowl", "kcal": 210, "protein_g": 10.0, "carbs_g": 30.0, "fat_g": 6.0, "fiber_g": 8.0},
    "paneer": {"unit": "serving", "kcal": 265, "protein_g": 18.0, "carbs_g": 6.0, "fat_g": 20.0, "fiber_g": 0.0},
    "samosa": {"unit": "piece", "kcal": 150, "protein_g": 3.0, "carbs_g": 17.0, "fat_g": 8.0, "fiber_g": 2.0},
    "upma": {"unit": "bowl", "kcal": 230, "protein_g": 6.0, "carbs_g": 33.0, "fat_g": 8.0, "fiber_g": 3.0},
    "sambar": {"unit": "bowl", "kcal": 120, "protein_g": 6.0, "carbs_g": 18.0, "fat_g": 3.0, "fiber_g": 4.0},
    "curd": {"unit": "bowl", "kcal": 98, "protein_g": 6.0, "carbs_g": 7.0, "fat_g": 5.0, "fiber_g": 0.0},
    "ghee": {"unit": "tsp", "kcal": 45, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 5.0, "fiber_g": 0.0},
    "rajma": {"unit": "bowl", "kcal": 220, "protein_g": 10.0, "carbs_g": 30.0, "fat_g": 6.0, "fiber_g": 9.0},
    "aloo_sabzi": {"unit": "bowl", "kcal": 180, "protein_g": 3.0, "carbs_g": 25.0, "fat_g": 8.0, "fiber_g": 3.0},
    "palak_paneer": {"unit": "bowl", "kcal": 280, "protein_g": 12.0, "carbs_g": 12.0, "fat_g": 20.0, "fiber_g": 4.0},
    "kheer": {"unit": "bowl", "kcal": 200, "protein_g": 5.0, "carbs_g": 30.0, "fat_g": 6.0, "fiber_g": 0.5},
    "dhokla": {"unit": "piece", "kcal": 65, "protein_g": 2.0, "carbs_g": 10.0, "fat_g": 2.0, "fiber_g": 1.0},
    "vada": {"unit": "piece", "kcal": 97, "protein_g": 3.0, "carbs_g": 10.0, "fat_g": 5.0, "fiber_g": 1.5},
    "pulao": {"unit": "bowl", "kcal": 280, "protein_g": 6.0, "carbs_g": 45.0, "fat_g": 8.0, "fiber_g": 2.0},
    "khichdi": {"unit": "bowl", "kcal": 220, "protein_g": 8.0, "carbs_g": 35.0, "fat_g": 5.0, "fiber_g": 3.0},
    "papad": {"unit": "piece", "kcal": 35, "protein_g": 2.0, "carbs_g": 6.0, "fat_g": 0.5, "fiber_g": 0.5},
    "lassi": {"unit": "glass", "kcal": 180, "protein_g": 5.0, "carbs_g": 25.0, "fat_g": 6.0, "fiber_g": 0.0},
    "buttermilk": {"unit": "glass", "kcal": 40, "protein_g": 2.0, "carbs_g": 4.0, "fat_g": 1.5, "fiber_g": 0.0},
    "egg_curry": {"unit": "bowl", "kcal": 220, "protein_g": 12.0, "carbs_g": 8.0, "fat_g": 16.0, "fiber_g": 2.0},
    "pav_bhaji": {"unit": "plate", "kcal": 400, "protein_g": 8.0, "carbs_g": 55.0, "fat_g": 16.0, "fiber_g": 6.0},
    # --- common Western ---
    "egg": {"unit": "piece", "kcal": 78, "protein_g": 6.0, "carbs_g": 0.6, "fat_g": 5.0, "fiber_g": 0.0},
    "toast": {"unit": "slice", "kcal": 120, "protein_g": 3.0, "carbs_g": 15.0, "fat_g": 5.0, "fiber_g": 1.0},
    "oatmeal": {"unit": "bowl", "kcal": 150, "protein_g": 5.0, "carbs_g": 27.0, "fat_g": 3.0, "fiber_g": 4.0},
    "chicken_breast": {"unit": "100g", "kcal": 165, "protein_g": 31.0, "carbs_g": 0.0, "fat_g": 3.6, "fiber_g": 0.0},
    "salad": {"unit": "bowl", "kcal": 100, "protein_g": 3.0, "carbs_g": 12.0, "fat_g": 5.0, "fiber_g": 4.0},
    "sandwich": {"unit": "piece", "kcal": 250, "protein_g": 8.0, "carbs_g": 35.0, "fat_g": 9.0, "fiber_g": 3.0},
    "banana": {"unit": "piece", "kcal": 105, "protein_g": 1.3, "carbs_g": 27.0, "fat_g": 0.4, "fiber_g": 3.1},
    "apple": {"unit": "piece", "kcal": 95, "protein_g": 0.5, "carbs_g": 25.0, "fat_g": 0.3, "fiber_g": 4.4},
    "coffee": {"unit": "cup", "kcal": 40, "protein_g": 1.5, "carbs_g": 5.0, "fat_g": 1.5, "fiber_g": 0.0},
    "milk": {"unit": "glass", "kcal": 150, "protein_g": 8.0, "carbs_g": 12.0, "fat_g": 8.0, "fiber_g": 0.0},
    "yogurt": {"unit": "cup", "kcal": 150, "protein_g": 8.0, "carbs_g": 11.0, "fat_g": 8.0, "fiber_g": 0.0},
    "pizza": {"unit": "slice", "kcal": 285, "protein_g": 12.0, "carbs_g": 36.0, "fat_g": 10.0, "fiber_g": 2.5},
    "burger": {"unit": "piece", "kcal": 350, "protein_g": 17.0, "carbs_g": 33.0, "fat_g": 17.0, "fiber_g": 2.0},
    "pasta": {"unit": "bowl", "kcal": 350, "protein_g": 12.0, "carbs_g": 55.0, "fat_g": 9.0, "fiber_g": 3.0},
    "cereal": {"unit": "bowl", "kcal": 200, "protein_g": 6.0, "carbs_g": 35.0, "fat_g": 4.0, "fiber_g": 2.0},
    "avocado": {"unit": "piece", "kcal": 240, "protein_g": 3.0, "carbs_g": 12.0, "fat_g": 22.0, "fiber_g": 10.0},
    "bacon": {"unit": "serving", "kcal": 90, "protein_g": 6.0, "carbs_g": 0.3, "fat_g": 7.0, "fiber_g": 0.0},
    "cheese": {"unit": "slice", "kcal": 100, "protein_g": 6.0, "carbs_g": 1.0, "fat_g": 8.0, "fiber_g": 0.0},
    "orange": {"unit": "piece", "kcal": 62, "protein_g": 1.2, "carbs_g": 15.0, "fat_g": 0.2, "fiber_g": 3.0},
    "almonds": {"unit": "handful", "kcal": 164, "protein_g": 6.0, "carbs_g": 6.0, "fat_g": 14.0, "fiber_g": 3.5},
    "peanut_butter": {"unit": "tbsp", "kcal": 95, "protein_g": 4.0, "carbs_g": 3.0, "fat_g": 8.0, "fiber_g": 1.0},
    "bread": {"unit": "slice", "kcal": 80, "protein_g": 3.0, "carbs_g": 14.0, "fat_g": 1.0, "fiber_g": 1.0},
    "butter": {"unit": "tsp", "kcal": 34, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 4.0, "fiber_g": 0.0},
    "orange_juice": {"unit": "glass", "kcal": 110, "protein_g": 2.0, "carbs_g": 26.0, "fat_g": 0.5, "fiber_g": 0.5},
    "smoothie": {"unit": "glass", "kcal": 220, "protein_g": 5.0, "carbs_g": 40.0, "fat_g": 5.0, "fiber_g": 3.0},
    "salmon": {"unit": "100g", "kcal": 208, "protein_g": 20.0, "carbs_g": 0.0, "fat_g": 13.0, "fiber_g": 0.0},
    "steak": {"unit": "100g", "kcal": 271, "protein_g": 25.0, "carbs_g": 0.0, "fat_g": 19.0, "fiber_g": 0.0},
    "fries": {"unit": "serving", "kcal": 365, "protein_g": 4.0, "carbs_g": 48.0, "fat_g": 17.0, "fiber_g": 4.0},
    "soup": {"unit": "bowl", "kcal": 120, "protein_g": 4.0, "carbs_g": 18.0, "fat_g": 3.0, "fiber_g": 3.0},
    "wrap": {"unit": "piece", "kcal": 350, "protein_g": 20.0, "carbs_g": 35.0, "fat_g": 14.0, "fiber_g": 3.0},
    "potato": {"unit": "piece", "kcal": 160, "protein_g": 4.0, "carbs_g": 37.0, "fat_g": 0.2, "fiber_g": 4.0},
}
