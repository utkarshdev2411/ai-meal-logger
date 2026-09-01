"""Nutrition resolution: free-text food items → macros.

Layering: a domain service. Callable by `app/agent` tools, never the reverse
(see docs/SCHEMA.md Part 2). Imports nothing from `app/agent`, `app/db`,
`app/memory`, or `app/vision`.
"""
