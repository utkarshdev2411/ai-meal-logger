# CalorAI Meal Logger

A conversational meal-logging agent — you text it what you ate, in half-sentences
or photos, and it logs it, keeps your running daily totals correct through
corrections, and remembers the things about you that matter.

> Work in progress. This README covers setup only; the full write-up (model
> choices, memory design, tool boundaries, measured latency) lands with the
> finished build.

## Setup

Requires Python 3.11+.

```bash
git clone <repo-url>
cd ai-meal-logger

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then put a real key in LLM_API_KEY
```

`.env.example` ships working defaults for everything except the API key, and
documents the trade-off each setting controls. The default provider is
OpenRouter (any OpenAI-compatible endpoint works — change `LLM_BASE_URL`), and
the default database is a local SQLite file, so there is nothing else to
install or provision.

### Verify model access

The default model IDs are free-tier, and free-tier IDs get renamed, retired and
rate-limited without notice. Run the preflight before relying on them:

```bash
python scripts/check_models.py
```

It pings all three model roles (text, vision, extractor), prints per-role
latency and ok/fail, and exits non-zero if any role is unreachable. If a role
fails, put a different model ID in `.env` and run it again.

### Tests

```bash
pytest
```

The suite runs offline and needs no API key.
