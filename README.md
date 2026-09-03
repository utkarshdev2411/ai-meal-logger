# CalorAI Meal Logger

A conversational agent that logs meals the way people actually text about food —
half-sentences, corrections, photos, "same as yesterday" — built on LangGraph, with a
separate vision model, typed cross-session memory, and derived (never stored) daily
totals so a correction can never double-count.

Built for CalorAI's AI Engineer (Conversational Agents) test task.

---

## Table of contents

- [Setup](#setup)
- [Core features](#core-features)
- [Architecture at a glance](#architecture-at-a-glance)
- [Model choices](#model-choices)
- [How memory works](#how-memory-works)
- [Tool design](#tool-design)
- [Multi-turn ambiguity handling](#multi-turn-ambiguity-handling)
- [Latency](#latency)
- [Assumptions and trade-offs](#assumptions-and-trade-offs)
- [Test conversation set — coverage](#test-conversation-set--coverage)
- [Bonus features implemented](#bonus-features-implemented)
- [Time breakdown](#time-breakdown)
- [What I'd fix or build next](#what-id-fix-or-build-next)
- [AI tool usage](#ai-tool-usage)

---

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

`.env.example` ships working defaults for everything except the API key. Default
provider is Gemini's own OpenAI-compatible endpoint (free, no card — get a key at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)); any other
OpenAI-compatible endpoint works by changing `LLM_BASE_URL` alone. Default database is
a local SQLite file — nothing else to install or provision.

**Optional — a key pool.** `LLM_API_KEYS_EXTRA` accepts additional comma-separated
keys, round-robin-ed with `LLM_API_KEY`. Free-tier Gemini keys carry per-model daily
quotas; a small pool buys real headroom without adding billing. Every key in the pool
is verified independently, not just the primary — see below.

### Verify model access before relying on it

```bash
python scripts/check_models.py
```

Free-tier model IDs get renamed, retired, and rate-limited without notice — this
script proved its worth twice during the build (see [Assumptions and
trade-offs](#assumptions-and-trade-offs)). It pings every key in the pool against all
three model roles, sending a real test image for the vision check, and exits non-zero
if any key/role combination is unreachable.

### Run it

```bash
python -m app.cli                  # terminal chat, accepts --image PATH
# or
python -m uvicorn app.api:app      # FastAPI + SSE + minimal web chat at :8000
```

### Tests, evals, and the offline verification suite

```bash
pytest                             # 36 tests, fully offline, no API key needed
python evals/run_evals.py          # 14 cases (11 brief messages + 3 graded regressions)
python scripts/verify_db.py        # + 4 more verify_*.py scripts, one per subsystem
python scripts/bench.py            # latency harness, scripted LLM by default
python scripts/bench.py --real     # same harness against the real configured LLM
```

All of the above run with **no API key and no network** except `check_models.py` and
`bench.py --real`, which need a live key on purpose.

---

## Core features

| # | Feature | Where |
|---|---------|-------|
| 1 | Conversational agent, tool calling | `app/agent/graph.py` (LangGraph), `app/agent/tools/` |
| 2 | Persistent database | `app/db/` — SQLAlchemy 2.0 async ORM, SQLite |
| 3 | Running daily totals, correct through edits | `app/db/repo.py::daily_totals` — always a SQL aggregate, never stored |
| 4 | Image on a separate model | `app/vision/` — `gemini-3.1-flash-lite`, routed as a graph node, not a tool |
| 5 | Persistent memory | `app/memory/` — typed facts, ranked + budgeted retrieval |
| 6 | Multi-turn ambiguity handling | `app/agent/prompts.py` — log-with-assumption policy |

---

## Architecture at a glance

```
                    ┌──────────────────────┐
   inbound msg ──▶  │  prefetch  ∥  vision  │   asyncio.gather — image path only
                    └──────────┬────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  agent (text model) │◀──┐ tool loop
                    │  + 6 tools          │───┘
                    └──────────┬──────────┘
                               ▼
                       reply to user
                               │
                               ▼  (fire-and-forget, after the reply)
                    ┌─────────────────────┐
                    │  background memory  │
                    │  fact extraction    │
                    └─────────────────────┘
```

`prefetch` gathers today's totals, a recent-meals digest, and ranked memory facts into
one block appended to the system prompt — this is what makes query turns
("how am I doing on calories?") and "same as yesterday" cost **zero tool calls**,
which shows up directly in the latency numbers below.

---

## Model choices

**Every role — text, vision, background memory extraction — points at the same
model: `gemini-3.1-flash-lite`, called through a round-robin pool of API keys.**
This is a deliberate late-build pivot, not the original design, and I want to be
upfront about the story rather than present it as if it were the plan from hour one.

### What I originally did, and why it changed

The build started with three genuinely distinct models — a stronger tool-calling model
for text, a separate vision model, and a cheap model for background extraction — which
is the textbook shape for this task and the one I'd defend by default. Two real
problems forced a pivot:

1. **A confirmed, open upstream bug.** Gemini's 3.x model generation attaches a
   mandatory `thought_signature` to every function call, which the API requires
   echoed back on the next turn — `langchain-openai` (the generic OpenAI-compat
   client) silently drops that field when parsing the response, so the *second*
   turn of any tool-using conversation gets rejected with a 400
   (`langchain-ai/langchain#34056`, still open). This broke exactly the graded
   correction case, silently, and only on the second turn — not something a
   single-shot smoke test catches. Fixed by switching the text role's client to
   `langchain-google-genai`, which round-trips the signature correctly (verified
   directly: a scripted two-turn log→correct exchange completes cleanly, confirmed
   with a real conversation afterward — see the DB dump under
   [Assumptions and trade-offs](#assumptions-and-trade-offs)).
2. **Cost minimization, stated explicitly as a hard constraint by whoever holds the
   API keys for this project.** Once that constraint was set, I re-verified every
   candidate model **live** — not from documentation, which I'd already been burned
   by twice (see below) — against three separate API keys/projects: real auth, a
   real `log_meal`-shaped tool call (including correctly parsing "two thirds of the
   box" as `quantity: 0.67`), and a real test image, on every key. `gemini-3.1-flash-lite`
   was the only model confirmed working across all three keys, all three
   capabilities. Quota headroom then comes from the key pool
   (`app/config.py::next_api_key`, round-robin across `LLM_API_KEY` +
   `LLM_API_KEYS_EXTRA`), not from spreading roles across model tiers.

Vision stays architecturally separate from text regardless of this pivot — it's a
distinct **call site** (`app/vision/extract.py`, invoked from a graph *node*, never a
tool), with its own structured-only prompt and its own uncertainty schema. It happens
to share a model ID with text right now; it never shares a *conversation* with text,
which is the actual red flag the brief names ("everything through one model, including
images"). If cost weren't the binding constraint, the three-distinct-models shape is
what I'd ship instead, and the code makes that a one-line config change, not a
rewrite — `TEXT_MODEL` / `VISION_MODEL` / `EXTRACTOR_MODEL` are three independent
settings; `app/config.py` is the only file where a model ID may appear.

### The documentation-drift problem, twice

Free-tier model catalogs move fast enough that trusting search results or docs pages
directly produced two dead ends during this build:

- The first model IDs I picked (from web search, cross-checked against a live
  `/models` listing) were already retired by the time I tried them.
- `gemini-2.5-flash` and `gemini-2.5-flash-lite` are both listed in the `/models`
  catalog for every key used here, but 404 with *"no longer available to new
  users"* on every actual call, on both the OpenAI-compat and native endpoints —
  Google's catalog endpoint doesn't filter by per-project eligibility, so *listed*
  and *usable* turned out to be different questions.

`scripts/check_models.py` exists specifically because of this — every model claim
in this README was verified with a live API call, not sourced from a docs page.

### Vision handling

Vision runs through `app/vision/extract.py` with a strict pipeline: request `json_object`
mode (the model advertises `response_format` but not guaranteed schema enforcement),
validate the response against a Pydantic `VisionObservation` schema, tolerate
markdown-fenced JSON, retry once with a stricter instruction, then raise rather than
fabricate an observation. A raised error becomes `Image.status='failed'` +
`Image.error`, and the agent is instructed to *ask the user to describe the plate*
instead of guessing — never silently invents food items.

**Uncertainty is surfaced, not hidden.** The schema carries `confidence`,
`alternatives`, and `unclear` per item; the agent hedges in its reply
("...and what I think is bhindi — correct me if that's off") and logs its best guess
anyway rather than blocking on a clarifying question, which would break the
messaging-speed feel the brief asks for.

**Photo + caption is one graph turn, not two.** Image presence routes deterministically
into a `vision` node that runs in parallel with `prefetch`; the resulting
`VisionObservation` is injected into the *same* agent turn as the caption text. There
is structurally one `log_meal` call per inbound message regardless of modality — this
is what prevents `[photo] "half of this was my brother's"` from logging two meals, and
it's covered by an eval case and a dedicated regression check in `scripts/verify_vision.py`.

---

## How memory works

**Three layers, kept physically separate on purpose** — the brief calls out "memory
that's just conversation history stuffed into the prompt" as a red flag, so the
separation has to be real in the code, not just described in prose.

| Layer | Lives in | Survives sessions? | Purpose |
|---|---|---|---|
| Working state | LangGraph checkpointer | No | `last_meal_id`, in-thread scratch state |
| **Durable facts** | `memories` table | **Yes** | diet, goals, routines, aliases, preferences |
| Meal history | `meals` / `meal_items` tables | Yes | queried, never dumped into the prompt |

### What gets stored

Typed `kind`s (`diet`, `goal`, `routine`, `alias`, `preference`, `dislike`), each keyed
`(user_id, kind, key)` with a **database-level partial unique index on the active
row** — `WHERE status='active'` — so a conflicting fact supersedes rather than
duplicates, enforced by the schema, not application etiquette.

**Explicitly never stored:** one-off meals (that's what `meals` is for), transient
states ("I'm full"), anything derivable from meal history, raw message text.

### When it's written

Two distinct paths, on purpose:

1. **Explicit** — the `remember` tool, called when the user plainly states something
   durable ("i'm vegetarian btw"), confidence 1.0, acknowledged in the reply so the
   user sees it land.
2. **Background** — `app/memory/extractor.py` runs *after* the reply has streamed,
   via `asyncio.create_task` on its own DB session, proposing candidate facts with a
   confidence floor. It can never add latency to a turn, and a failure there can
   never break the turn (caught, logged, swallowed).

### How it's retrieved

`app/memory/store.py::retrieve_memories` ranks active facts by
`kind_priority × recency × use_count`, and renders only as many as fit
`MEMORY_TOKEN_BUDGET` (300 tokens, configurable) — a hard ceiling regardless of how
many facts a user accumulates. This block is fetched **in parallel** with today's
totals and a recent-meals digest (`app/agent/prefetch.py`, one `asyncio.gather`), and
appended to the system prompt *after* the static instructions, so the cacheable
prefix never shifts.

**Verified live, not just asserted:** a real conversation through the CLI
(`i'm vegetarian btw` → `remember` tool call) produced a real DB row
(`diet | dietary_preference | {"text": "vegetarian"}`) that a *fresh process*, on a
later turn, correctly retrieved and referenced — the actual proof that this is memory,
not conversation history.

---

## Tool design

Six tools, deliberately split so no two can plausibly serve the same intent —
overlapping tools are what makes an agent pick wrong.

| Tool | Does | Why it's a separate tool |
|---|---|---|
| `log_meal` | Creates a new meal | Never touches existing rows — this boundary is what makes double-counting structurally impossible |
| `revise_meal` | Edits/deletes an *existing* meal (`meal_ref="last"` or an id) | All mutation lives here. A correction is always an UPDATE via `repo.replace_meal_items`, never a new `log_meal` call |
| `get_daily_totals` | Reads totals for a day | Read-only, no side effects — but redundant on *today*, since prefetch already answers it (see [Latency](#latency)) |
| `search_meals` | Reads past meals by date | Answers "same as yesterday" when the date falls outside the prefetched 2-day digest |
| `remember` | Explicit durable-fact write | Confidence 1.0, distinct from the background extractor's proposed facts |
| `recall` | Long-tail fact lookup | For anything not already in the prefetched memory block |

**Why `log_meal` and `revise_meal` are separate, specifically:** this is the single
tool-boundary decision the graded correction case hinges on. If editing lived inside
`log_meal`, "actually that was 3 rotis not 2" becomes ambiguous at the tool-selection
level — the model has to *infer* it's an edit. Splitting them makes the choice
structural: the system prompt states plainly that a correction is always
`revise_meal`, never a second `log_meal`, and the tools' own descriptions reinforce it.

`log_meal` and `revise_meal` both return the freshly-computed daily totals in their
result — the agent never needs a follow-up `get_daily_totals` call just to tell the
user their new total, which is the main lever behind the log/correct latency numbers.

---

## Multi-turn ambiguity handling

**Default: log with a stated assumption, not a question.** The assumption is spoken in
the reply ("assumed a medium bowl") so it's correctable in one message — a
non-blocking correction is cheaper than a blocking question, and matches the
"texting a friend" feel more than an interrogation would.

**Ask only when** the unknown could swing the total by roughly 30%+ *and* no
reasonable default exists. At most one question per turn. Never ask something the
prefetched context already answers.

| Message | Behavior |
|---|---|
| `"leftover biryani, maybe two thirds of the box"` | Logs with a stated assumption (a standard takeaway box), no question |
| `"skipped lunch but grazed all afternoon"` | No default exists — the one case where a clarifying question is warranted |
| `"same as yesterday"` | Reads yesterday's meal from the prefetched digest, logs silently if unambiguous |
| `"my usual"` | Hits a `routine`/`alias` memory fact if one exists; asks once and remembers the answer if not |
| `"actually that was 3 rotis not 2"` | Always `revise_meal`, confirms the *new* total so the user can see it changed, not doubled |

---

## Latency

**Required reading first — there is no numeric SLA in the brief.** It asks for
measurement, reasoning, and honesty about what couldn't be fixed: *"We care more about
the reasoning than the number."* The numbers below are that measurement; the reasoning
follows.

### Real numbers — `gemini-3.1-flash-lite`, real key pool, real network, `n=8` per case

Measured with `python scripts/bench.py --real --runs 8` against the live agent/DB/vision
stack — not a scripted stand-in. Zero failures across 40 real API turns.

| Case | TTFT / total p50 | p95 |
|---|---|---|
| Log intent (text) | 2.80 s | 3.59 s |
| **Query intent (text)** | **1.76 s** | 3.22 s |
| Correction intent (text) | 3.79 s | 6.44 s |
| Photo + caption, prewarmed | 5.12 s | 9.94 s |
| Photo + caption, cold | 5.19 s | 8.20 s |

Phase breakdown (text and image paths):

| Path | Phase | p50 | p95 |
|---|---|---|---|
| text | prefetch | 9 ms | 10 ms |
| text | llm (decide) | 1.67 s | 3.62 s |
| text | tool exec | 9 ms | 10 ms |
| text | llm (reply) | 1.27 s | 2.49 s |
| image | prefetch | 16 ms | 25 ms |
| image | vision extraction | 742 ms | 3.21 s |
| image | llm (decide) | 2.02 s | 4.27 s |
| image | llm (reply) | 1.62 s | 2.70 s |

**What the numbers say, and what I did about it:**

- **Query intent is the clear standout at 1.76 s p50 — because it's the only case
  costing exactly one LLM call, not two.** `get_daily_totals` and `search_meals` are
  both structurally unnecessary for it: prefetch already put today's totals in the
  system prompt. This is the single biggest lever in the whole design — every log/
  correct/image case pays for a decide-call *and* a reply-call, sequentially,
  because tool results have to come back before the model can compose a sentence
  about them.
- **DB and prefetch cost nothing.** 9–25 ms, against 1.2–2.0 *seconds* per LLM call —
  confirms the design bet that the number of sequential LLM round trips is the only
  latency term worth optimizing, and everything else (SQLite, parallel prefetch,
  in-process nutrition lookups) is genuinely free by comparison.
- **What was cut/parallelized/cached, concretely:** vision runs as a graph node in
  `asyncio.gather` with prefetch, not a tool (saves a full round trip on every image);
  `log_meal`/`revise_meal` return fresh totals inline (saves the third call on every
  log/correct turn); replies are capped at 80 tokens (`REPLY_MAX_TOKENS`) since
  decode time scales with output length and a friend texting back writes two
  sentences, not a paragraph; the local nutrition table resolves common foods with
  zero network calls.

### Honest gaps

- **Image p95 (8.2–9.9 s) sits right at the brief's own bad example** — *"not after a
  ten-second agent loop."* I did not fully close this gap. It is structurally two
  sequential LLM calls plus a vision call, on a free-tier model with meaningful
  latency variance; the phase breakdown shows exactly where the time goes rather than
  hiding it in one aggregate number.
- **Prewarming is proven correct but doesn't show up in the case totals above**, and I
  want to be precise about why rather than let the numbers imply the mechanism
  doesn't work. Isolated directly, outside the noisy aggregate: a cache-hit vision
  lookup takes **1.7 ms**; a real vision call takes **2,285 ms**. The mechanism saves
  exactly what it should. At `n=8`, the two real LLM calls in an image turn (each
  ranging 700 ms–4 s+) have more run-to-run variance than the vision call itself, so
  the case-level *totals* for prewarmed vs. cold land almost on top of each other —
  the win is real, it's just swamped in the aggregate at this sample size. The
  correct read is: **the image path's bottleneck is the LLM tool-loop, not vision
  extraction** — prewarming should be judged on the `vision` phase specifically
  (where the 1.7 ms vs. 2,285 ms gap is unambiguous), not the total.
- **Free-tier model latency has real variance** (p95 running 1.4–2.4× p50 across every
  case) that I could not fully control — it's provider-side queuing/variance on a
  free-tier key, not something client-side caching or parallelism reaches.

---

## Assumptions and trade-offs

- **No stored running total, anywhere in the schema.** Daily totals are always a SQL
  aggregate (`SUM(kcal) WHERE status='active'`) over `meal_items`, computed fresh on
  every read. A correction is structurally an UPDATE via `repo.replace_meal_items`,
  never an INSERT — double-counting isn't handled carefully, it's made impossible by
  there being no counter to increment twice. Verified with a property-style test
  (log → correct → delete → re-log, asserting totals at every step) written *before*
  the agent existed, so agent bugs could never be confused with data-layer bugs.
- **SQLite only, Postgres portable but not tested.** The ORM uses only portable types
  (`String(36)` PKs, `JSON`, UTC-aware `DateTime`, a partial unique index expressed
  once via `sqlite_where=`/`postgresql_where=`) so `DATABASE_URL` pointed at Postgres
  should work unchanged — but that path isn't independently verified in this build.
  Documented as a config-swap capability, not a proven one.
- **Nutrition data is a hybrid table + LLM fallback, not a real API.** ~60 hardcoded
  common items (Indian staples + Western basics) resolve with zero network calls;
  anything else batches into a single LLM call per turn (never per-item) and degrades
  to a flagged low-confidence estimate rather than raising. Per the brief's own FAQ,
  nutrition accuracy isn't what's being evaluated here — the boundary
  (`app/nutrition/resolve.py`) is where a real API (USDA/Nutritionix) would slot in
  without touching any caller.
- **All three model roles share one model ID, for cost reasons, not architecture
  ones** — covered in full under [Model choices](#model-choices). This is the one
  trade-off I'd most want to reverse with a larger budget.
- **Agent-level tool-call reliability on the free-tier lite model is not perfect.**
  Investigated directly rather than assumed: running the identical 4-turn scripted
  conversation (log → correct → query → remember) through the real graph, 4 separate
  times, produced 2 fully correct runs, 1 run where a correction silently didn't
  apply, and one live-CLI run that produced a genuine duplicate meal row. Root-caused
  by inspecting DB state and full message history directly, not by trusting console
  output — the conclusion: **the data-layer invariant never broke in any run** (every
  revision that *did* fire was a real UPDATE, never a duplicate INSERT); what's
  inconsistent is the small free-tier model occasionally choosing the wrong action
  (or none) on an ambiguous turn. This is a real, known limitation of the
  cost-minimized model choice above, not a code defect, and I'm stating it plainly
  rather than presenting cherry-picked clean runs. The brief's own FAQ explicitly
  says accuracy isn't the focus here — I'm treating this the same way: named, not
  hidden, not chased to perfection at the cost of the actual deliverables.
- **A small, separately-observed memory key mismatch:** the `remember` tool and the
  background extractor independently chose different `key`s for the same
  conceptual fact (`diet`/`diet` vs. `diet`/`dietary_preference`) in one live run, so
  the partial-unique-index supersede logic didn't catch a duplicate it structurally
  could have. Noted under [What I'd fix next](#what-id-fix-or-build-next) — a
  canonical-key normalization step, not a schema change.
- **No auth, per the brief.** Session isolation is `X-User-Id` header → `users.external_id`,
  nothing more.
- **`create_all()` instead of Alembic migrations.** A deliberate cut for a time-boxed
  build; the first thing to change for anything beyond this project.

---

## Test conversation set — coverage

All 11 brief messages plus the 3 regression cases run in `evals/run_evals.py`
(14/14 passing), asserting **tool calls and resulting DB state, never response
text** — prose is too unstable to assert on, so correctness is judged the same way a
human reviewer would judge it: did the right thing happen to the data.

| Message | Handling |
|---|---|
| "had 2 parathas and chai for breakfast" | `log_meal`, stated portion assumption |
| "leftover biryani, maybe two thirds of the box" | `log_meal` with quantity 0.67, no question asked |
| "skipped lunch but grazed all afternoon" | No default exists — one clarifying question |
| "same as yesterday" | Resolved from the prefetched recent-meals digest, zero `search_meals` calls |
| **"actually that was 3 rotis not 2"** | `revise_meal`, meal row count stays 1 — the graded regression |
| "how much protein have I had today?" | Zero tool calls, answered from prefetch |
| "how am I doing on calories?" | Zero tool calls, answered from prefetch |
| [photo of a plate] | Vision node → structured observation → one `log_meal` |
| **[photo] "half of this was my brother's"** | Caption + observation in one agent turn → exactly one `log_meal` at half portions |
| "my usual" | Routine/alias memory fact if present, else one question then remembered |
| "i'm vegetarian btw" | `remember` tool, no meal logged |

---

## Bonus features implemented

- **Eval set** — `evals/cases.yaml` + `evals/run_evals.py`, 14 cases, tool-call + DB-state
  assertions, verified to "have teeth" by deliberately reintroducing the double-count
  bug and confirming the suite catches it.
- **Session isolation** — `X-User-Id` header → isolated user, meals, memory, totals.
- **Streaming** — SSE on `POST /chat`; a `meal_logged` structured event fires the
  instant the tool returns, before the prose finishes streaming, so the UI can render
  the logged meal and updated totals without waiting for the full sentence.
- **Image prewarm** — `POST /upload` kicks off vision extraction in the background
  before the chat turn arrives, cached via `Image.status`; proven to cut real vision
  latency from ~2.3 s to ~2 ms on a cache hit.

**Not implemented:** LangSmith tracing (env-gated and stubbed for, never wired up —
would be the first bonus item with more time), the experimental single-call
`FAST_PATH` mode (flagged in code as the highest-risk optimization, deliberately never
enabled by default).

---

## Time breakdown

Approximate, grouped by activity rather than clock-punched — this was an iterative,
AI-paired build with real debugging and pivot time that doesn't show up as commit
timestamps alone.

| Activity | Approx. time |
|---|---|
| Foundation: config, scaffold, preflight tooling | 0.5 h |
| Persistence layer: ORM, repo, portability rules | 0.75 h |
| Nutrition resolution: table + normalizer + fallback | 0.5 h |
| **Logging & totals engine + correctness test suite** | 1.0 h |
| Agent core: LangGraph, tools, CLI | 1.0 h |
| Memory: store, background extractor, retrieval | 1.0 h |
| Prefetch fan-out + ambiguity policy | 0.5 h |
| Vision: model, node routing, caption merge | 0.75 h |
| FastAPI + SSE + minimal chat UI | 0.5 h |
| Telemetry + latency bench harness | 0.75 h |
| Eval suite (11 messages + 3 regressions) | 0.5 h |
| **Provider debugging: dead model IDs, `thought_signature` bug root-causing, migrating off `langchain-openai`, three-key verification and round-robin pool, DB-verified reliability investigation** | 2.5 h |
| README, docs, final verification pass | 0.75 h |
| **Total** | **~11.5 h** |

Over the suggested 6–8 hour budget — mostly because of the provider-debugging block,
which was genuinely unplanned: three separate real-world failures (dead model
generation, an upstream library bug, a free-tier rate-limit wall) each needed live
API verification to root-cause rather than guesswork, and I chose to spend that time
rather than ship a config that silently didn't work. Per the brief's own FAQ — *"we
value honest time management over heroic overtime"* — I'm reporting the real number
rather than a rounded-down one.

---

## What I'd fix or build next

Ranked by what I'd actually do first with more time, not by rubric weight:

1. **Reconcile the memory key mismatch** — normalize `remember`/extractor keys through
   one canonical-key function so semantically identical facts always collide into a
   supersede, not a duplicate.
2. **Reduce agent-level non-determinism on the free-tier model** — either a stronger
   system-prompt instruction ("never call `log_meal` unless the user just described
   eating something new") or moving `TEXT_MODEL` specifically back to a larger model
   once cost isn't the binding constraint, since text is the role most likely to
   benefit from stronger instruction-following.
3. **A real Postgres verification pass** — the portability is designed-in but
   untested; this is the highest-value thing to prove before calling it production-ready.
4. **LangSmith tracing** — cheapest remaining bonus signal, already env-gated in config.
5. **A larger, higher-`n` real bench run** once billing (or a bigger key pool) removes
   free-tier rate-limit pressure, to get a tighter p95 on the image path specifically.
6. **The `FAST_PATH` single-call mode** — already flagged in code as the highest-risk
   optimization in the whole design (structured-output placeholder substitution
   instead of native tool-calling); worth trying now that the round-robin pool gives
   room to A/B it safely, but deliberately shipped off.

---

## AI tool usage

Built with **Claude Code** (Claude Opus 5), used as an active engineering partner
rather than a code-completion tool, in three distinct modes:

1. **Planning documents first, code second.** Before any implementation, I had Claude
   produce a `CONTEXT.md` (architecture + rationale), a `PHASES.md` (a 13-phase
   execution plan with numbered functional requirements and exit criteria), and a
   `SCHEMA.md` (the concrete data model) — then built strictly against those, phase by
   phase, so later work couldn't silently drift from earlier decisions. These are
   internal working documents (gitignored, not part of this submission), kept
   separate from this README on purpose: this file is the reviewer-facing summary,
   those were the build-time source of truth.
2. **Delegated, independently-verified phase execution.** Most phases were built by a
   fresh sub-agent with a scoped brief, then **independently re-verified** — re-running
   its test/verify scripts myself, reading the actual diff, and in one case
   deliberately reintroducing a bug to confirm the eval suite would catch it — rather
   than trusting a phase-complete report at face value. This caught real issues
   phase-reports claimed were fine: a broken checkpoint-serialization path, an
   illegal-concurrent-session bug in the prefetch fan-out, a vision failure state
   that leaked raw error payloads into the prompt.
3. **Live verification over documentation trust, especially for anything
   provider-related.** Every model ID, price, and capability claim in this README was
   checked against a live API call or the provider's official pricing page — not
   copied from a search result — after getting burned twice by stale model names
   early in the build. The `thought_signature` bug was root-caused by directly
   inspecting a parsed `AIMessage`'s `additional_kwargs` (confirming the field was
   genuinely absent, not just unused) rather than assuming a library issue from the
   error message alone; the flash-lite reliability finding was root-caused by
   re-running the exact same conversation multiple times against the real API and
   diffing DB state, not by re-reading logs.

Net effect on speed and judgment: the phase-plan-first approach meant zero
architecture rewrites mid-build despite three real provider pivots; the
independent-verification habit caught bugs that would otherwise have shipped
silently; and the live-verification discipline is directly why this README's claims
are backed by real evidence rather than assumed correct.
