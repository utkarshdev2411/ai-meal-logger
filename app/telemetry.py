from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from app.config import get_settings
from app.db import repo
from app.db.client import async_session_factory

_pending: set[asyncio.Task] = set()
_cold_done = False


def db_backend_name(database_url: str | None = None) -> str:
    url = database_url or get_settings().database_url
    return "postgres" if url.startswith("postgresql") else "sqlite"


def consume_cold() -> bool:
    global _cold_done
    was_cold = not _cold_done
    _cold_done = True
    return was_cold


def reset_cold() -> None:
    global _cold_done
    _cold_done = False


def new_turn_id() -> str:
    return str(uuid.uuid4())


async def _write(payload: dict) -> None:
    try:
        async with async_session_factory() as session:
            await repo.insert_latency_sample(session, **payload)
    except Exception:
        pass


def _schedule(payload: dict) -> None:
    task = asyncio.create_task(_write(payload))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def flush() -> None:
    if _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)


@dataclass
class TurnTimer:
    turn_id: str
    path: str
    user_id: str | None = None
    fast_path: bool = False
    cold: bool = False
    db_backend: str = field(default_factory=db_backend_name)

    def record(self, phase: str, ms: int) -> None:
        _schedule(
            dict(
                turn_id=self.turn_id,
                path=self.path,
                phase=phase,
                ms=ms,
                db_backend=self.db_backend,
                user_id=self.user_id,
                fast_path=self.fast_path,
                cold=self.cold,
            )
        )

    @asynccontextmanager
    async def phase(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, int((time.perf_counter() - start) * 1000))
