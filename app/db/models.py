"""SQLAlchemy 2.0 declarative models — built verbatim from docs/SCHEMA.md Part 1.

Portable by construction: String(36) PKs with Python-side uuid4 defaults, JSON
columns, UTC-aware DateTime, Numeric for macros. See SCHEMA.md §1.11.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    tz: Mapped[str] = mapped_column(String(64), nullable=False, default="Asia/Kolkata")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    meals: Mapped[list["Meal"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    memories: Mapped[list["Memory"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Meal(Base):
    __tablename__ = "meals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    logged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    meal_slot: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500))
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    confidence: Mapped[float | None] = mapped_column(Numeric(3, 2))
    raw_input: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped["User"] = relationship(back_populates="meals")
    items: Mapped[list["MealItem"]] = relationship(back_populates="meal", cascade="all, delete-orphan")
    revisions: Mapped[list["MealRevision"]] = relationship(back_populates="meal", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_meals_user_date_status", "user_id", "local_date", "status"),
        Index("ix_meals_user_created", "user_id", "created_at"),
    )


class MealItem(Base):
    __tablename__ = "meal_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    meal_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("meals.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    canonical_key: Mapped[str | None] = mapped_column(String(120))
    quantity: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    unit: Mapped[str] = mapped_column(String(32), nullable=False)
    # Already multiplied by quantity — see SCHEMA.md §1.3.
    kcal: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    protein_g: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    carbs_g: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    fat_g: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    fiber_g: Mapped[float | None] = mapped_column(Numeric(8, 2))
    nutrition_source: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric(3, 2))

    meal: Mapped["Meal"] = relationship(back_populates="items")

    __table_args__ = (Index("ix_meal_items_meal", "meal_id"),)


class MealRevision(Base):
    __tablename__ = "meal_revisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    meal_id: Mapped[str] = mapped_column(String(36), ForeignKey("meals.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    before: Mapped[dict | list | None] = mapped_column(JSON)
    after: Mapped[dict | list | None] = mapped_column(JSON)
    reason: Mapped[str | None] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    meal: Mapped["Meal"] = relationship(back_populates="revisions")

    __table_args__ = (Index("ix_meal_revisions_meal", "meal_id", "created_at"),)


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[dict | list] = mapped_column(JSON, nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False)
    source_message: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    use_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    superseded_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("memories.id"))
    embedding: Mapped[dict | list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped["User"] = relationship(back_populates="memories")

    __table_args__ = (
        # Makes supersede-not-duplicate a DB guarantee, not app etiquette — SCHEMA.md §1.5.
        Index(
            "uq_memories_active",
            "user_id",
            "kind",
            "key",
            unique=True,
            sqlite_where=text("status = 'active'"),
            postgresql_where=text("status = 'active'"),
        ),
        Index("ix_memories_user_status_kind", "user_id", "status", "kind"),
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    image_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("images.id"))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("ix_messages_thread", "user_id", "thread_id", "created_at"),)


class Image(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    mime: Mapped[str | None] = mapped_column(String(64))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    bytes: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    observation: Mapped[dict | list | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_images_user_created", "user_id", "created_at"),)


class LatencySample(Base):
    __tablename__ = "latency_samples"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    turn_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"))
    path: Mapped[str] = mapped_column(String(16), nullable=False)
    phase: Mapped[str] = mapped_column(String(16), nullable=False)
    ms: Mapped[int] = mapped_column(Integer, nullable=False)
    fast_path: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    db_backend: Mapped[str] = mapped_column(String(16), nullable=False)
    cold: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_latency_path_phase", "path", "phase"),
        Index("ix_latency_turn", "turn_id"),
    )
