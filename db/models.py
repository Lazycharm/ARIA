"""SQLAlchemy models for ARIA."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id:          Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket:      Mapped[int]   = mapped_column(Integer, unique=True, index=True)
    pair:        Mapped[str]   = mapped_column(String(20), index=True)
    direction:   Mapped[str]   = mapped_column(String(10))     # long | short
    lots:        Mapped[float] = mapped_column(Float)
    entry:       Mapped[float] = mapped_column(Float)
    sl:          Mapped[float] = mapped_column(Float)
    tp1:         Mapped[float] = mapped_column(Float)
    tp2:         Mapped[float] = mapped_column(Float)
    tp3:         Mapped[float] = mapped_column(Float)
    close_price:       Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl:               Mapped[float | None] = mapped_column(Float, nullable=True)
    score:             Mapped[float] = mapped_column(Float, default=0.0)
    reason:            Mapped[str]   = mapped_column(Text, default="")
    session:           Mapped[str]   = mapped_column(String(30), default="")
    status:            Mapped[str]   = mapped_column(String(20), default="open")  # open | closed
    opened_at:         Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at:         Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Enhanced fields (added 2026-07-09)
    risk_pct:          Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_pips:       Mapped[float | None] = mapped_column(Float, nullable=True)
    regime:            Mapped[str | None]   = mapped_column(String(30), nullable=True)
    strategy_version:  Mapped[str | None]   = mapped_column(String(50), nullable=True)
    exit_reason_detail: Mapped[str | None]  = mapped_column(Text, nullable=True)
    ml_score:          Mapped[float | None] = mapped_column(Float, nullable=True)
    # Extended fields (added 2026-07-09 batch 2)
    slippage_pips:     Mapped[float | None] = mapped_column(Float, nullable=True)
    param_hash:        Mapped[str | None]   = mapped_column(String(64), nullable=True)
    lessons_learned:   Mapped[str | None]   = mapped_column(Text, nullable=True)


class Signal(Base):
    __tablename__ = "signals"

    id:        Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair:      Mapped[str]   = mapped_column(String(20), index=True)
    direction: Mapped[str]   = mapped_column(String(10))
    score:     Mapped[float] = mapped_column(Float)
    reason:    Mapped[str]   = mapped_column(Text, default="")
    executed:  Mapped[bool]  = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DayLog(Base):
    __tablename__ = "day_logs"

    id:              Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    date:            Mapped[str]   = mapped_column(String(10), unique=True, index=True)
    starting_balance: Mapped[float] = mapped_column(Float)
    ending_balance:  Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl:    Mapped[float] = mapped_column(Float, default=0.0)
    trades_taken:    Mapped[int]   = mapped_column(Integer, default=0)
    trades_won:      Mapped[int]   = mapped_column(Integer, default=0)
    trades_lost:     Mapped[int]   = mapped_column(Integer, default=0)
    profit_factor:   Mapped[float] = mapped_column(Float, default=0.0)


class SessionNote(Base):
    __tablename__ = "session_notes"

    id:         Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    date:       Mapped[str]   = mapped_column(String(10), index=True)
    session:    Mapped[str]   = mapped_column(String(30))
    note_type:  Mapped[str]   = mapped_column(String(30))   # pre_session | daily_report | brief
    content:    Mapped[str]   = mapped_column(Text)
    model_used: Mapped[str]   = mapped_column(String(50), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
