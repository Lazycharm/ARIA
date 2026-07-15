"""SQLite session manager for ARIA."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from db.models import Base

_engine = create_engine(
    f"sqlite:///{settings.db_path}",
    connect_args={"check_same_thread": False},
    echo=False,
)
_SessionLocal = sessionmaker(bind=_engine, autoflush=True, autocommit=False)

# New columns to add if they don't exist yet (SQLite ALTER TABLE migration)
_TRADE_COLUMNS: list[tuple[str, str]] = [
    ("risk_pct",           "REAL"),
    ("spread_pips",        "REAL"),
    ("regime",             "TEXT"),
    ("strategy_version",   "TEXT"),
    ("exit_reason_detail", "TEXT"),
    ("ml_score",           "REAL"),
    ("slippage_pips",      "REAL"),
    ("param_hash",         "TEXT"),
    ("lessons_learned",    "TEXT"),
]


def _migrate_columns() -> None:
    """Add new columns to existing tables without Alembic. Safe to run every start."""
    with _engine.connect() as conn:
        existing = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(trades)")).fetchall()
        }
        for col_name, col_type in _TRADE_COLUMNS:
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}"))
        conn.commit()


def init_db() -> None:
    Base.metadata.create_all(bind=_engine)
    _migrate_columns()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
