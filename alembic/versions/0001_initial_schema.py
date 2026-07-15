"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-07-10

Creates all four ARIA tables:
  trades, signals, day_logs, session_notes
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trades",
        sa.Column("id",                 sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("ticket",             sa.Integer,     unique=True, nullable=False),
        sa.Column("pair",               sa.String(20),  nullable=False),
        sa.Column("direction",          sa.String(10),  nullable=False),
        sa.Column("lots",               sa.Float,       nullable=False),
        sa.Column("entry",              sa.Float,       nullable=False),
        sa.Column("sl",                 sa.Float,       nullable=False),
        sa.Column("tp1",                sa.Float,       nullable=False),
        sa.Column("tp2",                sa.Float,       nullable=False),
        sa.Column("tp3",                sa.Float,       nullable=False),
        sa.Column("close_price",        sa.Float,       nullable=True),
        sa.Column("pnl",                sa.Float,       nullable=True),
        sa.Column("score",              sa.Float,       default=0.0),
        sa.Column("reason",             sa.Text,        default=""),
        sa.Column("session",            sa.String(30),  default=""),
        sa.Column("status",             sa.String(20),  default="open"),
        sa.Column("opened_at",          sa.DateTime,    nullable=False),
        sa.Column("closed_at",          sa.DateTime,    nullable=True),
        sa.Column("risk_pct",           sa.Float,       nullable=True),
        sa.Column("spread_pips",        sa.Float,       nullable=True),
        sa.Column("regime",             sa.String(30),  nullable=True),
        sa.Column("strategy_version",   sa.String(50),  nullable=True),
        sa.Column("exit_reason_detail", sa.Text,        nullable=True),
        sa.Column("ml_score",           sa.Float,       nullable=True),
        sa.Column("slippage_pips",      sa.Float,       nullable=True),
        sa.Column("param_hash",         sa.String(64),  nullable=True),
        sa.Column("lessons_learned",    sa.Text,        nullable=True),
    )
    op.create_index("ix_trades_pair",   "trades",   ["pair"])
    op.create_index("ix_trades_ticket", "trades",   ["ticket"], unique=True)

    op.create_table(
        "signals",
        sa.Column("id",         sa.Integer,    primary_key=True, autoincrement=True),
        sa.Column("pair",       sa.String(20), nullable=False),
        sa.Column("direction",  sa.String(10), nullable=False),
        sa.Column("score",      sa.Float,      nullable=False),
        sa.Column("reason",     sa.Text,       default=""),
        sa.Column("executed",   sa.Boolean,    default=False),
        sa.Column("created_at", sa.DateTime,   nullable=False),
    )
    op.create_index("ix_signals_pair", "signals", ["pair"])

    op.create_table(
        "day_logs",
        sa.Column("id",                sa.Integer,   primary_key=True, autoincrement=True),
        sa.Column("date",              sa.String(10), unique=True, nullable=False),
        sa.Column("starting_balance",  sa.Float,     nullable=False),
        sa.Column("ending_balance",    sa.Float,     default=0.0),
        sa.Column("realized_pnl",      sa.Float,     default=0.0),
        sa.Column("trades_taken",      sa.Integer,   default=0),
        sa.Column("trades_won",        sa.Integer,   default=0),
        sa.Column("trades_lost",       sa.Integer,   default=0),
        sa.Column("profit_factor",     sa.Float,     default=0.0),
    )
    op.create_index("ix_day_logs_date", "day_logs", ["date"], unique=True)

    op.create_table(
        "session_notes",
        sa.Column("id",         sa.Integer,    primary_key=True, autoincrement=True),
        sa.Column("date",       sa.String(10), nullable=False),
        sa.Column("session",    sa.String(30), nullable=False),
        sa.Column("note_type",  sa.String(30), nullable=False),
        sa.Column("content",    sa.Text,       nullable=False),
        sa.Column("model_used", sa.String(50), default=""),
        sa.Column("created_at", sa.DateTime,   nullable=False),
    )
    op.create_index("ix_session_notes_date", "session_notes", ["date"])


def downgrade() -> None:
    op.drop_table("session_notes")
    op.drop_table("day_logs")
    op.drop_table("signals")
    op.drop_table("trades")
