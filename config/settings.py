"""ARIA — Central configuration with capital, strategy, and session settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # ── LLM (cost-aware routing) ──────────────────────────────────
    # Haiku: signals/briefs/pre-session → ~$0.06/month
    # Sonnet: daily report (once/day) → ~$1.50/month
    # Fable 5: NEVER scheduled; only manual "deep analysis" trigger
    anthropic_api_key: str = Field(default="")
    model_brief: str = "claude-haiku-4-5-20251001"          # routine commentary
    model_session_analysis: str = "claude-haiku-4-5-20251001"  # pre-session levels
    model_daily_report: str = "claude-sonnet-4-6"            # end-of-day report
    model_deep_analysis: str = "claude-fable-5"              # manual trigger only

    # ── MT5 ───────────────────────────────────────────────────────
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = "ICMarketsSC-Demo"
    mt5_enabled: bool = True

    # ── Capital ───────────────────────────────────────────────────
    account_balance: float = 1000.0
    risk_per_trade_pct: float = 1.0
    daily_target_pct: float = 5.0
    max_daily_loss_pct: float = 9.0
    max_concurrent_trades: int = 3
    max_trades_per_day: int = 5
    emergency_drawdown_pct: float = 15.0

    # ── Strategy ──────────────────────────────────────────────────
    strategy: str = "hybrid_smc"
    htf_bias_tf: str = "H4"
    entry_tf: str = "M15"
    exec_tf: str = "M5"

    # ── Pairs ─────────────────────────────────────────────────────
    watchlist: str = "EURUSD,GBPUSD,USDJPY,XAUUSD,GBPJPY,NAS100,AUDUSD"

    # ── Sessions (UTC hours) ───────────────────────────────────────
    asian_start: int = 0
    asian_end: int = 7
    london_start: int = 7
    london_end: int = 16
    ny_start: int = 12
    ny_end: int = 21
    active_sessions: str = "asian,london,ny"

    # ── Filters ───────────────────────────────────────────────────
    max_spread_pips: float = 2.5
    news_buffer_minutes: int = 20
    min_signal_score: float = 65.0
    news_filter_enabled: bool = True

    # ── Dashboard ─────────────────────────────────────────────────
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8050
    signal_refresh_seconds: int = 5

    # ── Database ──────────────────────────────────────────────────
    db_path: Path = Path("./db/aria.db")

    # ── Notifications ─────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Obsidian Vault (shared brain) ─────────────────────────────
    obsidian_vault_path: Path = Path("D:/A office/AyoubOS/02-Obsidian-Vault")
    obsidian_aria_folder: str = "03 Projects/ARIA"

    # ── API ───────────────────────────────────────────────────────
    # ── Redis (optional fast-path cache — falls back to parquet) ─────────────
    redis_url: Optional[str] = None    # e.g. redis://localhost:6379/0

    # ── API ──────────────────────────────────────────────────────────────────
    api_key: str = ""          # X-API-Key for FastAPI endpoints (empty = no auth)
    api_host: str = "0.0.0.0"
    api_port: int = 8051

    # ── System ────────────────────────────────────────────────────
    dry_run: bool = False
    log_level: str = "INFO"

    # ── Computed ──────────────────────────────────────────────────
    @computed_field
    @property
    def pairs(self) -> list[str]:
        return [p.strip() for p in self.watchlist.split(",") if p.strip()]

    @computed_field
    @property
    def risk_amount(self) -> float:
        return self.account_balance * self.risk_per_trade_pct / 100

    @computed_field
    @property
    def daily_target_amount(self) -> float:
        return self.account_balance * self.daily_target_pct / 100

    @computed_field
    @property
    def max_loss_amount(self) -> float:
        return self.account_balance * self.max_daily_loss_pct / 100

    @computed_field
    @property
    def emergency_loss_amount(self) -> float:
        return self.account_balance * self.emergency_drawdown_pct / 100

    @computed_field
    @property
    def obsidian_trades_path(self) -> Path:
        return self.obsidian_vault_path / self.obsidian_aria_folder / "Trades"

    @computed_field
    @property
    def obsidian_daily_path(self) -> Path:
        return self.obsidian_vault_path / "01 Daily Notes" / "Trading"

    @computed_field
    @property
    def obsidian_analysis_path(self) -> Path:
        return self.obsidian_vault_path / self.obsidian_aria_folder / "Analysis"

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)
        Path("./exports").mkdir(exist_ok=True)
        self.obsidian_trades_path.mkdir(parents=True, exist_ok=True)
        self.obsidian_daily_path.mkdir(parents=True, exist_ok=True)
        self.obsidian_analysis_path.mkdir(parents=True, exist_ok=True)
        # Research subfolders
        aria_root = self.obsidian_vault_path / self.obsidian_aria_folder
        for sub in (
            "Research", "Hypotheses", "Accepted Strategies", "Rejected Strategies",
            "WalkForward", "Optimization", "Pair Profiles", "Market Regimes",
            "Economic Events", "Lessons Learned", "Experiments",
            "Performance Reports", "MonteCarlo",
        ):
            (aria_root / sub).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
