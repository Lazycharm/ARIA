# ARIA — Architecture

**Autonomous Research & Intelligent Allocation**
A hybrid SMC + Trend trading system for MT5, with Plotly Dash dashboard and AI session analysis.

---

## System Flow

```
Every 5 min (rules-based, $0)
┌──────────────────────────────────────────────────────────┐
│  MT5 Feed → Candles (M5/M15/H1/H4/D1)                  │
│       ↓                                                  │
│  Indicators (EMA/RSI/MACD/ATR/ADX)                     │
│       ↓                                                  │
│  SMC Analysis (Order Blocks, FVGs, BOS)                 │
│       ↓                                                  │
│  MTF Bias (D1+H4+H1 direction vote)                     │
│       ↓                                                  │
│  Confluence Score (0-100)                               │
│       ↓                                                  │
│  Signal Store → Dashboard (live signals, chart, gauge)  │
│       ↓ (if score ≥ 80 AND --live AND session active)  │
│  OrderManager → CapitalManager → MT5 Execution          │
└──────────────────────────────────────────────────────────┘

Every 60 sec
  TradeLifecycle: partial exits at TP1, breakeven SL, trail at TP2

06:30 UTC daily (Haiku ~$0.002)
  SessionAnalyst.run_presession() → key levels → Obsidian vault

17:30 UTC daily (Sonnet ~$0.05)
  SessionAnalyst.run_daily_report() → P&L summary → Obsidian vault
```

---

## Module Map

```
ARIA/
├── main.py                 ← entry point (--live / --dash-only / --scan-now)
├── config/settings.py      ← Pydantic settings (balance, risk, pairs, models)
├── core/
│   ├── capital.py          ← CapitalManager (CRITICAL: all sizing + kill switches)
│   ├── session.py          ← SessionManager (which session are we in)
│   └── brain.py            ← LLM routing (Haiku/Sonnet/Fable5)
├── data/
│   ├── mt5_feed.py         ← MT5 data (candles, ticks, account, positions)
│   └── calendar.py         ← ForexFactory economic calendar scraper
├── analysis/
│   ├── indicators.py       ← pandas-ta: EMA/RSI/MACD/ATR/Bollinger/ADX
│   ├── smc.py              ← Order Blocks, FVGs, BOS, liquidity levels
│   ├── structure.py        ← Market structure: HH/HL/LH/LL
│   ├── mtf.py              ← Multi-timeframe bias (D1+H4+H1+M15 vote)
│   └── confluence.py       ← Score aggregator (0-100)
├── signals/
│   ├── scanner.py          ← 5-min loop; writes to global signal store
│   ├── entry.py            ← Converts signal → entry/SL/TP levels
│   └── filter.py           ← Session, spread, news filters
├── execution/
│   ├── mt5_client.py       ← MT5 order placement/modify/close
│   ├── order_manager.py    ← Capital gate → MT5 execution bridge
│   └── trade_lifecycle.py  ← Partial exits, trailing stop, breakeven
├── ai/
│   ├── prompts.py          ← All LLM prompt templates
│   └── session_analyst.py  ← Pre-session (Haiku) + daily report (Sonnet)
├── knowledge/
│   └── obsidian.py         ← Writes to shared Obsidian vault
├── dashboard/
│   └── app.py              ← Plotly Dash live dashboard
├── db/
│   ├── models.py           ← SQLAlchemy: Trade, Signal, DayLog, SessionNote
│   └── session.py          ← SQLite session manager
└── scheduler/
    └── tasks.py            ← APScheduler: 5-min scan, 60s lifecycle, daily AI
```

---

## AI Cost Model

| Task | Model | Frequency | Cost/Month |
|------|-------|-----------|------------|
| Signals | Rules only | Every 5 min | $0.00 |
| Pre-session brief | Haiku | Once/day | $0.06 |
| Daily report | Sonnet | Once/day | $1.50 |
| Deep analysis | Fable 5 | Manual only | $0 auto |
| **Total** | | | **~$1.56** |

---

## Confluence Scoring

| Factor | Max Pts |
|--------|---------|
| MTF alignment (D1+H4+H1 agree) | 35 |
| At SMC level (OB or FVG) | 20 |
| RSI in favorable zone | 15 |
| M15 EMA trend aligned | 15 |
| Active trading session | 10 |
| Clean spread | 5 |
| **Total** | **100** |

**Signal threshold: 65** — emitted to dashboard
**Execution threshold: 80** — eligible for auto-trade (requires --live)

---

## Trade Lifecycle

```
Stage 1: Entry → TP1 (watch, update P&L)
Stage 2: At TP1 (2R profit) → close 50%, move SL to entry (breakeven)
Stage 3: At TP2 (4R profit) → close 30%, activate trailing (0.5 × ATR)
Stage 4: Trailing active until TP3 or stop-out
```

---

## Capital Safety

- **1%** risk per trade (configurable)
- **3%** max daily loss → halt trading
- **5%** daily profit target → lock in gains, halt trading
- **5%** emergency equity drawdown → close all positions
- Max **3** concurrent trades
- Max **5** trades per day
- Correlation protection (no EURUSD + GBPUSD long simultaneously)

---

## Sessions (UTC)

| Session | Hours | Mode | Best Pairs |
|---------|-------|------|-----------|
| Asian | 00-07 | Setup/mark zones | USDJPY, AUDUSD |
| Pre-London | 06:30-07 | AI analysis (Haiku) | — |
| London Open | 07-10 | Active trading | EURUSD, GBPUSD, XAUUSD |
| London Mid | 10-12 | Manage only | — |
| London-NY Overlap | 12-16 | Active trading | All majors + NAS100 |
| NY Close | 17-18 | Close + report | — |
| Dead Zone | 18-00 | Idle, vault update | — |

---

## Obsidian Vault Integration

Uses the shared AyoubOS vault: `D:/A office/AyoubOS/02-Obsidian-Vault`

```
02-Obsidian-Vault/
├── 01 Daily Notes/Trading/YYYY-MM-DD.md     ← daily P&L log
└── 03 Projects/ARIA/
    ├── ARIA Overview.md                      ← project index
    ├── Trades/YYYY-MM-DD-PAIR-LONG.md       ← per-trade notes
    └── Analysis/YYYY-MM-DD-pre-session.md   ← session analysis
```

---

## Quickstart

```bash
# 1. Install
pip install -e .

# 2. Configure
cp .env.example .env
# Edit: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, ANTHROPIC_API_KEY

# 3. Run (dashboard + scanner, DRY RUN — no trades)
python main.py

# 4. Enable live execution (only after testing on demo)
python main.py --live

# 5. Dashboard only
python main.py --dash-only

# 6. Manual scan
python main.py --scan-now
```

Dashboard opens at: http://127.0.0.1:8050
