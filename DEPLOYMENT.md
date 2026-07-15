# ARIA — Deployment Guide

Autonomous FX Trading Bot — MT5 + Plotly Dash Dashboard.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.12+ | (3.14 works on Windows) |
| MetaTrader 5 | Must be running and logged in |
| Exness demo/live account | Demo: server `Exness-MT5Trial9` |
| Anthropic API Key | For pre-session briefs + daily reports |
| Telegram Bot Token | For trade alerts |

---

## 1. Windows (Primary Platform)

MT5 Python API only works natively on Windows.

```powershell
cd "D:\A office\AyoubOS\03 Projects\ARIA"

# Create venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install
pip install -e ".[all]"
# or:
pip install -r requirements.txt

# Configure
copy .env.example .env
# Fill in: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, ANTHROPIC_API_KEY, TELEGRAM_*

# Run in DRY RUN mode (no real trades)
python main.py

# Run LIVE (executes real MT5 orders)
python main.py --live

# Dashboard only
python main.py --dash-only

# Single scan and exit
python main.py --scan-now
```

### Auto-start with Windows Task Scheduler

1. Open Task Scheduler → Create Basic Task
2. Name: `ARIA Trading Bot`
3. Trigger: At log on / At system startup
4. Action: Start a program
   - Program: `D:\A office\AyoubOS\03 Projects\ARIA\.venv\Scripts\python.exe`
   - Arguments: `main.py --live`
   - Start in: `D:\A office\AyoubOS\03 Projects\ARIA`
5. Enable "Run whether user is logged on or not"

---

## 2. VPS (Windows Server — for 24/7 running)

MT5 requires Windows. Use Windows Server 2019/2022 VPS.

**Recommended providers:** Contabo, Vultr, AWS EC2 (Windows)

```powershell
# On Windows VPS
# 1. Install MetaTrader 5 + login to your broker account
# 2. Install Python 3.12 from python.org
# 3. Clone/copy ARIA folder

# Install dependencies
pip install -r requirements.txt

# Run as background service via NSSM
# Download NSSM from nssm.cc
nssm install ARIA "C:\Python312\python.exe" "D:\ARIA\main.py --live"
nssm set ARIA AppDirectory "D:\ARIA"
nssm start ARIA
```

---

## 3. Docker (DRY RUN only — MT5 requires Windows)

```bash
# Build image
docker build -t aria-trading:latest .

# Run dashboard + analysis (no live MT5)
docker run -d \
  --name aria-trading \
  --env-file .env \
  -p 8050:8050 \
  -p 8051:8051 \
  -v $(pwd)/db:/app/db \
  -v $(pwd)/logs:/app/logs \
  aria-trading:latest

# Full stack with PostgreSQL + Redis
docker-compose up -d
```

Note: Live MT5 trading from Docker requires a Windows container or Wine setup.

---

## 4. Environment Variables (.env)

```env
# MT5 Connection
MT5_LOGIN=436699881
MT5_PASSWORD=your_password
MT5_SERVER=Exness-MT5Trial9

# AI (required)
ANTHROPIC_API_KEY=sk-ant-...

# Telegram Alerts (required)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Obsidian Vault (for knowledge output)
OBSIDIAN_VAULT_PATH=D:/A office/AyoubOS/02-Obsidian-Vault

# Risk Settings
RISK_PCT=0.01          # 1% per trade
MAX_DAILY_LOSS=0.03    # 3% daily halt
MAX_DAILY_GAIN=0.05    # 5% daily target halt
MAX_CONCURRENT=3       # Max open positions
MAX_TRADES_DAY=5       # Max trades per day

# Signal Settings
MIN_SIGNAL_SCORE=65    # Minimum score to emit signal (0-100)
AUTO_EXECUTE_SCORE=70  # Minimum score for auto-execution in live mode
```

---

## 5. Dashboard

Access: **http://127.0.0.1:8050**

Pages:
| Page | URL | Description |
|------|-----|-------------|
| Dashboard | / | Live signals, positions, risk meters |
| Signals | /signals | All pair signals + scores |
| Backtest | /backtest | Run backtests inline |
| WFO | /wfo | Walk-forward optimization |
| Monte Carlo | /montecarlo | MC validation |
| Portfolio | /portfolio | Equity curve, correlation |
| Risk | /risk | Risk engine status |
| Learning | /learning | ML status, hypothesis queue |
| Settings | /settings | Pair config, thresholds |
| Pair Profiles | /profiles | Per-pair statistics |

### FastAPI (optional)
```bash
python -c "from api.server import run_api; run_api()"
# http://localhost:8051/health
# http://localhost:8051/signals
# ws://localhost:8051/ws/signals
```

---

## 6. CLI Reference

```bash
python main.py --live             # Live trading (MT5 execution enabled)
python main.py                    # Dry run (signals only, no orders)
python main.py --dash-only        # Dashboard only, no scanner
python main.py --scan-now         # Single scan cycle and exit
python main.py --presession       # Run pre-session AI brief and exit

python backtest.py --pair EURUSDm --days 90
python backtest.py --pair EURUSDm --days 90 --analyze
python wfo.py --pair EURUSDm --total 365 --is 90 --oos 30
python montecarlo.py --pair EURUSDm --sims 1000
python stress.py --pair EURUSDm
python sensitivity.py --pair EURUSDm
python -m backtest.multi_pair --days 90 --score 70
python -m backtest.oos_check EURUSDm --days 180

pytest tests/ -v                  # Run all 132 tests
```

---

## 7. Monitoring

- **Dashboard**: http://127.0.0.1:8050 — live state
- **Logs**: `logs/aria_YYYY-MM-DD.log` (14-day rotation)
- **DB**: `db/aria.db` (SQLite, backed up daily to `db/backups/`)
- **Obsidian**: `02-Obsidian-Vault/Trades/` — per-trade notes
- **Telegram**: All alerts sent to configured chat

---

## 8. Emergency Procedures

### Emergency close all positions
- **Dashboard**: Risk page → "EMERGENCY CLOSE ALL" (two-click confirm)
- **Telegram**: Sent automatically on close
- **Manual**: In MT5 terminal → close all positions

### Halt trading
- Dashboard: ARIA halts automatically on daily loss/gain limits
- Manual halt: Set `HARD_HALT=true` in .env + restart

### Recovery after crash
ARIA auto-reconciles MT5 positions on startup. If something looks wrong:
```powershell
# Check MT5 state
python -c "import MetaTrader5 as mt5; mt5.initialize(); print(mt5.positions_get())"

# Run reconciliation only
python main.py --dash-only  # dashboard shows reconciled state
```

---

## 9. Security

- [ ] `.env` not committed to git (in `.gitignore`)
- [ ] Dashboard bound to `127.0.0.1` (not exposed publicly)
- [ ] Telegram alerts go to private chat only
- [ ] MT5 demo account used until strategy validated on live
- [ ] Emergency close button tested before going live
