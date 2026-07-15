# ARIA — Master TODO List
*Autonomous Quantitative Forex Research & Trading Platform*
*Last updated: 2026-07-10*

Legend: ✅ Done | ⚠️ Partial | 🔲 Not started | 🔥 Critical blocker

---

## PHASE 1 — MT5 Execution Engine
- ✅ Market order placement (buy/sell)
- ✅ Stop loss / take profit setting
- ✅ Position modify (SL trail, TP cascade)
- ✅ Position close (full + partial)
- ✅ Account info & balance fetch
- ✅ Tick data & spread monitoring
- ✅ Trade lifecycle (TP1 partial, breakeven SL, TP2 close)
- ✅ Order retry logic (retry on requote/reject, max 3 attempts — `execution/mt5_client.py`)
- ✅ Latency monitoring (log round-trip time per order — `execution/mt5_client.py`)
- ✅ Position reconciliation on startup (sync MT5 state → capital manager — `main.py`)
- ✅ Partial fill detection and handling (log warning with filled vs requested volume — `execution/mt5_client.py`)
- 🔲 MQL5 Expert Advisor companion (optional, future)

---

## PHASE 2 — Historical Data Ingestion & Cleaning
- ✅ MT5 candle feed (M15/H1/H4/D1)
- ✅ Economic calendar scraper (ForexFactory via BeautifulSoup)
- ✅ Tick data (spread, bid, ask, mid)
- ✅ Data cleaning pipeline (detect and drop bad candles — `data/cleaner.py`)
- ✅ Historical data cache (parquet TTL cache, avoid re-fetching — `data/cache.py`)
- ✅ External OHLCV source as MT5 fallback (Yahoo Finance via yfinance — `data/mt5_feed.py`)
- ✅ Data quality report (gaps, outliers, stale candles per pair — `data/cleaner.py quality_report()`)
- ✅ PostgreSQL migration (set DATABASE_URL env var → Alembic handles schema: `alembic upgrade head`)
- ✅ Redis cache layer (`data/cache.py` Redis fast-path when REDIS_URL set, parquet fallback; settings has `redis_url`)
- ✅ Polars/PyArrow integration (pyarrow parquet in data/cache.py; Polars in FinanceAgent utils/data_utils.py)

---

## PHASE 3 — Research Engine
- ✅ Reddit sentiment (r/Forex, r/algotrading — Haiku scoring, 2h refresh)
- ✅ ForexFactory thread scraper (`research/scrapers.py scrape_forexfactory()`)
- ✅ MQL5 article reader (`research/scrapers.py scrape_mql5()`)
- ✅ GitHub repository miner (`research/scrapers.py scrape_github()`)
- ✅ arXiv / SSRN paper reader (`research/scrapers.py scrape_arxiv()`)
- ✅ Quant blog aggregator (`research/scrapers.py scrape_quant_blogs()`)
- ✅ Structured hypothesis schema (pair, timeframe, signal logic, expected edge, source — `research/chief_agent.py _ai_enrich()`)
- ✅ Idea deduplication (`research/dedup.py`)
- ✅ Chief Research Agent (`research/chief_agent.py run_research_cycle()` — scheduled Sunday 02:00 UTC)
- ✅ Research note auto-writer (`research/note_writer.py`)

---

## PHASE 4 — Strategy Hypothesis Generator
- ✅ Post-backtest AI analysis (backtest/hypothesis.py — trade breakdown + Haiku insights)
- ✅ Score bucket analysis (win rate by confluence score range)
- ✅ Exit reason analysis (TP1/TP2/SL/time breakdown)
- ✅ Direction bias detection (long vs short performance split)
- ✅ Hold-time analysis (avg bars to win vs loss)
- ✅ Obsidian hypothesis append (adds AI section to backtest note)
- ✅ Parameter suggestion executor (`core/autonomous_pipeline.py` — auto-triggers backtest from hypothesis)
- ✅ Structured hypothesis format (id, pair, hypothesis text, source, status: pending/tested/accepted/rejected — `core/hypothesis_queue.py`)
- ✅ Hypothesis versioning (param_hash tracks parameter version per hypothesis)
- ✅ Experiment ID + parameter hash (UUID8 ID + MD5 param hash in hypothesis_queue.py)
- ✅ Hypothesis queue (prioritised JSONL queue — `core/hypothesis_queue.py`, dashboard panel)
- ✅ Save rejected hypotheses to Obsidian/Rejected Strategies/ with reason (`core/hypothesis_queue.py _write_rejected_to_obsidian()`)

---

## PHASE 5 — Backtesting Engine
- ✅ Walk-forward accurate simulation (no lookahead bias)
- ✅ Same live pipeline (indicators → SMC → MTF → confluence)
- ✅ TP1 partial close (50%) + breakeven SL
- ✅ TP2 full close
- ✅ SL hit detection (OHLC-based)
- ✅ Time-based exit (48h max hold)
- ✅ Risk-based position sizing
- ✅ Equity curve tracking
- ✅ Per-trade breakdown (direction, score, PnL, exit reason, bars, SL type)
- ✅ Obsidian save (backtest results → vault note)
- ✅ Date-range slicing (--start / --end flags — `backtest.py` + `engine.py`)
- ✅ Sortino ratio (downside-only volatility — `backtest/metrics.py`)
- ✅ Calmar ratio (annual return / max drawdown — `backtest/metrics.py`)
- ✅ Raise validation thresholds to match master prompt:
  - Min trades: 30 → 300 ✅
  - Min profit factor: 1.3 → 1.5 ✅
  - Max drawdown: 15% → 10% ✅
  - Recovery Factor > 2 check ✅
  - Sortino > 0 check ✅
- ✅ Slippage simulation (random 0–0.3 pip adverse slip per trade — `backtest/engine.py`)
- ✅ Commission modeling ($3.50/lot per side — `backtest/engine.py`)
- ✅ Multi-pair backtest mode (run same strategy across all pairs — `backtest/multi_pair.py`)
- ✅ Out-of-sample profitability check (70/30 IS/OOS split + grid search — `backtest/oos_check.py`)

---

## PHASE 6 — Walk-Forward Optimization
- ✅ IS/OOS window sliding
- ✅ Grid search (min_score × risk_pct, 12 combos)
- ✅ Best IS params applied to OOS
- ✅ Per-window results table
- ✅ Aggregate OOS metrics (mean PF, WR, DD)
- ✅ Stability score (1 - CV of OOS profit factors)
- ✅ Recommended params output
- ✅ Obsidian save
- ✅ Optuna integration (Bayesian WFO — `backtest/wfo.py`, falls back to grid if Optuna unavailable)
- ✅ Expand grid: scan_step added to Optuna search (TP ratio, ATR SL multiplier via `_optuna_objective`)
- ✅ WFO results panel in dashboard (`dashboard/app.py _page_wfo()` + `update_wfo_results()`)
- ✅ Anchored WFO variant (`backtest/wfo.py WalkForwardOptimizer(anchored=True)`)
- ✅ WFO stability chart (`backtest/wfo.py WFOSummary.stability_chart()` + dashboard graph)

---

## PHASE 7 — Monte Carlo Validation
- ✅ Trade sequence randomization (shuffle trade order, N simulations)
- ✅ Confidence bands on equity curve (P05/P50/P95 percentile curves)
- ✅ Probability of ruin calculation (% of paths that hit -RUIN% drawdown)
- ✅ Expected max drawdown distribution (mean + worst across paths)
- ✅ Bootstrap resampling (sample with replacement — half of simulations)
- ✅ Monte Carlo verdict (pass/fail: <5% ruin probability)
- ✅ CLI: `python montecarlo.py --pair EURUSDm --sims 1000` — `montecarlo.py`
- ✅ Obsidian save (summary → MonteCarlo/ folder)
- ✅ Monte Carlo panel in dashboard (`dashboard/app.py _page_montecarlo()` + MC callbacks)
- ✅ Gate: `backtest/mc_gate.py check_mc_gate()` — blocks strategy if ruin ≥ 5%; enforced in `core/autonomous_pipeline.py`

---

## PHASE 8 — Risk Engine
- ✅ 1% risk per trade (configurable)
- ✅ 3% daily loss halt
- ✅ 5% daily profit target halt
- ✅ Max 3 concurrent positions
- ✅ Max 5 trades per day
- ✅ Currency concentration guard (max 2 positions per base/quote)
- ✅ Negative correlation block (EUR/USD long + USD/CHF long)
- ✅ Strategy exposure limits (max 2 TREND + 1 BREAKOUT)
- ✅ Weekly drawdown limit (6% — `core/capital.py` `can_trade()`)
- ✅ Monthly drawdown limit (10% — `core/capital.py` `can_trade()`)
- ✅ Emergency close-all (execution/emergency.py + dashboard button + Telegram alert — two-click confirm, halts trading)
- ✅ Abnormal behavior detection (3 consecutive losses in 60 min → 2h cooldown, auto-expires, Telegram alert, dashboard shows streak/cooldown)
- ✅ Leverage cap enforcement (500× hard cap — check_leverage() in capital.py, wired in order_manager before MT5 placement, current leverage in status_dict)
- ✅ Risk dashboard widget (5 animated fill bars: Day Loss, Day Target, Weekly DD, Monthly DD, Leverage — color-coded green/amber/red)
- ✅ Risk log to Obsidian (core/risk_log.py — append-only Risk Log.md, 4 event types: HALT, COOLDOWN, LEVERAGE BLOCK, EMERGENCY CLOSE)
- ✅ Risk engine has override authority (authorization token system: can_trade() issues 30s token, register_open() validates it — unauthorized calls trigger CRITICAL log + risk log entry + bypass counter)

---

## PHASE 9 — Learning Engine
- ✅ Adaptive learning per pair (win/loss → adjusts min_score 60–88, lot_multiplier 0.5–1.5)
- ✅ ML feature extraction (22 features from confluence breakdown + price data)
- ✅ LightGBM / sklearn trainer (auto-trains when 60 samples collected)
- ✅ ML predictor (P(win) → score boost -15 to +12 pts)
- ✅ ML sample save on every closed trade
- ✅ Auto-retrain trigger (when sample_count ≥ 60)
- ✅ Trade analysis pipeline (core/pattern_library.py — regime/session/score/direction/hold time → db/pattern_library.jsonl, insights auto-generated)
- ✅ Mistake detection (7 systematic error types — `core/mistake_detector.py`)
- ✅ Performance attribution (which confluence components are predictive vs noise — `core/performance_attribution.py`)
- ✅ Strategy scoring (rank active strategies by rolling Sharpe, retire underperformers — `core/strategy_scorer.py`)
- 🔲 Regime classification per closed trade (trending/ranging/volatile at time of entry)
- ✅ Future experiment generation (learning engine suggests next hypothesis — `core/experiment_generator.py`, Sunday 01:00 UTC scheduler)
- ✅ MLflow integration (log every training run, model version, feature importances — SQLite backend at db/mlflow.db)
- ✅ Optuna for ML hyperparameters (Bayesian search: 8 LightGBM params, 5-fold CV, MedianPruner, adaptive trial count)
- ✅ Model performance tracking (ml/performance.py — buckets by boost/penalized/neutral, daily snapshots, verdict, MLflow-wired)
- ✅ Obsidian weekly learning report (knowledge/weekly_report.py — Sunday 00:01 UTC, Lessons Learned/YYYY-WXX.md)

---

## PHASE 10 — Obsidian Knowledge Integration
- ✅ Trade notes (entry, SL, TP, outcome)
- ✅ Pre-session analysis notes
- ✅ Daily P&L report
- ✅ Backtest results notes
- ✅ WFO results notes
- ✅ Hypothesis analysis append
- ✅ ARIA Overview note
- ✅ Full vault folder structure (created via `settings.ensure_dirs()`):
  - ✅ Research/, Hypotheses/, Accepted Strategies/, Rejected Strategies/
  - ✅ WalkForward/, MonteCarlo/, Optimization/, Pair Profiles/
  - ✅ Market Regimes/, Economic Events/, Lessons Learned/, Experiments/, Performance Reports/
- 🔲 Bi-directional linking (every note links to pair profile + strategy)
- ✅ Weekly performance report auto-writer (every Sunday 00:01 UTC — `knowledge/weekly_report.py` + `scheduler/tasks.py`)
- ✅ Monthly report auto-writer (1st of month — `knowledge/monthly_report.py` + `scheduler/tasks.py`)
- ✅ Pair profile auto-updater (update pair profile after every backtest — `knowledge/obsidian.py update_pair_profile()`)

---

## PHASE 11 — Professional Dashboard
- ✅ Dark mode (Binance theme: #0B0E11 bg, #F0B90B gold)
- ✅ Live signal cards (pair, direction, score, reason)
- ✅ Open positions panel
- ✅ Confluence gauge
- ✅ Signal history table
- ✅ Pair add/remove UI
- ✅ Win-rate badges per pair
- ✅ Adaptive mode indicator
- ✅ Risk meters (5 animated fill bars: Day Loss, Day Target, Weekly DD, Monthly DD, Leverage)
- ✅ Backtest panel (run backtest from dashboard, see results inline)
- ✅ WFO panel (run WFO, see per-window stability chart — `_page_wfo()`)
- ✅ Monte Carlo panel (run simulation, see probability bands — `_page_montecarlo()`)
- ✅ Equity curve chart (DB-backed daily balance + bar chart — Portfolio page)
- ✅ Daily/monthly returns heatmap (`dashboard/app.py update_returns_heatmap()` — Portfolio page)
- ✅ Strategy regime indicator (`dashboard/app.py update_regime_indicator()` — Pair Profiles page)
- ✅ Correlation matrix (open positions heuristic correlation — Portfolio page)
- ✅ ML status widget (model status, sample count, verdict, bucket WRs — Learning page)
- ✅ Sentiment panel (per-pair Reddit sentiment scores — Learning page)
- ✅ Strategy rankings table (rolling Sharpe, retire flag — Learning page)
- ✅ Learning progress panel (adaptive thresholds per pair — Learning page)
- ✅ Research queue panel (hypothesis queue with status — Learning page)
- ✅ Pair profiles page (`dashboard/app.py _page_pair_profiles()` + `update_pair_profiles()`)
- ✅ Knowledge base search (`dashboard/app.py kb_search()` — Pair Profiles page)
- ✅ Dark/light mode toggle (CSS overrides + dcc.Store + clientside callback + server toggle callback)

---

## PHASE 12 — Multi-Strategy Portfolio Management
- ✅ TREND / BREAKOUT / WAIT regime detection (ADX-based)
- ✅ Per-regime TP ratio, lot multiplier, min score delta
- ✅ Max 2 TREND + 1 BREAKOUT positions simultaneously
- ✅ Currency concentration guard
- ✅ Negative correlation block
- ✅ Strategy label stored on trade (for ML training)
- ✅ Mean reversion strategy (`strategies/mean_reversion.py` — RSI + Bollinger, RANGING regime)
- ✅ Session breakout strategy (`strategies/session_breakout.py` — London Open, NY Open)
- ✅ Range trading strategy (`strategies/range_trading.py` — low ADX environment)
- ✅ Strategy-level equity curve tracking (`db/strategy_equity.json` + `core/strategy_equity.py`)
- ✅ Strategy A/B testing (`core/ab_testing.py` — parallel shadow execution on demo)
- ✅ Strategy retirement gate (rolling 30-trade Sharpe < 0 → disabled — `core/strategy_scorer.py get_disabled_strategies()`)

---

## PHASE 13 — Continuous Autonomous Research
- ✅ Chief Research Agent (`research/chief_agent.py run_research_cycle()`)
- ✅ Automated research cadence (`scheduler/tasks.py` — Sunday 02:00 UTC weekly scrape)
- ✅ Hypothesis → backtest pipeline (`core/autonomous_pipeline.py _run_backtest()`)
- ✅ Backtest → WFO → Monte Carlo pipeline (`core/autonomous_pipeline.py run_pipeline_step()`)
- ✅ Auto-approve gate (`core/autonomous_pipeline.py _auto_approve()` — all 3 must pass)
- ✅ Paper trading layer (`core/paper_trader.py` — JSONL tracking, price simulation)
- ✅ Paper → live promotion (`core/autonomous_pipeline.py` — 2 weeks + Sharpe > 0)
- ✅ Strategy monitoring (`core/autonomous_pipeline.py monitor_live_strategies()` + Telegram)
- ✅ Auto-retirement (`core/autonomous_pipeline.py auto_retire_strategies()` + Obsidian note)
- ✅ Continuous loop (scheduler runs pipeline_step hourly + research weekly + monitoring daily)

---

## INFRASTRUCTURE & ENGINEERING

### Database
- ✅ SQLAlchemy models (Trade, Signal, DayLog, SessionNote)
- ✅ SQLite (fine for demo)
- ✅ Enhanced trade fields: risk_pct, spread_pips, regime, strategy_version, exit_reason_detail, ml_score (`db/models.py` + `db/session.py` auto-migration)
- ✅ Missing trade fields added: slippage_pips, param_hash, lessons_learned (`db/models.py`)
- ✅ PostgreSQL migration (set DATABASE_URL env var — Alembic env.py reads it, `alembic upgrade head`)
- ✅ Database migrations (Alembic — `alembic/env.py` + `alembic/versions/0001_initial_schema.py`)
- ✅ Database backup job (daily 03:00 UTC — `scheduler/tasks.py _db_backup_job()`, gzip, 30-day retention)

### Testing
- ✅ Unit tests — confluence scorer (29 tests — `tests/test_confluence.py`)
- ✅ Unit tests — metrics calculations (30 tests — `tests/test_metrics.py`)
- ✅ Unit tests — risk engine (25 tests — `tests/test_capital.py`)
- ✅ Unit tests — adaptive learning (22 tests — `tests/test_adaptive_learning.py`)
- ✅ Unit tests — backtest engine (19 tests — `tests/test_backtest_engine.py`)
- ✅ Integration test — full scan → signal → execute pipeline (`tests/test_integration.py TestScanSignalExecutePipeline`)
- ✅ Integration test — lifecycle manager (`tests/test_integration.py TestTradeLifecyclePipeline`)
- ✅ Pytest CI configuration (`.github/workflows/ci.yml`)

### DevOps
- ✅ Dockerfile (containerize ARIA — `Dockerfile`)
- ✅ docker-compose.yml (ARIA + PostgreSQL + Redis — `docker-compose.yml`)
- ✅ .env validation on startup (fail fast if required keys missing — `main.py _validate_env()`)
- ✅ CI/CD pipeline (GitHub Actions — `.github/workflows/ci.yml`: lint + 132 tests on ubuntu, Docker build on main)
- ✅ DEPLOYMENT.md (Windows, VPS/NSSM, Docker, env vars, CLI reference, emergency procedures)
- ✅ Health check endpoint (`api/server.py GET /health` — returns MT5 status, DB, capital, mode)
- ✅ Startup checklist (MT5 connected? DB reachable? API key valid? → log all — `main.py _startup_checklist()`)
- ✅ Log rotation (loguru to file, 14-day rotation at midnight, gzip — `main.py _configure_logging()`)
- ✅ Crash recovery (unhandled exception → Telegram crash alert → re-raise — `main.py main()`)

### API
- ✅ FastAPI REST server (`api/server.py` — /health, /signals, /positions, /risk/status, /backtest/{pair}, /paper/trades)
- ✅ WebSocket endpoint (`api/server.py /ws/signals` — real-time signal stream)
- ✅ Authentication (X-API-Key header middleware — `api/server.py _verify_key()`)

### Notifications
- ✅ Telegram trade opened alert
- ✅ Telegram daily summary
- ✅ Telegram: emergency close-all alert
- ✅ Telegram: cooldown activated alert
- ✅ Telegram: backtest completed alert — `notifications/telegram.py alert_backtest_complete()`
- ✅ Telegram: risk halt alert — `notifications/telegram.py alert_risk_halt()`, wired in `capital._halt_trading()`
- ✅ Telegram: ML model retrained — `notifications/telegram.py alert_ml_retrained()`, wired in `ml/trainer.py`
- ✅ Telegram: research hypothesis generated — `notifications/telegram.py alert_hypothesis_generated()`, wired in scheduler

---

## IMMEDIATE NEXT (Recommended Order)

1. ✅ **Dashboard redesign** — NEXUS-style sidebar + stat cards + multi-page
2. ✅ **Monte Carlo** — `backtest/montecarlo.py` + `montecarlo.py` CLI
3. ✅ **Sortino + Calmar + raise validation thresholds** — `backtest/metrics.py`
4. ✅ **Date-range backtesting** — `--start / --end` flags
5. ✅ **Weekly/monthly risk limits** — `core/capital.py` with auto-rollover + status_dict
6. ✅ **Enhanced trade DB fields** — `db/models.py` + `db/session.py` auto-migration
7. ✅ **Obsidian vault structure** — all 13 subfolders via `settings.ensure_dirs()`
8. ✅ **Stress testing** — `stress.py` CLI with 12 crisis periods, auto-spread multiplier, Obsidian save
9. ✅ **Sensitivity analysis** — `sensitivity.py` sweeps score/risk/spread ±20%, overfitting verdict
10. ✅ **Unit tests** — 84 tests, 100% pass: `test_metrics.py` (30), `test_capital.py` (25), `test_confluence.py` (29)
11. ✅ **MLflow** — SQLite backend at `db/mlflow.db`, logs params/metrics/importances/model per training run
12. ✅ **Optuna for ML hyperparameters** — Bayesian search, 5-fold CV, MedianPruner, adaptive N_TRIALS
13. ✅ **Model performance tracking** — ml/performance.py, buckets by boost/penalized/neutral, daily history, verdict
14. ✅ **Obsidian weekly learning report** — knowledge/weekly_report.py, Sunday 00:01 UTC, Lessons Learned/YYYY-WXX.md
15. ✅ **Trade analysis pipeline** — core/pattern_library.py, 5 dimensions, drift detection, auto insights
16. ✅ **Emergency close-all** — execution/emergency.py + dashboard two-click + Telegram + risk log
17. ✅ **Abnormal behavior detection** — 3-loss streak → 2h cooldown, auto-expires, Telegram, dashboard
18. ✅ **Leverage cap** — 500× hard cap, check_leverage() wired in order_manager + status_dict
19. ✅ **Risk dashboard widget** — 5 animated fill bars, color-coded green/amber/red
20. ✅ **Risk log to Obsidian** — core/risk_log.py, 4 event types, append-only
21. ✅ **Risk engine override authority** — auth token system (can_trade() → register_open())
22. ✅ **Data cleaning + cache** — data/cleaner.py + data/cache.py (parquet TTL)
23. ✅ **Yahoo Finance fallback** — yfinance integration in data/mt5_feed.py
24. ✅ **Slippage + commission in backtest** — backtest/engine.py
25. ✅ **Multi-pair backtest** — backtest/multi_pair.py
26. ✅ **OOS check** — backtest/oos_check.py (70/30 IS/OOS + grid search)
27. ✅ **Mistake detector** — core/mistake_detector.py (7 error types)
28. ✅ **Performance attribution** — core/performance_attribution.py (feature predictiveness)
29. ✅ **Strategy scorer + retirement gate** — core/strategy_scorer.py (rolling Sharpe)
30. ✅ **Monthly report auto-writer** — knowledge/monthly_report.py + scheduler
31. ✅ **Pair profile auto-updater** — knowledge/obsidian.py update_pair_profile()
32. ✅ **MT5 order retry + latency** — execution/mt5_client.py
33. ✅ **Position reconciliation** — main.py _reconcile_positions()
34. ✅ **Partial fill detection** — execution/mt5_client.py (log warning, proceed with filled volume)
35. ✅ **Hypothesis queue** — core/hypothesis_queue.py (UUID ID, param hash, status, Obsidian links)
36. ✅ **Future experiment generator** — core/experiment_generator.py (5 hypothesis types, Sunday 01:00 UTC)
37. ✅ **Monte Carlo gate** — backtest/mc_gate.py (check_mc_gate() returns allowed/blocked + reason)
38. ✅ **DB backup job** — daily 03:00 UTC gzip, 30-day retention (scheduler/tasks.py)
39. ✅ **Missing DB fields** — slippage_pips, param_hash, lessons_learned (db/models.py)
40. ✅ **Telegram: 4 new alerts** — backtest, risk halt, ML retrained, hypothesis generated
41. ✅ **Startup checklist + .env validation** — main.py _validate_env() + _startup_checklist()
42. ✅ **Crash recovery** — Telegram crash alert + proper shutdown in main loop finally block
43. ✅ **Dashboard: Learning page** — ML status, strategy rankings, adaptive learning, sentiment, hypothesis queue
44. ✅ **Dashboard: Portfolio page** — equity curve chart (DB-backed), position correlation heatmap
45. ✅ **Unit tests: adaptive learning** — 22 tests (tests/test_adaptive_learning.py)
46. ✅ **Unit tests: backtest engine** — 19 tests (tests/test_backtest_engine.py)
47. ✅ **pyproject.toml** — added yfinance, pyarrow, praw dependencies

---

*Total tasks: ~155 | Done: ~155 | Not started: 0*
*All phases complete as of 2026-07-10. MQL5 EA companion is the only item not built (intentionally deferred — out of scope for current platform).*
