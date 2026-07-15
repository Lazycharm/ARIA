"""
FastAPI REST server + WebSocket endpoint for ARIA.

Endpoints:
  GET  /health              — system status
  GET  /signals             — current signals
  GET  /positions           — open positions
  GET  /risk/status         — capital manager status
  GET  /backtest/{pair}     — trigger quick backtest
  GET  /paper/trades        — paper trading results
  WS   /ws/signals          — real-time signal stream (JSON)

Authentication: X-API-Key header (from settings.api_key).
Run standalone:  uvicorn api.server:app --port 8051
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security.api_key import APIKeyHeader
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = object  # type: ignore

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False) if _FASTAPI_AVAILABLE else None


def _get_settings():
    from config.settings import settings
    return settings


def _verify_key(api_key: Optional[str] = None) -> str:
    s = _get_settings()
    expected = getattr(s, "api_key", None) or ""
    if not expected:
        return "anonymous"
    if api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


if _FASTAPI_AVAILABLE:
    app = FastAPI(title="ARIA Trading API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Health ─────────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health(key: str = Depends(_verify_key)):
        try:
            from data.mt5_feed import feed
            mt5_ok = feed.is_connected() if hasattr(feed, "is_connected") else False
        except Exception:
            mt5_ok = False

        try:
            from db.session import get_session
            from db.models import Trade
            with get_session() as db:
                db.query(Trade).limit(1).all()
            db_ok = True
        except Exception:
            db_ok = False

        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mt5_connected": mt5_ok,
            "db_ok": db_ok,
        }

    # ── Signals ────────────────────────────────────────────────────────────────

    @app.get("/signals")
    async def get_signals_api(key: str = Depends(_verify_key)):
        from signals.scanner import get_signals
        sigs = get_signals()
        return {
            pair: {
                "direction": s.direction,
                "score":     round(s.score, 1),
                "reason":    s.reason,
            }
            for pair, s in sigs.items()
        }

    # ── Positions ──────────────────────────────────────────────────────────────

    @app.get("/positions")
    async def get_positions_api(key: str = Depends(_verify_key)):
        try:
            from data.mt5_feed import feed
            positions = feed.get_positions()
            return {"positions": positions or []}
        except Exception as e:
            return {"positions": [], "error": str(e)}

    # ── Risk status ────────────────────────────────────────────────────────────

    @app.get("/risk/status")
    async def get_risk_status(key: str = Depends(_verify_key)):
        try:
            from core.capital import CapitalManager
            # Return last known state (singleton or shared)
            from signals.scanner import _capital as cap  # type: ignore
            if cap:
                return cap.status_dict
        except Exception:
            pass
        return {"halted": False, "realized_pnl": 0.0}

    # ── Quick backtest ─────────────────────────────────────────────────────────

    @app.get("/backtest/{pair}")
    async def quick_backtest(pair: str, days: int = 30, key: str = Depends(_verify_key)):
        try:
            from backtest.engine import BacktestEngine
            engine = BacktestEngine(pair=pair, days=days)
            res = engine.run()
            return {
                "pair":          res.pair,
                "trades":        res.total_trades,
                "win_rate":      round(res.win_rate, 1),
                "profit_factor": round(res.profit_factor, 2),
                "max_drawdown":  round(res.max_drawdown, 1),
                "net_pnl":       round(res.net_pnl, 2),
                "verdict":       res.verdict(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Paper trades ───────────────────────────────────────────────────────────

    @app.get("/paper/trades")
    async def paper_trades(key: str = Depends(_verify_key)):
        try:
            from core.paper_trader import get_all_trades, paper_performance
            trades = get_all_trades()
            perf   = paper_performance()
            return {"trades": trades, "performance": perf}
        except Exception as e:
            return {"trades": [], "performance": {}, "error": str(e)}

    # ── WebSocket — real-time signal stream ────────────────────────────────────

    _ws_clients: list[WebSocket] = []

    @app.websocket("/ws/signals")
    async def websocket_signals(websocket: WebSocket):
        await websocket.accept()
        _ws_clients.append(websocket)
        try:
            while True:
                await asyncio.sleep(5)
                from signals.scanner import get_signals
                sigs = get_signals()
                payload = json.dumps({
                    "type": "signals",
                    "ts":   datetime.now(timezone.utc).isoformat(),
                    "data": {
                        pair: {"direction": s.direction, "score": round(s.score, 1)}
                        for pair, s in sigs.items()
                    },
                })
                await websocket.send_text(payload)
        except WebSocketDisconnect:
            _ws_clients.remove(websocket)
        except Exception:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)

else:
    # Stub when FastAPI not installed
    class app:  # type: ignore
        pass


def run_api(host: str = "0.0.0.0", port: int = 8051, reload: bool = False) -> None:
    if not _FASTAPI_AVAILABLE:
        print("[API] FastAPI not installed. Run: pip install fastapi uvicorn")
        return
    import uvicorn
    uvicorn.run("api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_api()
