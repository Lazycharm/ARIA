"""
ARIA — NEXUS-Style Premium Trading Terminal
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html
from loguru import logger

from analysis.indicators import apply_all
from config.settings import settings
from config.pairs_config import get_pairs, add_pair, remove_pair
from core.adaptive_learning import adaptive
from core.capital import CapitalManager
from core.session import SessionManager
from data.mt5_feed import feed
from data.sentiment import sentiment_cache
from ml.predictor import predictor as ml_predictor
from signals.scanner import get_last_scan_time, get_signals, get_signal_history

# ── State ──────────────────────────────────────────────────────────────────────
_bt_result: dict = {}
_bt_lock   = threading.Lock()
_capital: CapitalManager | None = None
_order_mgr = None
_live_mode: bool = False
_session_mgr = SessionManager()


def set_capital(capital: CapitalManager) -> None:
    global _capital
    _capital = capital


def set_order_manager(om, live: bool = False) -> None:
    global _order_mgr, _live_mode
    _order_mgr = om
    _live_mode = live


# ── Palette ────────────────────────────────────────────────────────────────────
BG, SIDE, PANEL, PANEL2 = "#0A0B0F", "#0D0E17", "#111827", "#141B2D"
ACCENT, ACCENT2         = "#7C3AED", "#2563EB"
GREEN, RED, YELLOW      = "#10B981", "#EF4444", "#F59E0B"
TEXT, MUTED, MUTED2     = "#F1F5F9", "#94A3B8", "#64748B"
PURPLE                  = "#A78BFA"
BORDER                  = "rgba(255,255,255,0.06)"   # backward-compat
BLUE                    = ACCENT2                     # backward-compat

# ── Nav ────────────────────────────────────────────────────────────────────────
NAV = [
    ("dashboard", "◈", "Dashboard",  "MAIN"),
    ("signals",   "⚡", "Signals",    None),
    ("backtest",  "▶", "Backtest",   None),
    ("wfo",       "⟳", "WFO",        None),
    ("montecarlo","◉", "Monte Carlo", None),
    ("portfolio", "◎", "Portfolio",  "TRADING"),
    ("risk",      "⊗", "Risk",       None),
    ("learning",  "✦", "Learning",   None),
    ("profiles",  "⊞", "Pair Profiles", None),
    ("settings",  "⚙", "Settings",   "SYSTEM"),
]
PAGE_IDS = [n[0] for n in NAV]

# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;padding:0;background:#0A0B0F;color:#F1F5F9;font-family:'Inter',sans-serif;overflow:hidden;height:100vh}

/* Sidebar */
.aria-sidebar{width:200px;min-width:200px;background:#0D0E17;border-right:1px solid rgba(124,58,237,.15);display:flex;flex-direction:column;height:100vh;overflow:hidden;flex-shrink:0}
.aria-logo{padding:18px 16px 14px;border-bottom:1px solid rgba(255,255,255,.04);flex-shrink:0}
.aria-logo-text{font-size:17px;font-weight:700;color:#A78BFA;letter-spacing:3px;display:block}
.aria-logo-sub{font-size:8px;color:#64748B;letter-spacing:2px;text-transform:uppercase;margin-top:3px;display:block}
.nav-section{font-size:8px;font-weight:600;color:#64748B;letter-spacing:1.5px;text-transform:uppercase;padding:12px 14px 5px;flex-shrink:0}
.nav-item{display:flex;align-items:center;gap:9px;padding:8px 12px;cursor:pointer;border-radius:6px;font-size:12px;color:#94A3B8;font-weight:500;border:none;background:transparent;width:calc(100% - 10px);margin:1px 5px;text-align:left;transition:all .15s;font-family:'Inter',sans-serif}
.nav-item:hover{background:rgba(124,58,237,.08);color:#F1F5F9}
.nav-item.active{background:linear-gradient(90deg,rgba(124,58,237,.2) 0%,rgba(124,58,237,.04) 100%);color:#A78BFA;border-left:2px solid #7C3AED;padding-left:10px}
.nav-icon{font-size:13px;width:16px;text-align:center;flex-shrink:0}
.nav-wl-header{font-size:8px;font-weight:600;color:#64748B;letter-spacing:1.5px;text-transform:uppercase;padding:10px 12px 5px;flex-shrink:0}
.pw-row-new{padding:6px 12px;cursor:pointer;transition:background .1s;font-size:11px}
.pw-row-new:hover{background:rgba(255,255,255,.02)}
.pw-row-new.active{background:rgba(124,58,237,.08);border-left:2px solid #7C3AED;padding-left:10px}
.pw-name{font-weight:600;color:#F1F5F9;font-size:11px;display:block}
.pw-price{font-family:'JetBrains Mono',monospace;color:#F1F5F9;font-size:10px}
.pw-spread{color:#64748B;font-size:9px}
.pair-add-box-new{padding:8px 10px;border-top:1px solid rgba(255,255,255,.04);flex-shrink:0}
.pair-add-input-new{width:100%;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);color:#F1F5F9;font-size:10px;padding:5px 8px;border-radius:5px;outline:none;margin-bottom:5px;font-family:'Inter',sans-serif}
.pair-add-input-new:focus{border-color:#7C3AED}
.pair-add-btn-new{width:100%;background:linear-gradient(90deg,#7C3AED,#6D28D9);color:#fff;border:none;border-radius:5px;padding:5px;font-size:10px;font-weight:600;cursor:pointer;font-family:'Inter',sans-serif;transition:opacity .1s}
.pair-add-btn-new:hover{opacity:.85}

/* Theme toggle */
.theme-toggle-btn{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:4px 10px;font-size:11px;cursor:pointer;color:#94A3B8;font-family:'Inter',sans-serif;transition:all .2s;flex-shrink:0;margin-left:10px}
.theme-toggle-btn:hover{background:rgba(124,58,237,.15);color:#A78BFA;border-color:#7C3AED}

/* Light mode overrides */
body.light-mode{background:#F8FAFC;color:#0F172A}
.light-mode .aria-sidebar{background:#FFFFFF;border-right:1px solid #E2E8F0}
.light-mode .aria-logo-text{color:#7C3AED}
.light-mode .aria-logo-sub{color:#94A3B8}
.light-mode .nav-item{color:#475569}
.light-mode .nav-item:hover{background:rgba(124,58,237,.07);color:#1E293B}
.light-mode .nav-item.active{color:#7C3AED;background:rgba(124,58,237,.08)}
.light-mode .nav-wl-header,.light-mode .nav-section{color:#94A3B8}
.light-mode .aria-header{background:#FFFFFF;border-bottom:1px solid #E2E8F0}
.light-mode .ticker-pair{color:#94A3B8}
.light-mode .ticker-price{color:#0F172A}
.light-mode .acct-item .al{color:#94A3B8}
.light-mode .acct-item .av{color:#0F172A}
.light-mode .aria-card,.light-mode .card-panel,.light-mode .stat-card,.light-mode .rp-card{background:#FFFFFF;border-color:#E2E8F0}
.light-mode .card-header{color:#475569;border-bottom:1px solid #E2E8F0}
.light-mode .stat-val,.light-mode .stat-value{color:#0F172A}
.light-mode .stat-label{color:#64748B}
.light-mode .pw-name{color:#0F172A}
.light-mode .pair-add-input-new{background:#F1F5F9;border-color:#E2E8F0;color:#0F172A}
.light-mode .theme-toggle-btn{color:#475569;border-color:#E2E8F0;background:#F1F5F9}

/* Header */
.aria-header{height:46px;background:#0D0E17;border-bottom:1px solid rgba(255,255,255,.04);display:flex;align-items:center;padding:0 12px;gap:0;overflow:hidden;flex-shrink:0}
.ticker-strip{display:flex;overflow:hidden;flex:1;height:100%}
.ticker-item{padding:0 11px;height:100%;display:flex;flex-direction:column;justify-content:center;border-right:1px solid rgba(255,255,255,.04);cursor:pointer;min-width:80px;transition:background .1s}
.ticker-item:hover{background:rgba(255,255,255,.02)}
.ticker-pair{font-size:8px;font-weight:600;color:#64748B;letter-spacing:.5px}
.ticker-price{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;color:#F1F5F9;margin-top:1px}
.ticker-spread{font-size:8px}
.tc-up{color:#10B981}.tc-dn{color:#EF4444}
.header-acct{display:flex;align-items:center;gap:18px;padding-left:14px;border-left:1px solid rgba(255,255,255,.04);flex-shrink:0}
.acct-item .al{font-size:8px;color:#64748B;letter-spacing:.8px;text-transform:uppercase;display:block}
.acct-item .av{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;color:#F1F5F9;display:block}

/* Stat cards */
.stat-cards-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:12px 12px 0;flex-shrink:0}
.stat-card{background:#111827;border:1px solid rgba(255,255,255,.05);border-radius:10px;padding:12px 14px;position:relative;overflow:hidden}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#7C3AED,#2563EB)}
.stat-card-label{font-size:8px;color:#94A3B8;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:7px;display:block}
.stat-card-value{font-family:'JetBrains Mono',monospace;font-size:19px;font-weight:700;color:#F1F5F9;display:block}

/* Cards */
.aria-card{background:#111827;border:1px solid rgba(255,255,255,.05);border-radius:10px;overflow:hidden}
.card-header{padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.04);display:flex;align-items:center;justify-content:space-between;font-size:9px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:1px;flex-shrink:0}

/* Right panel */
.aria-right{width:285px;min-width:285px;background:#0D0E17;border-left:1px solid rgba(255,255,255,.04);display:flex;flex-direction:column;overflow-y:auto;overflow-x:hidden;flex-shrink:0}
.insight-card{margin:10px;background:#111827;border:1px solid rgba(124,58,237,.2);border-radius:10px;padding:12px;position:relative;overflow:hidden;flex-shrink:0}
.insight-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#7C3AED,#2563EB)}
.insight-label{font-size:8px;font-weight:600;color:#7C3AED;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;display:block}
.insight-text{font-size:11px;color:#94A3B8;line-height:1.5}
.rp-card{margin:0 10px 8px}

/* Signal cards in right panel */
.sig-card-new{padding:9px 12px;border-bottom:1px solid rgba(255,255,255,.03)}
.sig-pair-name{font-size:12px;font-weight:600;color:#F1F5F9}
.sig-score-badge{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700}
.sig-bar-wrap{background:rgba(255,255,255,.06);height:2px;border-radius:1px;overflow:hidden;margin:4px 0}
.sig-bar{height:100%}
.sig-reason-text{font-size:9px;color:#64748B;line-height:1.4;margin-bottom:6px}
.btn-buy-new{background:linear-gradient(90deg,#059669,#10B981);color:#fff;border:none;border-radius:5px;padding:5px 0;font-size:10px;font-weight:700;cursor:pointer;width:100%;transition:opacity .1s;font-family:'Inter',sans-serif}
.btn-buy-new:hover{opacity:.85}
.btn-sell-new{background:linear-gradient(90deg,#DC2626,#EF4444);color:#fff;border:none;border-radius:5px;padding:5px 0;font-size:10px;font-weight:700;cursor:pointer;width:100%;transition:opacity .1s;font-family:'Inter',sans-serif}
.btn-sell-new:hover{opacity:.85}

/* Position rows */
.pos-row-new{padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.03);font-size:11px}
.pos-pnl.up{color:#10B981;font-weight:700;font-family:'JetBrains Mono',monospace}
.pos-pnl.dn{color:#EF4444;font-weight:700;font-family:'JetBrains Mono',monospace}
.pos-meta{color:#64748B;font-size:9px;margin:3px 0;font-family:'JetBrains Mono',monospace}
.btn-close-new{background:transparent;color:#EF4444;border:1px solid rgba(239,68,68,.3);border-radius:4px;padding:2px 8px;font-size:9px;cursor:pointer;font-weight:600;transition:all .1s;font-family:'Inter',sans-serif}
.btn-close-new:hover{background:rgba(239,68,68,.1)}
.btn-emergency{background:rgba(239,68,68,.06);color:#EF4444;border:1px solid rgba(239,68,68,.25);border-radius:5px;padding:5px 0;font-size:9px;font-weight:700;cursor:pointer;width:100%;transition:all .2s;font-family:'Inter',sans-serif;text-transform:uppercase;letter-spacing:.8px}
.btn-emergency:hover{background:rgba(239,68,68,.14);border-color:rgba(239,68,68,.5)}
.btn-emergency-armed{background:rgba(239,68,68,.22);color:#fff;border:1px solid #EF4444;border-radius:5px;padding:5px 0;font-size:9px;font-weight:700;cursor:pointer;width:100%;font-family:'Inter',sans-serif;text-transform:uppercase;letter-spacing:.8px;animation:blink .5s step-start infinite}

/* Stats grid */
.stat-grid-new{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:rgba(255,255,255,.04)}
.stat-cell-new{padding:8px 12px;background:#0D0E17}
.stat-l{font-size:8px;color:#64748B;display:block;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px}
.stat-v{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;display:block}

/* Chart controls */
.chart-controls{display:flex;align-items:center;gap:7px;padding:7px 12px;background:#0D0E17;border-bottom:1px solid rgba(255,255,255,.04);flex-shrink:0}
.tf-btn{background:transparent;border:1px solid rgba(255,255,255,.07);color:#64748B;padding:3px 9px;font-size:10px;font-weight:500;cursor:pointer;border-radius:4px;font-family:'Inter',sans-serif;transition:all .15s}
.tf-btn:hover{border-color:rgba(124,58,237,.5);color:#A78BFA}
.tf-btn.active{background:rgba(124,58,237,.14);border-color:#7C3AED;color:#A78BFA}

/* Market overview */
.market-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:7px;padding:10px 12px;flex-shrink:0}
.market-card{background:#111827;border:1px solid rgba(255,255,255,.05);border-radius:8px;padding:8px 10px;cursor:pointer;transition:all .15s}
.market-card:hover{border-color:rgba(124,58,237,.3);background:rgba(124,58,237,.04)}
.market-pair{font-size:8px;font-weight:600;color:#94A3B8;margin-bottom:3px;display:block;letter-spacing:.5px}
.market-price{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;color:#F1F5F9;display:block}
.market-spread{font-size:8px;color:#64748B;margin-top:2px;display:block}

/* Tabs */
.tab-bar{display:flex;border-bottom:1px solid rgba(255,255,255,.04);padding:0 12px;background:#0D0E17;flex-shrink:0}
.tab-btn{background:none;border:none;color:#64748B;font-size:10px;font-weight:500;padding:6px 14px;cursor:pointer;border-bottom:2px solid transparent;font-family:'Inter',sans-serif;transition:all .15s;margin-bottom:-1px}
.tab-btn:hover{color:#94A3B8}
.tab-btn.active{color:#A78BFA;border-bottom-color:#7C3AED}

/* Feed tables */
.sfeed-hdr-new{display:grid;grid-template-columns:46px 60px 52px 42px 1fr;gap:5px;padding:5px 12px;color:#64748B;font-size:8px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(255,255,255,.04);position:sticky;top:0;background:#111827}
.sfeed-row-new{display:grid;grid-template-columns:46px 60px 52px 42px 1fr;gap:5px;padding:4px 12px;border-bottom:1px solid rgba(255,255,255,.025);font-size:10px;align-items:center}
.sfeed-row-new:hover{background:rgba(255,255,255,.015)}
.sfeed-row-new.hit{background:rgba(124,58,237,.04);border-left:2px solid #7C3AED;padding-left:10px}
.tlog-hdr-new{display:grid;grid-template-columns:44px 60px 36px 36px 82px 58px;gap:5px;padding:5px 12px;color:#64748B;font-size:8px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(255,255,255,.04);position:sticky;top:0;background:#111827}
.tlog-row-new{display:grid;grid-template-columns:44px 60px 36px 36px 82px 58px;gap:5px;padding:4px 12px;border-bottom:1px solid rgba(255,255,255,.025);font-size:10px;align-items:center}
.tlog-row-new:hover{background:rgba(255,255,255,.015)}

/* Risk meters */
.risk-meters{padding:8px 12px 4px}
.rm-row{margin-bottom:7px}
.rm-header{display:flex;justify-content:space-between;margin-bottom:3px}
.rm-label{font-size:8px;color:#64748B;text-transform:uppercase;letter-spacing:.8px;font-weight:600}
.rm-pct{font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700}
.rm-track{background:rgba(255,255,255,.06);border-radius:2px;height:4px;overflow:hidden}
.rm-fill{height:100%;border-radius:2px;transition:width .4s ease}

/* Backtest page */
.bt-form-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
.bt-input-new{width:100%;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:#F1F5F9;font-size:12px;padding:7px 10px;border-radius:6px;outline:none;font-family:'JetBrains Mono',monospace}
.bt-input-new:focus{border-color:#7C3AED}
.bt-run-btn-new{width:100%;background:linear-gradient(90deg,#7C3AED,#2563EB);color:#fff;border:none;border-radius:6px;padding:9px;font-size:12px;font-weight:700;cursor:pointer;font-family:'Inter',sans-serif;transition:opacity .1s;margin-bottom:10px}
.bt-run-btn-new:hover{opacity:.85}
.bt-result-box{font-size:10px;color:#94A3B8;white-space:pre;background:rgba(0,0,0,.25);padding:10px;border-radius:6px;font-family:'JetBrains Mono',monospace;min-height:80px;border:1px solid rgba(255,255,255,.05);overflow-y:auto;max-height:300px}

/* Badges */
.badge-live{background:rgba(16,185,129,.12);color:#10B981;border:1px solid rgba(16,185,129,.25);font-size:8px;padding:2px 7px;border-radius:20px;font-weight:600;letter-spacing:.5px}
.badge-dry{background:rgba(148,163,184,.1);color:#94A3B8;border:1px solid rgba(148,163,184,.2);font-size:8px;padding:2px 7px;border-radius:20px;font-weight:600;letter-spacing:.5px}
.badge-halt{background:rgba(239,68,68,.12);color:#EF4444;border:1px solid rgba(239,68,68,.25);font-size:8px;padding:2px 7px;border-radius:20px;font-weight:600;letter-spacing:.5px;animation:blink 1s step-start infinite}
@keyframes blink{50%{opacity:0}}
.learn-badge{font-size:8px;padding:1px 5px;border-radius:4px;font-weight:600;margin-left:3px}
.learn-up{background:rgba(16,185,129,.12);color:#10B981}
.learn-dn{background:rgba(239,68,68,.12);color:#EF4444}
.learn-neutral{background:rgba(148,163,184,.08);color:#94A3B8}
.no-data-new{padding:14px 12px;color:#64748B;font-size:11px;text-align:center}

/* Scrollbar */
::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.15)}

/* Select dropdown */
.Select-control,.Select--single>.Select-control{background:rgba(255,255,255,.05)!important;border:1px solid rgba(255,255,255,.1)!important;color:#F1F5F9!important;border-radius:6px!important;height:30px!important;min-height:30px!important}
.Select-placeholder,.Select--single .Select-value{line-height:28px!important;color:#94A3B8!important;font-size:11px!important;padding-left:8px!important}
.Select-input{height:28px!important}
.Select-arrow-zone .Select-arrow{border-color:#64748B transparent transparent!important}
.Select-menu-outer{background:#141B2D!important;border:1px solid rgba(124,58,237,.3)!important;z-index:9999!important;margin-top:1px!important}
.Select-option{background:#141B2D!important;color:#F1F5F9!important;font-size:11px!important;padding:5px 10px!important}
.Select-option:hover,.Select-option.is-focused,.VirtualizedSelectFocusedOption{background:rgba(124,58,237,.1)!important}

/* Placeholder pages */
.page-placeholder{display:flex;align-items:center;justify-content:center;flex:1;flex-direction:column;gap:10px;color:#64748B}
.ph-icon{font-size:40px;opacity:.25}
.ph-title{font-size:15px;font-weight:600;color:#94A3B8}
.ph-sub{font-size:11px;color:#64748B}
"""

# ── App init ───────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    title="ARIA — Autonomous FX",
    update_title=None,
    suppress_callback_exceptions=True,
)

app.index_string = (
    "<!DOCTYPE html><html><head>"
    "{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
    "<style>" + _CSS + "</style>"
    "</head><body style=\"background:#0A0B0F;margin:0;padding:0\">"
    "{%app_entry%}{%config%}{%scripts%}{%renderer%}"
    "</body></html>"
)

_first   = get_pairs()[0] if get_pairs() else "EURUSDm"
_TF_LIST = ["M5", "M15", "H1", "H4", "D1"]


# ── Layout builders ────────────────────────────────────────────────────────────

def _sidebar() -> html.Div:
    nav_items: list = []
    last_section = None
    for page_id, icon, label, section in NAV:
        if section and section != last_section:
            nav_items.append(html.Div(section, className="nav-section"))
            last_section = section
        nav_items.append(
            html.Button(
                [html.Span(icon, className="nav-icon"), label],
                id={"type": "nav-btn", "page": page_id},
                n_clicks=0,
                className="nav-item active" if page_id == "dashboard" else "nav-item",
            )
        )

    return html.Div(
        className="aria-sidebar",
        children=[
            html.Div(className="aria-logo", children=[
                html.Span("▲ ARIA", className="aria-logo-text"),
                html.Span("AUTONOMOUS FX", className="aria-logo-sub"),
            ]),
            html.Div(nav_items, style={"paddingTop": "6px", "flexShrink": "0"}),
            html.Div(
                style={"flex": "1", "overflowY": "auto", "overflowX": "hidden"},
                children=[
                    html.Div("WATCHLIST", className="nav-wl-header"),
                    html.Div(id="watchlist-panel"),
                ]
            ),
            html.Div(className="pair-add-box-new", children=[
                dcc.Input(
                    id="pair-add-input",
                    placeholder="Add symbol…",
                    debounce=False,
                    className="pair-add-input-new",
                    n_submit=0,
                ),
                html.Button("+ ADD", id="pair-add-btn", className="pair-add-btn-new", n_clicks=0),
                html.Div(id="pair-add-msg", style={"fontSize": "9px", "color": MUTED2, "marginTop": "3px"}),
            ]),
        ]
    )


def _right_panel() -> html.Div:
    return html.Div(
        className="aria-right",
        children=[
            html.Div(className="insight-card", children=[
                html.Span("ARIA INSIGHT", className="insight-label"),
                html.Div(id="aria-insight", className="insight-text",
                         children="Pre-session analysis available after 06:30 UTC."),
            ]),
            html.Div(className="rp-card aria-card", children=[
                html.Div(className="card-header", children=[
                    html.Span("ACTIVE SIGNALS"),
                    html.Span(id="signals-header", children=[]),
                ]),
                html.Div(id="signals-panel", style={"maxHeight": "185px", "overflowY": "auto"}),
            ]),
            html.Div(className="rp-card aria-card", children=[
                html.Div("OPEN POSITIONS", className="card-header"),
                html.Div(id="positions-panel", style={"maxHeight": "130px", "overflowY": "auto"}),
            ]),
            html.Div(className="rp-card aria-card", style={"marginBottom": "10px"}, children=[
                html.Div("RISK STATUS", className="card-header"),
                html.Div(id="stats-panel"),
                html.Div(id="risk-meters"),
                html.Div(style={"padding": "0 10px 10px"}, children=[
                    html.Button(
                        "EMERGENCY CLOSE ALL",
                        id="emg-close-btn",
                        n_clicks=0,
                        className="btn-emergency",
                    ),
                    html.Div(
                        id="emg-close-status",
                        style={"fontSize": "9px", "color": "#EF4444", "textAlign": "center",
                               "marginTop": "4px", "minHeight": "12px"},
                    ),
                ]),
            ]),
        ]
    )


def _page_dashboard() -> list:
    return [
        html.Div(className="stat-cards-row", children=[
            html.Div(className="stat-card", children=[
                html.Span("BALANCE", className="stat-card-label"),
                html.Span(id="dash-stat-balance", className="stat-card-value", children="—"),
            ]),
            html.Div(className="stat-card", children=[
                html.Span("EQUITY", className="stat-card-label"),
                html.Span(id="dash-stat-equity", className="stat-card-value", children="—"),
            ]),
            html.Div(className="stat-card", children=[
                html.Span("DAY P&L", className="stat-card-label"),
                html.Span(id="dash-stat-pnl", className="stat-card-value", children="—"),
            ]),
            html.Div(className="stat-card", children=[
                html.Span("WIN RATE", className="stat-card-label"),
                html.Span(id="dash-stat-wr", className="stat-card-value", children="—"),
            ]),
        ]),
        html.Div(
            style={"flex": "1", "padding": "10px 12px 0", "display": "flex",
                   "flexDirection": "column", "overflow": "hidden", "minHeight": 0},
            children=[
                html.Div(
                    className="aria-card",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                    children=[
                        html.Div(className="chart-controls", children=[
                            dcc.Dropdown(
                                id="pair-select",
                                options=[{"label": p, "value": p} for p in get_pairs()],
                                value=_first,
                                clearable=False,
                                style={"width": "128px", "flexShrink": "0"},
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "4px"},
                                children=[
                                    html.Button(
                                        tf,
                                        id={"type": "tf-btn", "tf": tf},
                                        n_clicks=0,
                                        className="tf-btn active" if tf == "M15" else "tf-btn",
                                    )
                                    for tf in _TF_LIST
                                ],
                            ),
                            html.Span(id="chart-price-label", style={"marginLeft": "auto"}),
                        ]),
                        dcc.Graph(
                            id="main-chart",
                            style={"flex": "1", "minHeight": "0"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="market-overview", className="market-grid"),
        html.Div(className="tab-bar", children=[
            html.Button("Signal Feed", id="tab-btn-feed", className="tab-btn active", n_clicks=0),
            html.Button("Trade Log",   id="tab-btn-log",  className="tab-btn",        n_clicks=0),
        ]),
        html.Div(
            style={"height": "130px", "overflow": "hidden", "flexShrink": "0",
                   "background": PANEL, "borderTop": "1px solid rgba(255,255,255,.04)"},
            children=[
                html.Div(id="signal-feed", style={"height": "100%", "overflowY": "auto"}),
                html.Div(id="trade-log",   style={"height": "100%", "overflowY": "auto", "display": "none"}),
            ],
        ),
    ]


def _page_signals() -> list:
    return [
        html.Div(
            style={"flex": "1", "padding": "12px", "overflow": "hidden",
                   "display": "flex", "flexDirection": "column"},
            children=[
                html.Div(
                    className="aria-card",
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                    children=[
                        html.Div("SIGNAL HISTORY — ALL PAIRS", className="card-header"),
                        html.Div(id="signal-feed-full", style={"flex": "1", "overflowY": "auto"}),
                    ],
                ),
            ],
        ),
    ]


def _page_backtest() -> list:
    return [
        html.Div(
            style={"flex": "1", "padding": "12px", "overflow": "auto"},
            children=[
                html.Div(className="aria-card", children=[
                    html.Div("BACKTEST ENGINE", className="card-header"),
                    html.Div(style={"padding": "14px"}, children=[
                        html.Div(style={"marginBottom": "10px"}, children=[
                            html.Label("Pair", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                       "marginBottom": "4px", "letterSpacing": "1px",
                                                       "textTransform": "uppercase"}),
                            dcc.Dropdown(
                                id="bt-pair-select",
                                options=[{"label": p, "value": p} for p in get_pairs()],
                                value=_first,
                                clearable=False,
                            ),
                        ]),
                        html.Div(className="bt-form-grid", children=[
                            html.Div(children=[
                                html.Label("Days", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                           "marginBottom": "4px", "letterSpacing": "1px",
                                                           "textTransform": "uppercase"}),
                                dcc.Input(id="bt-days-input",  type="number", value=90,
                                          min=7, max=365, className="bt-input-new",
                                          style={"width": "100%"}),
                            ]),
                            html.Div(children=[
                                html.Label("Min Score", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                                "marginBottom": "4px", "letterSpacing": "1px",
                                                                "textTransform": "uppercase"}),
                                dcc.Input(id="bt-score-input", type="number", value=70,
                                          min=50, max=95, className="bt-input-new",
                                          style={"width": "100%"}),
                            ]),
                        ]),
                        html.Button("▶  RUN BACKTEST", id="bt-run-btn", n_clicks=0,
                                    className="bt-run-btn-new"),
                        html.Div(id="bt-result", className="bt-result-box",
                                 children="Configure parameters and click Run Backtest."),
                    ]),
                ]),
            ],
        ),
    ]


def _page_placeholder(title: str, icon: str = "◈") -> list:
    return [
        html.Div(className="page-placeholder", children=[
            html.Div(icon, className="ph-icon"),
            html.Div(title, className="ph-title"),
            html.Div("Coming in the next sprint", className="ph-sub"),
        ]),
    ]


def _page_wfo() -> list:
    """WFO runner page: configure pair + windows, see per-window stability chart."""
    return [
        html.Div(
            style={"flex": "1", "padding": "12px", "overflow": "auto"},
            children=[
                html.Div(className="aria-card", children=[
                    html.Div("WALK-FORWARD OPTIMIZER", className="card-header"),
                    html.Div(style={"padding": "14px"}, children=[
                        html.Div(style={"marginBottom": "10px"}, children=[
                            html.Label("Pair", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                       "marginBottom": "4px", "letterSpacing": "1px",
                                                       "textTransform": "uppercase"}),
                            dcc.Dropdown(
                                id="wfo-pair-select",
                                options=[{"label": p, "value": p} for p in get_pairs()],
                                value=_first,
                                clearable=False,
                            ),
                        ]),
                        html.Div(className="bt-form-grid", children=[
                            html.Div(children=[
                                html.Label("Total Days", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                                 "marginBottom": "4px", "letterSpacing": "1px",
                                                                 "textTransform": "uppercase"}),
                                dcc.Input(id="wfo-total-days", type="number", value=365,
                                          min=90, max=730, className="bt-input-new", style={"width": "100%"}),
                            ]),
                            html.Div(children=[
                                html.Label("IS Days", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                              "marginBottom": "4px", "letterSpacing": "1px",
                                                              "textTransform": "uppercase"}),
                                dcc.Input(id="wfo-is-days", type="number", value=90,
                                          min=30, max=365, className="bt-input-new", style={"width": "100%"}),
                            ]),
                            html.Div(children=[
                                html.Label("OOS Days", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                               "marginBottom": "4px", "letterSpacing": "1px",
                                                               "textTransform": "uppercase"}),
                                dcc.Input(id="wfo-oos-days", type="number", value=30,
                                          min=10, max=90, className="bt-input-new", style={"width": "100%"}),
                            ]),
                            html.Div(children=[
                                html.Label("Trials", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                             "marginBottom": "4px", "letterSpacing": "1px",
                                                             "textTransform": "uppercase"}),
                                dcc.Input(id="wfo-trials", type="number", value=20,
                                          min=5, max=100, className="bt-input-new", style={"width": "100%"}),
                            ]),
                        ]),
                        html.Div(style={"display": "flex", "gap": "8px", "marginBottom": "10px"}, children=[
                            html.Button("▶  RUN WFO", id="wfo-run-btn", n_clicks=0,
                                        className="bt-run-btn-new", style={"flex": "1"}),
                            html.Button("⚓ ANCHORED", id="wfo-anchored-btn", n_clicks=0,
                                        className="tf-btn", style={"flexShrink": "0"}),
                        ]),
                        dcc.Store(id="wfo-anchored-store", data=False),
                        html.Div(id="wfo-result", className="bt-result-box",
                                 children="Configure parameters and click Run WFO."),
                    ]),
                ]),
                html.Div(className="aria-card", style={"marginTop": "10px", "minHeight": "280px"}, children=[
                    html.Div("WFO STABILITY CHART (OOS Profit Factor per Window)", className="card-header"),
                    dcc.Graph(id="wfo-stability-chart", style={"height": "260px"},
                              config={"displayModeBar": False}),
                ]),
            ],
        ),
    ]


def _page_montecarlo() -> list:
    """Monte Carlo panel: run simulation, see probability bands."""
    return [
        html.Div(
            style={"flex": "1", "padding": "12px", "overflow": "auto"},
            children=[
                html.Div(className="aria-card", children=[
                    html.Div("MONTE CARLO VALIDATION", className="card-header"),
                    html.Div(style={"padding": "14px"}, children=[
                        html.Div(style={"marginBottom": "10px"}, children=[
                            html.Label("Pair", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                       "marginBottom": "4px", "letterSpacing": "1px",
                                                       "textTransform": "uppercase"}),
                            dcc.Dropdown(
                                id="mc-pair-select",
                                options=[{"label": p, "value": p} for p in get_pairs()],
                                value=_first,
                                clearable=False,
                            ),
                        ]),
                        html.Div(className="bt-form-grid", children=[
                            html.Div(children=[
                                html.Label("Simulations", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                                   "marginBottom": "4px", "letterSpacing": "1px",
                                                                   "textTransform": "uppercase"}),
                                dcc.Input(id="mc-sims-input", type="number", value=500,
                                          min=100, max=5000, className="bt-input-new", style={"width": "100%"}),
                            ]),
                            html.Div(children=[
                                html.Label("Backtest Days", style={"fontSize": "9px", "color": MUTED, "display": "block",
                                                                    "marginBottom": "4px", "letterSpacing": "1px",
                                                                    "textTransform": "uppercase"}),
                                dcc.Input(id="mc-days-input", type="number", value=90,
                                          min=30, max=365, className="bt-input-new", style={"width": "100%"}),
                            ]),
                        ]),
                        html.Button("▶  RUN MONTE CARLO", id="mc-run-btn", n_clicks=0,
                                    className="bt-run-btn-new"),
                        html.Div(id="mc-result", className="bt-result-box",
                                 children="Configure and click Run Monte Carlo."),
                    ]),
                ]),
                html.Div(className="aria-card", style={"marginTop": "10px", "minHeight": "280px"}, children=[
                    html.Div("PROBABILITY BANDS (P05 / P50 / P95)", className="card-header"),
                    dcc.Graph(id="mc-bands-chart", style={"height": "260px"},
                              config={"displayModeBar": False}),
                ]),
            ],
        ),
    ]


def _page_pair_profiles() -> list:
    """Pair profiles page: per-pair stats with best session, score range, avg win."""
    return [
        html.Div(
            style={"flex": "1", "padding": "12px", "overflow": "auto"},
            children=[
                html.Div(className="aria-card", children=[
                    html.Div("PAIR PROFILES", className="card-header"),
                    html.Div(id="pair-profiles-panel", style={"padding": "10px"}),
                ]),
                html.Div(className="aria-card", style={"marginTop": "10px"}, children=[
                    html.Div("KNOWLEDGE BASE SEARCH", className="card-header"),
                    html.Div(style={"padding": "10px"}, children=[
                        html.Div(style={"display": "flex", "gap": "8px", "marginBottom": "8px"}, children=[
                            dcc.Input(
                                id="kb-search-input",
                                placeholder="Search vault notes…",
                                debounce=True,
                                style={"flex": "1", "background": "rgba(255,255,255,.05)",
                                       "border": "1px solid rgba(255,255,255,.1)", "color": "#F1F5F9",
                                       "fontSize": "12px", "padding": "7px 10px", "borderRadius": "6px",
                                       "outline": "none", "fontFamily": "'JetBrains Mono',monospace"},
                                n_submit=0,
                            ),
                            html.Button("SEARCH", id="kb-search-btn", n_clicks=0,
                                        className="bt-run-btn-new", style={"flex": "0 0 90px", "margin": "0"}),
                        ]),
                        html.Div(id="kb-search-results", style={"color": MUTED, "fontSize": "11px"}),
                    ]),
                ]),
                html.Div(className="aria-card", style={"marginTop": "10px"}, children=[
                    html.Div("DAILY / MONTHLY RETURNS HEATMAP", className="card-header"),
                    dcc.Graph(id="returns-heatmap", style={"height": "280px"},
                              config={"displayModeBar": False}),
                ]),
                html.Div(className="aria-card", style={"marginTop": "10px"}, children=[
                    html.Div("STRATEGY REGIME INDICATOR", className="card-header"),
                    html.Div(id="regime-indicator-panel", style={"padding": "10px"}),
                ]),
            ],
        ),
    ]


def _page_learning() -> list:
    return [
        html.Div(
            style={"flex": "1", "padding": "12px", "overflow": "auto", "display": "flex",
                   "flexDirection": "column", "gap": "10px"},
            children=[
                # ML Status widget
                html.Div(className="aria-card", children=[
                    html.Div("ML MODEL STATUS", className="card-header"),
                    html.Div(id="ml-status-panel", style={"padding": "10px"}),
                ]),
                # Strategy rankings
                html.Div(className="aria-card", children=[
                    html.Div("STRATEGY RANKINGS (Rolling Sharpe)", className="card-header"),
                    html.Div(id="strategy-rankings-panel", style={"padding": "10px"}),
                ]),
                # Adaptive learning per pair
                html.Div(className="aria-card", children=[
                    html.Div("ADAPTIVE LEARNING — PER PAIR", className="card-header"),
                    html.Div(id="adaptive-panel", style={"padding": "10px"}),
                ]),
                # Sentiment panel
                html.Div(className="aria-card", children=[
                    html.Div("SENTIMENT (REDDIT)", className="card-header"),
                    html.Div(id="sentiment-panel", style={"padding": "10px"}),
                ]),
                # Hypothesis queue
                html.Div(className="aria-card", children=[
                    html.Div("HYPOTHESIS QUEUE", className="card-header"),
                    html.Div(id="hypothesis-queue-panel", style={"padding": "10px"}),
                ]),
                # Mistake detector
                html.Div(className="aria-card", children=[
                    html.Div("MISTAKE DETECTOR", className="card-header"),
                    html.Div(id="mistake-detector-panel", style={"padding": "10px"}),
                ]),
                # Performance attribution
                html.Div(className="aria-card", children=[
                    html.Div("PERFORMANCE ATTRIBUTION (Feature → Win Rate)", className="card-header"),
                    html.Div(id="perf-attribution-panel", style={"padding": "10px"}),
                ]),
                # Autonomous pipeline
                html.Div(className="aria-card", children=[
                    html.Div(className="card-header", children=[
                        html.Span("AUTONOMOUS PIPELINE"),
                        html.Button(
                            "▶ RUN NOW",
                            id="pipeline-run-btn",
                            n_clicks=0,
                            className="tf-btn",
                            style={"fontSize": "9px", "padding": "2px 8px"},
                        ),
                    ]),
                    html.Div(id="pipeline-panel", style={"padding": "10px"}),
                ]),
                # A/B tests
                html.Div(className="aria-card", children=[
                    html.Div("A/B STRATEGY TESTS", className="card-header"),
                    html.Div(id="ab-test-panel", style={"padding": "10px"}),
                ]),
                # Paper trading
                html.Div(className="aria-card", children=[
                    html.Div("PAPER TRADING", className="card-header"),
                    html.Div(id="paper-trading-panel", style={"padding": "10px"}),
                ]),
            ],
        ),
    ]


def _page_risk() -> list:
    def _gauge_bar(label: str, el_id: str) -> html.Div:
        return html.Div(style={"marginBottom": "10px"}, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "3px"}, children=[
                html.Span(label, style={"fontSize": "9px", "color": MUTED, "textTransform": "uppercase", "letterSpacing": "1px"}),
                html.Span(id=f"{el_id}-label", style={"fontSize": "9px", "color": TEXT, "fontFamily": "'JetBrains Mono',monospace"}),
            ]),
            html.Div(style={"background": "rgba(255,255,255,.06)", "borderRadius": "3px", "height": "6px", "overflow": "hidden"}, children=[
                html.Div(id=f"{el_id}-bar", style={"height": "100%", "borderRadius": "3px", "transition": "width .5s"}),
            ]),
        ])

    return [
        html.Div(
            style={"flex": "1", "padding": "12px", "overflow": "auto", "display": "flex",
                   "flexDirection": "column", "gap": "10px"},
            children=[
                # Row 1: trading gate status
                html.Div(className="aria-card", children=[
                    html.Div("TRADING GATE", className="card-header"),
                    html.Div(id="risk-gate-panel", style={"padding": "10px"}),
                ]),

                # Row 2: drawdown gauges
                html.Div(className="aria-card", children=[
                    html.Div("DRAWDOWN LIMITS", className="card-header"),
                    html.Div(style={"padding": "12px 14px"}, children=[
                        _gauge_bar("Daily PnL vs Max Loss", "risk-daily"),
                        _gauge_bar("Weekly Drawdown (limit 6%)", "risk-weekly"),
                        _gauge_bar("Monthly Drawdown (limit 10%)", "risk-monthly"),
                        _gauge_bar("Leverage vs Max 500×", "risk-leverage"),
                    ]),
                ]),

                # Row 3: open exposure
                html.Div(className="aria-card", children=[
                    html.Div("OPEN EXPOSURE", className="card-header"),
                    html.Div(id="risk-exposure-panel", style={"padding": "10px"}),
                ]),

                # Row 4: loss streak / cooldown
                html.Div(className="aria-card", children=[
                    html.Div("LOSS STREAK & COOLDOWN", className="card-header"),
                    html.Div(id="risk-streak-panel", style={"padding": "10px"}),
                ]),
            ],
        ),
    ]


def _page_portfolio() -> list:
    return [
        html.Div(
            style={"flex": "1", "padding": "12px", "overflow": "auto", "display": "flex",
                   "flexDirection": "column", "gap": "10px"},
            children=[
                # Equity curve chart
                html.Div(
                    className="aria-card",
                    style={"minHeight": "300px"},
                    children=[
                        html.Div("EQUITY CURVE", className="card-header"),
                        dcc.Graph(
                            id="equity-chart",
                            style={"height": "280px"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                # Strategy equity curves
                html.Div(
                    className="aria-card",
                    style={"minHeight": "280px"},
                    children=[
                        html.Div("STRATEGY EQUITY CURVES", className="card-header"),
                        dcc.Graph(
                            id="strategy-equity-chart",
                            style={"height": "220px"},
                            config={"displayModeBar": False},
                        ),
                        html.Div(id="strategy-equity-stats", style={"padding": "0 10px 10px"}),
                    ],
                ),
                # Correlation matrix (positions)
                html.Div(className="aria-card", children=[
                    html.Div("POSITION CORRELATION", className="card-header"),
                    html.Div(id="correlation-panel", style={"padding": "10px"}),
                ]),
            ],
        ),
    ]


def _page_settings() -> list:
    def _field(label: str, el_id: str, type_: str = "number", step: str = "0.1", placeholder: str = "") -> html.Div:
        return html.Div(style={"marginBottom": "12px"}, children=[
            html.Label(label, style={"fontSize": "9px", "color": MUTED, "textTransform": "uppercase",
                                     "letterSpacing": "1px", "display": "block", "marginBottom": "4px"}),
            dcc.Input(
                id=el_id, type=type_, step=step, placeholder=placeholder,
                style={"width": "100%", "background": "rgba(255,255,255,.05)",
                       "border": "1px solid rgba(255,255,255,.1)", "color": TEXT,
                       "fontSize": "12px", "padding": "7px 10px", "borderRadius": "6px",
                       "outline": "none", "fontFamily": "'JetBrains Mono',monospace"},
                debounce=False,
            ),
        ])

    return [
        html.Div(
            style={"flex": "1", "padding": "12px", "overflow": "auto", "display": "flex",
                   "flexDirection": "column", "gap": "10px"},
            children=[
                # Trading parameters
                html.Div(className="aria-card", children=[
                    html.Div("TRADING PARAMETERS", className="card-header"),
                    html.Div(style={"padding": "12px 14px", "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr", "gap": "0 24px"}, children=[
                        _field("Risk Per Trade (%)", "cfg-risk-pct", step="0.1"),
                        _field("Daily Target (%)", "cfg-daily-target", step="0.1"),
                        _field("Max Daily Loss (%)", "cfg-max-loss", step="0.1"),
                        _field("Emergency Drawdown (%)", "cfg-emergency-dd", step="0.1"),
                        _field("Max Concurrent Trades", "cfg-max-concurrent", step="1"),
                        _field("Max Trades Per Day", "cfg-max-daily-trades", step="1"),
                    ]),
                ]),

                # Signal filters
                html.Div(className="aria-card", children=[
                    html.Div("SIGNAL FILTERS", className="card-header"),
                    html.Div(style={"padding": "12px 14px", "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr", "gap": "0 24px"}, children=[
                        _field("Min Signal Score", "cfg-min-score", step="1"),
                        _field("Max Spread (pips)", "cfg-max-spread", step="0.1"),
                        _field("News Buffer (minutes)", "cfg-news-buffer", step="1"),
                        html.Div(style={"marginBottom": "12px"}, children=[
                            html.Label("News Filter", style={"fontSize": "9px", "color": MUTED,
                                                              "textTransform": "uppercase",
                                                              "letterSpacing": "1px", "display": "block", "marginBottom": "4px"}),
                            dcc.Checklist(
                                id="cfg-news-filter",
                                options=[{"label": "  Enabled", "value": "on"}],
                                inputStyle={"marginRight": "6px"},
                                labelStyle={"fontSize": "12px", "color": TEXT},
                            ),
                        ]),
                    ]),
                ]),

                # System info (read-only)
                html.Div(className="aria-card", children=[
                    html.Div("SYSTEM INFO", className="card-header"),
                    html.Div(id="settings-sysinfo", style={"padding": "12px 14px"}),
                ]),

                # Save button + status
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px"}, children=[
                    html.Button("SAVE & APPLY", id="settings-save-btn", n_clicks=0,
                                className="bt-run-btn-new", style={"width": "160px"}),
                    html.Div(id="settings-save-status", style={"fontSize": "11px", "color": MUTED}),
                ]),
            ],
        ),
    ]


def _write_env_key(key: str, value: str) -> None:
    """Update or append KEY=value in .env, preserving comments."""
    env_path = settings.model_config.get("env_file", ".env")
    if not env_path:
        env_path = ".env"
    from pathlib import Path
    p = Path(env_path)
    if not p.exists():
        p.write_text(f"{key}={value}\n", encoding="utf-8")
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    upper = key.upper()
    found = False
    for i, line in enumerate(lines):
        stripped = line.split("#")[0].strip()
        if "=" in stripped:
            lk = stripped.split("=")[0].strip().upper()
            if lk == upper:
                lines[i] = f"{key.upper()}={value}"
                found = True
                break
    if not found:
        lines.append(f"{key.upper()}={value}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    id="aria-root",
    style={"height": "100vh", "display": "flex", "overflow": "hidden", "background": BG},
    children=[
        dcc.Store(id="active-page",      data="dashboard"),
        dcc.Store(id="tf-store",         data="M15"),
        dcc.Store(id="chart-pair",       data=_first),
        dcc.Store(id="pairs-store",      data=get_pairs()),
        dcc.Store(id="emg-close-arm",    data=False),
        dcc.Store(id="wfo-result-store", data=None),
        dcc.Store(id="mc-result-store",  data=None),
        dcc.Interval(id="interval", interval=15000, n_intervals=0),
        dcc.Interval(id="bt-poll",  interval=5000,  n_intervals=0),
        dcc.Interval(id="wfo-poll", interval=5000,  n_intervals=0),
        dcc.Interval(id="mc-poll",  interval=5000,  n_intervals=0),

        dbc.Toast(
            id="action-toast",
            header="Trade Action",
            is_open=False,
            duration=5000,
            icon="success",
            style={
                "position": "fixed", "top": 56, "right": 16, "width": 340,
                "zIndex": 9999, "background": PANEL2,
                "border": "1px solid rgba(124,58,237,.35)",
                "color": TEXT, "fontSize": "12px",
            },
        ),

        _sidebar(),

        html.Div(
            style={"flex": "1", "display": "flex", "flexDirection": "column",
                   "minWidth": "0", "overflow": "hidden"},
            children=[
                html.Div(id="top-bar", className="aria-header"),

                html.Div(
                    style={"flex": "1", "display": "flex", "overflow": "hidden"},
                    children=[
                        html.Div(
                            style={"flex": "1", "display": "flex", "flexDirection": "column",
                                   "overflow": "hidden", "minWidth": "0"},
                            children=[
                                html.Div(id="page-dashboard",   style={"display": "flex", "flexDirection": "column", "flex": "1", "overflow": "hidden"}, children=_page_dashboard()),
                                html.Div(id="page-signals",     style={"display": "none", "flexDirection": "column", "flex": "1", "overflow": "hidden"}, children=_page_signals()),
                                html.Div(id="page-backtest",    style={"display": "none", "flexDirection": "column", "flex": "1", "overflow": "hidden"}, children=_page_backtest()),
                                html.Div(id="page-wfo",         style={"display": "none", "flexDirection": "column", "flex": "1", "overflow": "hidden"}, children=_page_wfo()),
                                html.Div(id="page-montecarlo",  style={"display": "none", "flexDirection": "column", "flex": "1", "overflow": "hidden"}, children=_page_montecarlo()),
                                html.Div(id="page-portfolio",   style={"display": "none", "flexDirection": "column", "flex": "1", "overflow": "hidden"}, children=_page_portfolio()),
                                html.Div(id="page-risk",        style={"display": "none", "flexDirection": "column", "flex": "1", "overflow": "hidden"}, children=_page_risk()),
                                html.Div(id="page-learning",    style={"display": "none", "flexDirection": "column", "flex": "1", "overflow": "hidden"}, children=_page_learning()),
                                html.Div(id="page-profiles",    style={"display": "none", "flexDirection": "column", "flex": "1", "overflow": "hidden"}, children=_page_pair_profiles()),
                                html.Div(id="page-settings",    style={"display": "none", "flexDirection": "column", "flex": "1", "overflow": "hidden"}, children=_page_settings()),
                            ],
                        ),
                        _right_panel(),
                    ],
                ),
            ],
        ),
    ],
)


# ── Navigation callbacks ───────────────────────────────────────────────────────

@app.callback(
    Output("active-page", "data"),
    Input({"type": "nav-btn", "page": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def navigate(clicks):
    triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
    if not triggered_value:
        raise dash.exceptions.PreventUpdate
    if not ctx.triggered_id or isinstance(ctx.triggered_id, str):
        raise dash.exceptions.PreventUpdate
    return ctx.triggered_id["page"]


@app.callback(
    Output({"type": "nav-btn", "page": dash.ALL}, "className"),
    Input("active-page", "data"),
)
def update_nav_classes(active: str):
    return ["nav-item active" if pid == active else "nav-item" for pid in PAGE_IDS]


@app.callback(
    Output("page-dashboard",  "style"),
    Output("page-signals",    "style"),
    Output("page-backtest",   "style"),
    Output("page-wfo",        "style"),
    Output("page-montecarlo", "style"),
    Output("page-portfolio",  "style"),
    Output("page-risk",       "style"),
    Output("page-learning",   "style"),
    Output("page-profiles",   "style"),
    Output("page-settings",   "style"),
    Input("active-page", "data"),
)
def switch_page(active: str):
    visible = {"display": "flex", "flexDirection": "column", "flex": "1", "overflow": "hidden"}
    hidden  = {"display": "none",  "flexDirection": "column", "flex": "1", "overflow": "hidden"}
    return [visible if pid == active else hidden for pid in PAGE_IDS]


# ── TF button callbacks ────────────────────────────────────────────────────────

@app.callback(
    Output("tf-store", "data"),
    Input({"type": "tf-btn", "tf": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_tf(clicks):
    triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
    if not triggered_value:
        raise dash.exceptions.PreventUpdate
    if not ctx.triggered_id or isinstance(ctx.triggered_id, str):
        raise dash.exceptions.PreventUpdate
    return ctx.triggered_id["tf"]


@app.callback(
    Output({"type": "tf-btn", "tf": dash.ALL}, "className"),
    Input("tf-store", "data"),
)
def update_tf_classes(active_tf: str):
    return ["tf-btn active" if tf == active_tf else "tf-btn" for tf in _TF_LIST]


# ── Dashboard tab (signal feed / trade log) ────────────────────────────────────

@app.callback(
    Output("signal-feed",    "style"),
    Output("trade-log",      "style"),
    Output("tab-btn-feed",   "className"),
    Output("tab-btn-log",    "className"),
    Input("tab-btn-feed",    "n_clicks"),
    Input("tab-btn-log",     "n_clicks"),
    prevent_initial_call=True,
)
def switch_dash_tab(feed_clicks, log_clicks):
    show = {"height": "100%", "overflowY": "auto"}
    hide = {"height": "100%", "overflowY": "auto", "display": "none"}
    if ctx.triggered_id == "tab-btn-feed":
        return show, hide, "tab-btn active", "tab-btn"
    if ctx.triggered_id == "tab-btn-log":
        return hide, show, "tab-btn", "tab-btn active"
    raise dash.exceptions.PreventUpdate


# ── Header (top bar) ───────────────────────────────────────────────────────────

@app.callback(
    Output("top-bar", "children"),
    Input("interval",  "n_intervals"),
)
def update_top_bar(n: int) -> list:
    account    = feed.get_account_info()
    cap_status = _capital.status_dict if _capital else {}
    session_inf = _session_mgr.session_info()

    balance = account.get("balance", settings.account_balance) if account else settings.account_balance
    equity  = account.get("equity",  balance)                  if account else balance
    pnl     = cap_status.get("realized_pnl",  0.0)
    trades  = cap_status.get("trades_taken",   0)
    wins    = cap_status.get("win_rate",        0.0)
    halted  = cap_status.get("halted",          False)
    pnl_col = GREEN if pnl >= 0 else RED

    ticks = []
    for pair in get_pairs():
        tick = feed.get_tick(pair)
        if tick:
            bid = tick.get("bid", 0)
            ask = tick.get("ask", 0)
            mid = (bid + ask) / 2 if bid and ask else 0
            is_jpy = "JPY" in pair.upper()
            is_xau = "XAU" in pair.upper() or "GOLD" in pair.upper()
            spread_pips = round((ask - bid) / (0.01 if is_jpy else 0.0001), 1) if bid and ask else 0
            price_str  = f"{mid:.3f}" if (is_jpy or is_xau) else f"{mid:.5f}"
            spread_str = f"{spread_pips}p"
            sp_cls     = "ticker-spread tc-dn" if spread_pips and spread_pips > 2 else "ticker-spread tc-up"
        else:
            price_str  = "—"
            spread_str = ""
            sp_cls     = "ticker-spread"

        ticks.append(html.Div(className="ticker-item", children=[
            html.Span(pair.rstrip("m"), className="ticker-pair"),
            html.Span(price_str,        className="ticker-price"),
            html.Span(spread_str,       className=sp_cls),
        ]))

    if halted:
        badge = html.Span("HALTED",  className="badge-halt", style={"marginRight": "10px"})
    elif _live_mode:
        badge = html.Span("LIVE",    className="badge-live", style={"marginRight": "10px"})
    else:
        badge = html.Span("DRY RUN", className="badge-dry",  style={"marginRight": "10px"})

    scan_time = get_last_scan_time()
    scan_str  = scan_time.strftime("%H:%M:%S") if scan_time else "—"

    return [
        badge,
        html.Div(ticks, className="ticker-strip"),
        html.Div(className="header-acct", children=[
            html.Div(className="acct-item", children=[
                html.Span("BALANCE",  className="al"),
                html.Span(f"${balance:,.2f}", className="av"),
            ]),
            html.Div(className="acct-item", children=[
                html.Span("EQUITY",   className="al"),
                html.Span(f"${equity:,.2f}",  className="av"),
            ]),
            html.Div(className="acct-item", children=[
                html.Span("DAY P&L",  className="al"),
                html.Span(f"${pnl:+.2f}", className="av", style={"color": pnl_col}),
            ]),
            html.Div(className="acct-item", children=[
                html.Span("SESSION",  className="al"),
                html.Span(session_inf["label"].split()[-1], className="av",
                          style={"color": YELLOW, "fontSize": "11px"}),
            ]),
            html.Div(className="acct-item", children=[
                html.Span("SCAN",     className="al"),
                html.Span(scan_str,   className="av", style={"color": MUTED2, "fontSize": "11px"}),
            ]),
        ]),
    ]


# ── Dashboard stat cards ───────────────────────────────────────────────────────

@app.callback(
    Output("dash-stat-balance", "children"),
    Output("dash-stat-equity",  "children"),
    Output("dash-stat-pnl",     "children"),
    Output("dash-stat-wr",      "children"),
    Output("dash-stat-pnl",     "style"),
    Input("interval", "n_intervals"),
)
def update_dash_stats(n: int):
    account    = feed.get_account_info()
    cap_status = _capital.status_dict if _capital else {}
    balance = account.get("balance", settings.account_balance) if account else settings.account_balance
    equity  = account.get("equity",  balance)                  if account else balance
    pnl     = cap_status.get("realized_pnl", 0.0)
    wins    = cap_status.get("win_rate",       0.0)
    pnl_col = GREEN if pnl >= 0 else RED
    return (
        f"${balance:,.2f}",
        f"${equity:,.2f}",
        f"${pnl:+.2f}",
        f"{wins:.0f}%",
        {"fontFamily": "'JetBrains Mono',monospace", "fontSize": "19px",
         "fontWeight": "700", "color": pnl_col, "display": "block"},
    )


# ── ARIA Insight (session + scan info) ────────────────────────────────────────

@app.callback(
    Output("aria-insight", "children"),
    Input("interval", "n_intervals"),
)
def update_insight(n: int):
    session_info = _session_mgr.session_info()
    scan_time    = get_last_scan_time()
    scan_str     = scan_time.strftime("%H:%M:%S") if scan_time else "—"
    signals      = get_signals()
    sig_count    = len(signals)

    session_label = session_info.get("label", "Unknown")
    session_col   = YELLOW if "LONDON" in session_label.upper() or "OVERLAP" in session_label.upper() else MUTED

    return [
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "7px"},
            children=[
                html.Span(session_label, style={"fontSize": "11px", "fontWeight": "600",
                                                 "color": session_col}),
                html.Span(f"{sig_count} signal{'s' if sig_count != 1 else ''}", className="badge-live"),
            ],
        ),
        html.Div(
            style={"display": "flex", "justifyContent": "space-between"},
            children=[
                html.Span(f"Last scan {scan_str}", style={"fontSize": "10px", "color": MUTED2}),
                html.Span("ARIA v1.0", style={"fontSize": "10px", "color": MUTED2}),
            ],
        ),
    ]


# ── Market overview ────────────────────────────────────────────────────────────

@app.callback(
    Output("market-overview", "children"),
    Input("interval", "n_intervals"),
)
def update_market_overview(n: int) -> list:
    cards = []
    for pair in get_pairs():
        tick = feed.get_tick(pair)
        if tick:
            bid = tick.get("bid", 0)
            ask = tick.get("ask", 0)
            mid = (bid + ask) / 2 if bid and ask else 0
            is_jpy = "JPY" in pair.upper()
            is_xau = "XAU" in pair.upper() or "GOLD" in pair.upper()
            price_str = f"{mid:.3f}" if (is_jpy or is_xau) else f"{mid:.5f}"
            spread_p  = round((ask - bid) / (0.01 if is_jpy else 0.0001), 1) if bid and ask else 0
        else:
            price_str = "—"
            spread_p  = 0

        signals   = get_signals()
        has_sig   = pair in signals
        sig_col   = (GREEN if signals[pair].direction == "long" else RED) if has_sig else None
        border    = f"1px solid {sig_col}55" if has_sig else "1px solid rgba(255,255,255,.05)"

        cards.append(
            html.Div(
                className="market-card",
                id={"type": "market-card", "pair": pair},
                n_clicks=0,
                style={"borderColor": sig_col + "55"} if has_sig else {},
                children=[
                    html.Span(pair.rstrip("m"), className="market-pair"),
                    html.Span(price_str, className="market-price",
                              style={"color": sig_col} if has_sig else {}),
                    html.Span(f"{spread_p}p", className="market-spread"),
                ],
            )
        )
    return cards


# ── Watchlist ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("watchlist-panel", "children"),
    Output("pair-select",     "options"),
    Input("interval",         "n_intervals"),
    Input("pairs-store",      "data"),
    State("pair-select",      "value"),
)
def update_watchlist(n: int, pairs_data: list, selected: str) -> tuple:
    current_pairs = get_pairs()
    rows = []
    for pair in current_pairs:
        tick = feed.get_tick(pair)
        if tick:
            bid = tick.get("bid", 0)
            ask = tick.get("ask", 0)
            mid = (bid + ask) / 2 if bid and ask else 0
            is_jpy = "JPY" in pair.upper()
            is_xau = "XAU" in pair.upper() or "GOLD" in pair.upper()
            price_str  = f"{mid:.3f}" if (is_jpy or is_xau) else f"{mid:.5f}"
            spread_p   = round((ask - bid) / (0.01 if is_jpy else 0.0001), 1) if bid and ask else 0
            spread_str = f"{spread_p}p"
        else:
            price_str  = "—"
            spread_str = ""

        stats = adaptive.get_stats(pair)
        if stats and stats.total_trades >= 3:
            wr  = stats.win_rate
            cls = "learn-badge learn-up" if wr >= 55 else ("learn-badge learn-dn" if wr < 40 else "learn-badge learn-neutral")
            badge = html.Span(f"{wr:.0f}%W", className=cls)
        else:
            badge = None

        active_cls = "pw-row-new active" if pair == selected else "pw-row-new"
        children = [
            html.Span(pair.rstrip("m"), className="pw-name"),
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                html.Span(price_str,  className="pw-price"),
                html.Span(spread_str, className="pw-spread"),
            ]),
        ]
        if badge:
            children.append(badge)

        rows.append(
            html.Div(
                className=active_cls,
                id={"type": "pair-btn", "pair": pair},
                n_clicks=0,
                children=children,
            )
        )

    return rows, [{"label": p, "value": p} for p in current_pairs]


@app.callback(
    Output("pairs-store",   "data"),
    Output("pair-add-msg",  "children"),
    Input("pair-add-btn",   "n_clicks"),
    Input("pair-add-input", "n_submit"),
    State("pair-add-input", "value"),
    prevent_initial_call=True,
)
def handle_pair_add(n_clicks: int, n_submit: int, symbol: str) -> tuple:
    triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
    if not triggered_value:
        raise dash.exceptions.PreventUpdate
    if not symbol or not symbol.strip():
        return get_pairs(), "Enter a symbol name"
    ok, msg = add_pair(symbol.strip())
    return get_pairs(), msg


@app.callback(
    Output("pairs-store",  "data",     allow_duplicate=True),
    Output("pair-add-msg", "children", allow_duplicate=True),
    Input({"type": "rm-pair-btn", "pair": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def handle_pair_remove(clicks) -> tuple:
    triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
    if not triggered_value:
        raise dash.exceptions.PreventUpdate
    if not ctx.triggered_id or isinstance(ctx.triggered_id, str):
        raise dash.exceptions.PreventUpdate
    pair = ctx.triggered_id["pair"]
    ok, msg = remove_pair(pair)
    return get_pairs(), msg


@app.callback(
    Output("pair-select", "value"),
    Input({"type": "pair-btn",    "pair": dash.ALL}, "n_clicks"),
    Input({"type": "market-card", "pair": dash.ALL}, "n_clicks"),
    State("pair-select",  "value"),
    prevent_initial_call=True,
)
def select_pair_from_watchlist(clicks, market_clicks, current: str) -> str:
    if not ctx.triggered_id or isinstance(ctx.triggered_id, str):
        raise dash.exceptions.PreventUpdate
    triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
    if not triggered_value:
        raise dash.exceptions.PreventUpdate
    return ctx.triggered_id["pair"]


# ── Chart ──────────────────────────────────────────────────────────────────────

@app.callback(
    Output("main-chart",       "figure"),
    Output("chart-price-label", "children"),
    Input("interval",   "n_intervals"),
    Input("pair-select", "value"),
    Input("tf-store",    "data"),
)
def update_chart(n: int, pair: str, tf: str):
    df = feed.get_candles(pair, tf, count=120)

    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )

    if df.empty:
        empty_fig.add_annotation(text="No data — MT5 not connected",
                                  xref="paper", yref="paper", x=0.5, y=0.5,
                                  font=dict(color=MUTED, size=13), showarrow=False)
        return empty_fig, "—"

    df = apply_all(df)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name=pair,
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        increasing_fillcolor=GREEN,
        decreasing_fillcolor=RED,
        line=dict(width=1),
    ))

    for col, color, name in [("ema21", YELLOW, "EMA21"), ("ema50", ACCENT2, "EMA50"), ("ema200", MUTED, "EMA200")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines",
                line=dict(color=color, width=1), name=name, opacity=0.85,
            ))

    signals = get_signals()
    if pair in signals:
        sig   = signals[pair]
        tick  = feed.get_tick(pair)
        price = tick.get("mid", df["close"].iloc[-1]) if tick else df["close"].iloc[-1]
        arrow_color = GREEN if sig.direction == "long" else RED
        symbol      = "triangle-up" if sig.direction == "long" else "triangle-down"
        label       = f"{'BUY' if sig.direction == 'long' else 'SELL'} {sig.score:.0f}"
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[price],
            mode="markers+text",
            marker=dict(size=12, color=arrow_color, symbol=symbol),
            text=[label], textposition="top right",
            textfont=dict(color=arrow_color, size=10),
            name=label, showlegend=False,
        ))

    last_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2] if len(df) > 1 else last_close
    chg        = last_close - prev_close
    chg_pct    = chg / prev_close * 100 if prev_close else 0
    chg_color  = GREEN if chg >= 0 else RED
    is_jpy     = "JPY" in pair.upper()
    is_xau     = "XAU" in pair.upper() or "GOLD" in pair.upper()
    price_fmt  = f"{last_close:.3f}" if (is_jpy or is_xau) else f"{last_close:.5f}"

    price_label = [
        html.Span(price_fmt,
                  style={"color": chg_color, "fontWeight": "700", "fontSize": "13px",
                         "fontFamily": "'JetBrains Mono',monospace"}),
        html.Span(f"  {chg:+.5f}  ({chg_pct:+.2f}%)",
                  style={"color": chg_color, "fontSize": "10px", "marginLeft": "8px"}),
    ]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        margin=dict(l=50, r=10, t=8, b=30),
        xaxis=dict(rangeslider=dict(visible=False), color=MUTED,
                   gridcolor="rgba(255,255,255,.04)", showgrid=True, zeroline=False),
        yaxis=dict(color=MUTED, gridcolor="rgba(255,255,255,.04)",
                   showgrid=True, zeroline=False, side="right"),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PANEL2, font_color=TEXT, font_size=11),
    )
    return fig, price_label


# ── Signals panel (right panel) ────────────────────────────────────────────────

@app.callback(
    Output("signals-header", "children"),
    Output("signals-panel",  "children"),
    Input("interval", "n_intervals"),
)
def update_signals(n: int):
    badge = (html.Span("LIVE", className="badge-live") if _live_mode
             else html.Span("DRY RUN", className="badge-dry"))
    signals = get_signals()

    if not signals:
        session_info = _session_mgr.session_info()
        return badge, [html.Div(f"No signals — {session_info['label']}", className="no-data-new")]

    sorted_sigs = sorted(signals.values(), key=lambda s: s.score, reverse=True)
    cards = []
    for sig in sorted_sigs[:5]:
        is_long   = sig.direction == "long"
        dir_label = "▲ BUY" if is_long else "▼ SELL"
        sig_col   = GREEN if is_long else RED
        reason    = sig.entry_reason[:55] + "…" if len(sig.entry_reason) > 55 else sig.entry_reason

        cards.append(html.Div(className="sig-card-new", children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "alignItems": "center", "marginBottom": "3px"}, children=[
                html.Span(sig.pair.rstrip("m"), className="sig-pair-name"),
                html.Span(f"{sig.score:.0f}", className="sig-score-badge", style={"color": sig_col}),
            ]),
            html.Span(dir_label, style={"fontSize": "10px", "fontWeight": "700", "color": sig_col}),
            html.Div(className="sig-bar-wrap", children=[
                html.Div(className="sig-bar", style={"width": f"{sig.score}%", "background": sig_col}),
            ]),
            html.Div(reason, className="sig-reason-text"),
            html.Button(
                f"{'BUY' if is_long else 'SELL'} {sig.pair.rstrip('m')}",
                className="btn-buy-new" if is_long else "btn-sell-new",
                id={"type": "exec-btn", "pair": sig.pair},
                n_clicks=0,
            ),
        ]))
    return badge, cards


# ── Positions (right panel) ────────────────────────────────────────────────────

@app.callback(
    Output("positions-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_positions(n: int) -> list:
    positions = feed.get_positions()
    if not positions:
        return [html.Div("No open positions", className="no-data-new")]

    rows = []
    for pos in positions:
        pnl     = pos["pnl"]
        pnl_pct = pnl / settings.account_balance * 100 if settings.account_balance else 0
        is_up   = pnl >= 0
        pnl_cls = "pos-pnl up" if is_up else "pos-pnl dn"
        dir_col = GREEN if pos["direction"] == "long" else RED
        dir_sym = "▲" if pos["direction"] == "long" else "▼"
        is_jpy  = "JPY" in pos["pair"].upper()
        entry_f = f"{pos['entry']:.3f}" if is_jpy else f"{pos['entry']:.5f}"
        sl_f    = f"{pos['sl']:.3f}"    if is_jpy else f"{pos['sl']:.5f}"

        rows.append(html.Div(className="pos-row-new", children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "alignItems": "baseline", "marginBottom": "2px"}, children=[
                html.Span([
                    html.Span(dir_sym, style={"color": dir_col, "marginRight": "4px"}),
                    html.Span(pos["pair"].rstrip("m"), style={"fontWeight": "600", "color": TEXT}),
                ]),
                html.Span(f"${pnl:+.2f} ({pnl_pct:+.1f}%)", className=pnl_cls),
            ]),
            html.Div(f"E:{entry_f}  SL:{sl_f}  {pos['lots']}L", className="pos-meta"),
            html.Button(
                "✕ CLOSE",
                className="btn-close-new",
                id={"type": "close-btn", "ticket": str(pos["ticket"])},
                n_clicks=0,
            ),
        ]))
    return rows


# ── Risk / Stats (right panel) ─────────────────────────────────────────────────

@app.callback(
    Output("stats-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_stats(n: int) -> html.Div:
    cap      = _capital.status_dict if _capital else {}
    day_pnl  = cap.get("realized_pnl",   0.0)
    target   = cap.get("target_amount",   0.0)
    max_loss = cap.get("max_loss_amount", 0.0)
    trades   = cap.get("trades_taken",    0)
    wins     = cap.get("win_rate",        0.0)
    halted   = cap.get("halted",          False)
    cooldown = cap.get("cooldown_active", False)
    cduntil  = cap.get("cooldown_until",  None)
    streak   = cap.get("consecutive_losses", 0)
    pnl_col  = GREEN if day_pnl >= 0 else RED
    halt_col = RED if (halted or cooldown) else GREEN

    streak_label = f"{streak}/3  CD:{cduntil}" if cooldown else ("NO" if not halted else "YES")

    return html.Div(className="stat-grid-new", children=[
        html.Div(className="stat-cell-new", children=[
            html.Span("DAY P&L",  className="stat-l"),
            html.Span(f"${day_pnl:+.2f}", className="stat-v", style={"color": pnl_col}),
        ]),
        html.Div(className="stat-cell-new", children=[
            html.Span("TARGET",   className="stat-l"),
            html.Span(f"${target:.2f}", className="stat-v"),
        ]),
        html.Div(className="stat-cell-new", children=[
            html.Span("MAX LOSS", className="stat-l"),
            html.Span(f"${max_loss:.2f}", className="stat-v", style={"color": MUTED}),
        ]),
        html.Div(className="stat-cell-new", children=[
            html.Span("TRADES",   className="stat-l"),
            html.Span(
                f"{trades}{'  ⚡' if adaptive.is_global_conservative() else ''}",
                className="stat-v",
                style={"color": RED if adaptive.is_global_conservative() else TEXT},
            ),
        ]),
        html.Div(className="stat-cell-new", children=[
            html.Span("WIN RATE", className="stat-l"),
            html.Span(f"{wins:.0f}%", className="stat-v",
                      style={"color": GREEN if wins >= 50 else RED}),
        ]),
        html.Div(className="stat-cell-new", children=[
            html.Span("HALTED",   className="stat-l"),
            html.Span(streak_label, className="stat-v", style={"color": halt_col}),
        ]),
    ])


# ── Risk meters ───────────────────────────────────────────────────────────────

@app.callback(
    Output("risk-meters", "children"),
    Input("interval",     "n_intervals"),
)
def update_risk_meters(n: int) -> html.Div:
    cap = _capital.status_dict if _capital else {}

    day_pnl    = cap.get("realized_pnl",    0.0)
    max_loss   = cap.get("max_loss_amount",  1.0) or 1.0
    target     = cap.get("target_amount",    1.0) or 1.0
    week_dd    = cap.get("weekly_dd_pct",    0.0)
    week_lim   = cap.get("weekly_dd_limit",  6.0) or 6.0
    month_dd   = cap.get("monthly_dd_pct",   0.0)
    month_lim  = cap.get("monthly_dd_limit", 10.0) or 10.0
    leverage   = cap.get("current_leverage", 0.0)
    max_lev    = cap.get("max_leverage",     500.0) or 500.0

    def _bar_color(pct: float) -> str:
        if pct >= 80:
            return RED
        if pct >= 50:
            return "#F59E0B"
        return GREEN

    def _meter(label: str, value: float, limit: float, fmt: str = ".1f", unit: str = "%") -> html.Div:
        pct    = min(100.0, abs(value) / limit * 100) if limit else 0.0
        color  = _bar_color(pct)
        v_str  = f"{value:{fmt}}{unit}"
        return html.Div(className="rm-row", children=[
            html.Div(className="rm-header", children=[
                html.Span(label, className="rm-label"),
                html.Span(v_str, className="rm-pct", style={"color": color}),
            ]),
            html.Div(className="rm-track", children=[
                html.Div(className="rm-fill", style={"width": f"{pct:.1f}%", "background": color}),
            ]),
        ])

    day_loss_used = max(0.0, -day_pnl)
    day_gain      = max(0.0, day_pnl)

    return html.Div(className="risk-meters", children=[
        _meter("Day Loss",    day_loss_used, max_loss,  "+.2f", "$"),
        _meter("Day Target",  day_gain,      target,    "+.2f", "$"),
        _meter("Weekly DD",   week_dd,       week_lim,  ".1f",  "%"),
        _meter("Monthly DD",  month_dd,      month_lim, ".1f",  "%"),
        _meter("Leverage",    leverage,      max_lev,   ".0f",  "×"),
    ])


# ── Trade log ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("trade-log", "children"),
    Input("interval",   "n_intervals"),
)
def update_trade_log(n: int) -> html.Div:
    try:
        deals = feed.get_history_deals(days=1)
    except Exception:
        deals = []

    if not deals:
        return html.Div("No closed trades today", className="no-data-new")

    rows = [
        html.Div(className="tlog-hdr-new", children=[
            html.Span("TIME"), html.Span("PAIR"), html.Span("DIR"),
            html.Span("LOTS"), html.Span("ENTRY"), html.Span("P&L"),
        ])
    ]
    for d in reversed(deals[-30:]):
        pnl     = d.get("profit", 0)
        t       = d.get("time",   "")
        time_s  = t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)[:5]
        pnl_col = GREEN if pnl >= 0 else RED
        dir_col = GREEN if d.get("direction") == "long" else RED
        rows.append(html.Div(className="tlog-row-new", children=[
            html.Span(time_s,                                style={"color": MUTED}),
            html.Span(d.get("pair", "—").rstrip("m"),        style={"fontWeight": "600"}),
            html.Span(d.get("direction", "—").upper(),       style={"color": dir_col}),
            html.Span(f"{d.get('lots', 0):.2f}"),
            html.Span(f"{d.get('entry', 0):.5f}",           style={"fontFamily": "'JetBrains Mono',monospace"}),
            html.Span(f"${pnl:+.2f}",                       style={"color": pnl_col, "fontWeight": "700",
                                                                     "fontFamily": "'JetBrains Mono',monospace"}),
        ]))
    return html.Div(rows)


# ── Execute / Close ────────────────────────────────────────────────────────────

@app.callback(
    Output("action-toast", "is_open"),
    Output("action-toast", "children"),
    Output("action-toast", "icon"),
    Input({"type": "exec-btn",   "pair":   dash.ALL}, "n_clicks"),
    Input({"type": "close-btn",  "ticket": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def handle_trade_action(exec_clicks, close_clicks):
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate

    triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
    if not triggered_value:
        raise dash.exceptions.PreventUpdate

    triggered = ctx.triggered_id

    if isinstance(triggered, dict) and triggered.get("type") == "exec-btn":
        pair    = triggered["pair"]
        signals = get_signals()
        if pair not in signals:
            return True, f"Signal for {pair} expired — rescan pending", "warning"
        sig = signals[pair]
        if not _order_mgr or not _live_mode:
            return (True,
                    f"DRY RUN: Would {'BUY' if sig.direction == 'long' else 'SELL'} "
                    f"{pair} @ score {sig.score:.0f}. Restart with --live.",
                    "info")
        try:
            from signals.entry import build_setup
            tick  = feed.get_tick(pair)
            price = tick.get("mid", 0) if tick else 0
            if not price:
                return True, f"Cannot get price for {pair}", "warning"
            df15  = apply_all(feed.get_candles(pair, "M15", count=50))
            setup = build_setup(sig, price, df15)
            if not setup:
                return True, f"Could not build trade setup for {pair}", "warning"
            ticket = _order_mgr.execute(sig, setup)
            if ticket:
                direction = "BUY" if sig.direction == "long" else "SELL"
                return (True,
                        f"✅ {direction} {pair} opened — ticket #{ticket} "
                        f"| SL {setup.sl:.5f} | TP {setup.tp1:.5f}",
                        "success")
            perm   = _capital.can_trade(pair) if _capital else None
            reason = perm.reason if perm else "Capital check failed"
            return True, f"Trade blocked: {reason}", "warning"
        except Exception as e:
            logger.error(f"Execute error: {e}")
            return True, f"Error: {str(e)[:80]}", "danger"

    if isinstance(triggered, dict) and triggered.get("type") == "close-btn":
        ticket = int(triggered["ticket"])
        if not _live_mode:
            return True, f"DRY RUN: Would close ticket #{ticket}. Restart with --live.", "info"
        try:
            from execution.mt5_client import mt5_client
            result = mt5_client.close_position(ticket)
            if result.success:
                if _capital:
                    positions = feed.get_positions()
                    pos = next((p for p in positions if p["ticket"] == ticket), None)
                    if pos and pos["pair"] in (_capital.open_positions or {}):
                        _capital.register_close(pos["pair"], pos["current"], pos["pnl"])
                return True, f"✅ Position #{ticket} closed", "success"
            return True, f"Close failed: {result.error}", "danger"
        except Exception as e:
            logger.error(f"Close error: {e}")
            return True, f"Error closing #{ticket}: {str(e)[:80]}", "danger"

    raise dash.exceptions.PreventUpdate


# ── Signal feed (dashboard tab) ────────────────────────────────────────────────

def _build_signal_feed_rows(limit: int = 40) -> html.Div:
    history = get_signal_history()
    if not history:
        return html.Div("Waiting for first scan…", className="no-data-new")

    rows = [
        html.Div(className="sfeed-hdr-new", children=[
            html.Span("TIME"), html.Span("PAIR"), html.Span("DIR"),
            html.Span("SCORE"), html.Span("REASON"),
        ])
    ]
    for h in history[:limit]:
        direction = h["direction"]
        score     = h["score"]
        above     = h["above_threshold"]
        is_long   = direction == "long"
        is_short  = direction == "short"
        dir_col   = GREEN if is_long else (RED if is_short else MUTED)
        dir_lbl   = "▲ BUY" if is_long else ("▼ SELL" if is_short else "→ WAIT")
        score_col = PURPLE if above else (GREEN if score >= 50 else MUTED)
        t         = h["time"]
        time_s    = t.strftime("%H:%M") if hasattr(t, "strftime") else "—"
        reason    = h["reason"][:44] + "…" if len(h["reason"]) > 44 else h["reason"]
        row_cls   = "sfeed-row-new hit" if above else "sfeed-row-new"

        rows.append(html.Div(className=row_cls, children=[
            html.Span(time_s,                  style={"color": MUTED2}),
            html.Span(h["pair"].rstrip("m"),   style={"fontWeight": "600"}),
            html.Span(dir_lbl,                 style={"color": dir_col, "fontWeight": "600"}),
            html.Span(f"{score:.0f}",          style={"color": score_col, "fontWeight": "700",
                                                       "fontFamily": "'JetBrains Mono',monospace"}),
            html.Span(reason,                  style={"color": MUTED2}),
        ]))
    return html.Div(rows)


@app.callback(
    Output("signal-feed", "children"),
    Input("interval",     "n_intervals"),
)
def update_signal_feed(n: int) -> html.Div:
    return _build_signal_feed_rows(40)


@app.callback(
    Output("signal-feed-full", "children"),
    Input("interval",          "n_intervals"),
)
def update_signal_feed_full(n: int) -> html.Div:
    return _build_signal_feed_rows(200)


# ── Backtest callbacks ─────────────────────────────────────────────────────────

def _run_backtest_thread(pair: str, days: int, score: float) -> None:
    global _bt_result
    with _bt_lock:
        _bt_result = {"status": "running", "text": f"Running {pair} {days}d…"}
    try:
        from backtest.engine import BacktestEngine
        engine = BacktestEngine(pair=pair, days=days, min_score=score)
        res    = engine.run()
        text   = (
            f"Pair   : {res.pair}\n"
            f"Trades : {res.total_trades}\n"
            f"Win %%  : {res.win_rate:.1f}%%\n"
            f"PF     : {res.profit_factor:.2f}\n"
            f"DD     : {res.max_drawdown:.1f}%%\n"
            f"Net    : ${res.net_pnl:+.2f}\n"
            f"\n{res.verdict()}"
        )
        with _bt_lock:
            _bt_result = {"status": "done", "text": text}
    except Exception as e:
        with _bt_lock:
            _bt_result = {"status": "error", "text": f"Error: {e}"}


@app.callback(
    Output("bt-result",  "children"),
    Output("bt-run-btn", "disabled"),
    Output("bt-run-btn", "children"),
    Input("bt-run-btn",  "n_clicks"),
    Input("bt-poll",     "n_intervals"),
    State("bt-pair-select", "value"),
    State("bt-days-input",  "value"),
    State("bt-score-input", "value"),
    prevent_initial_call=True,
)
def handle_backtest(n_clicks, poll, pair, days, score):
    triggered_id = ctx.triggered_id

    if triggered_id == "bt-run-btn":
        triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
        if not triggered_value:
            raise dash.exceptions.PreventUpdate
        days  = int(days  or 90)
        score = float(score or 70)
        t = threading.Thread(target=_run_backtest_thread,
                             args=(pair, days, score), daemon=True)
        t.start()
        return "Starting…", True, "⏳ RUNNING…"

    with _bt_lock:
        state = dict(_bt_result)

    if not state:
        raise dash.exceptions.PreventUpdate

    status = state.get("status", "")
    text   = state.get("text",   "")

    if status == "running":
        return text, True, "⏳ RUNNING…"
    if status in ("done", "error"):
        return text, False, "▶  RUN BACKTEST"

    raise dash.exceptions.PreventUpdate


# ── Emergency close-all ────────────────────────────────────────────────────────

@app.callback(
    Output("emg-close-btn",    "children"),
    Output("emg-close-btn",    "className"),
    Output("emg-close-status", "children"),
    Output("emg-close-arm",    "data"),
    Input("emg-close-btn",     "n_clicks"),
    State("emg-close-arm",     "data"),
    prevent_initial_call=True,
)
def handle_emergency_close(n_clicks: int, armed: bool):
    if not armed:
        return "!! CLICK AGAIN TO CONFIRM !!", "btn-emergency-armed", "", True

    # Second click: execute
    if _capital is None:
        return "EMERGENCY CLOSE ALL", "btn-emergency", "Error: capital manager unavailable", False

    try:
        from execution.emergency import execute_emergency_close
        result = execute_emergency_close(_capital, reason="Dashboard emergency close")
        msg = f"Closed {result['closed']}/{result['n_total']} | P&L: ${result['total_pnl']:+.2f} | Trading halted"
    except Exception as e:
        msg = f"Error: {e}"

    return "EMERGENCY CLOSE ALL", "btn-emergency", msg, False


# ── ML Status Widget ──────────────────────────────────────────────────────────

@app.callback(
    Output("ml-status-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_ml_status(n: int) -> html.Div:
    try:
        from ml.performance import tracker as ml_perf
        from ml.predictor import predictor as ml_pred
        stats  = ml_perf.get_stats()
        ready  = ml_pred.is_ready()
        n_samp = getattr(ml_pred, "_n_samples", "—")
    except Exception:
        stats  = {"boosted_wr": 0, "penalized_wr": 0, "neutral_wr": 0,
                  "boosted_trades": 0, "penalized_trades": 0, "neutral_trades": 0,
                  "verdict": "—"}
        ready  = False
        n_samp = "—"

    status_col  = GREEN if ready else RED
    status_txt  = "READY" if ready else "NOT TRAINED"
    verdict_col = GREEN if "positive" in stats["verdict"].lower() else (
                  RED if "negative" in stats["verdict"].lower() else MUTED)

    return html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}, children=[
        html.Div(style={"minWidth": "120px", "background": "rgba(0,0,0,.2)",
                        "borderRadius": "6px", "padding": "8px 12px"}, children=[
            html.Span("STATUS", style={"fontSize": "8px", "color": MUTED, "display": "block",
                                       "letterSpacing": "1px", "textTransform": "uppercase", "marginBottom": "4px"}),
            html.Span(status_txt, style={"color": status_col, "fontWeight": "700", "fontSize": "13px",
                                          "fontFamily": "'JetBrains Mono',monospace"}),
        ]),
        html.Div(style={"minWidth": "120px", "background": "rgba(0,0,0,.2)",
                        "borderRadius": "6px", "padding": "8px 12px"}, children=[
            html.Span("SAMPLES", style={"fontSize": "8px", "color": MUTED, "display": "block",
                                         "letterSpacing": "1px", "textTransform": "uppercase", "marginBottom": "4px"}),
            html.Span(str(n_samp), style={"color": TEXT, "fontWeight": "700", "fontSize": "13px",
                                           "fontFamily": "'JetBrains Mono',monospace"}),
        ]),
        html.Div(style={"minWidth": "120px", "background": "rgba(0,0,0,.2)",
                        "borderRadius": "6px", "padding": "8px 12px"}, children=[
            html.Span("VERDICT", style={"fontSize": "8px", "color": MUTED, "display": "block",
                                         "letterSpacing": "1px", "textTransform": "uppercase", "marginBottom": "4px"}),
            html.Span(stats["verdict"][:30], style={"color": verdict_col, "fontWeight": "600", "fontSize": "11px"}),
        ]),
        html.Div(style={"flex": "1", "minWidth": "200px", "background": "rgba(0,0,0,.2)",
                        "borderRadius": "6px", "padding": "8px 12px"}, children=[
            html.Span("BUCKETS", style={"fontSize": "8px", "color": MUTED, "display": "block",
                                         "letterSpacing": "1px", "textTransform": "uppercase", "marginBottom": "4px"}),
            html.Span(
                f"Boosted: {stats['boosted_wr']:.0f}% WR ({stats['boosted_trades']}T)  "
                f"Penalized: {stats['penalized_wr']:.0f}% WR ({stats['penalized_trades']}T)  "
                f"Neutral: {stats['neutral_wr']:.0f}% WR ({stats['neutral_trades']}T)",
                style={"fontSize": "10px", "color": MUTED2, "fontFamily": "'JetBrains Mono',monospace"},
            ),
        ]),
    ])


# ── Strategy Rankings ─────────────────────────────────────────────────────────

@app.callback(
    Output("strategy-rankings-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_strategy_rankings(n: int) -> html.Div:
    try:
        from core.strategy_scorer import score_strategies
        rows_data = score_strategies()
    except Exception:
        rows_data = []

    if not rows_data:
        return html.Div("No data — need 5+ trades per strategy", style={"color": MUTED, "fontSize": "11px"})

    header = html.Div(
        style={"display": "grid", "gridTemplateColumns": "100px 60px 60px 70px 70px 60px",
               "gap": "5px", "padding": "4px 0", "borderBottom": "1px solid rgba(255,255,255,.06)",
               "marginBottom": "4px"},
        children=[
            html.Span("STRATEGY",  style={"fontSize": "8px", "color": MUTED, "fontWeight": "600", "textTransform": "uppercase"}),
            html.Span("TRADES",    style={"fontSize": "8px", "color": MUTED, "fontWeight": "600", "textTransform": "uppercase"}),
            html.Span("WIN RATE",  style={"fontSize": "8px", "color": MUTED, "fontWeight": "600", "textTransform": "uppercase"}),
            html.Span("AVG P&L",   style={"fontSize": "8px", "color": MUTED, "fontWeight": "600", "textTransform": "uppercase"}),
            html.Span("SHARPE",    style={"fontSize": "8px", "color": MUTED, "fontWeight": "600", "textTransform": "uppercase"}),
            html.Span("STATUS",    style={"fontSize": "8px", "color": MUTED, "fontWeight": "600", "textTransform": "uppercase"}),
        ],
    )
    rows = [header]
    for r in rows_data:
        sharpe_col = GREEN if r["sharpe"] > 0 else RED
        retire_txt = "RETIRE" if r["retire"] else "ACTIVE"
        retire_col = RED if r["retire"] else GREEN
        rows.append(html.Div(
            style={"display": "grid", "gridTemplateColumns": "100px 60px 60px 70px 70px 60px",
                   "gap": "5px", "padding": "3px 0", "borderBottom": "1px solid rgba(255,255,255,.03)"},
            children=[
                html.Span(r["strategy"], style={"fontSize": "10px", "fontWeight": "600", "color": TEXT}),
                html.Span(str(r["trades"]), style={"fontSize": "10px", "color": MUTED2}),
                html.Span(f"{r['win_rate']:.0f}%", style={"fontSize": "10px",
                          "color": GREEN if r["win_rate"] >= 50 else RED}),
                html.Span(f"${r['avg_pnl']:+.4f}", style={"fontSize": "9px", "color": MUTED2,
                           "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(f"{r['sharpe']:+.3f}", style={"fontSize": "10px", "color": sharpe_col,
                           "fontWeight": "700", "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(retire_txt, style={"fontSize": "9px", "color": retire_col, "fontWeight": "600"}),
            ],
        ))
    return html.Div(rows)


# ── Adaptive learning per pair ────────────────────────────────────────────────

@app.callback(
    Output("adaptive-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_adaptive_panel(n: int) -> html.Div:
    try:
        al_stats = adaptive.all_stats()
    except Exception:
        al_stats = {}

    if not al_stats:
        return html.Div("No adaptive data yet — needs 3+ trades per pair", style={"color": MUTED, "fontSize": "11px"})

    header = html.Div(
        style={"display": "grid", "gridTemplateColumns": "90px 50px 55px 60px 70px 60px",
               "gap": "5px", "padding": "4px 0", "borderBottom": "1px solid rgba(255,255,255,.06)", "marginBottom": "4px"},
        children=[
            html.Span(c, style={"fontSize": "8px", "color": MUTED, "fontWeight": "600", "textTransform": "uppercase"})
            for c in ["PAIR", "W/L", "WIN%", "MIN SCORE", "LOT MULT", "AVG P&L"]
        ],
    )
    rows = [header]
    for pair, s in sorted(al_stats.items(), key=lambda x: x[1].win_rate, reverse=True):
        wr_col = GREEN if s.win_rate >= 50 else RED
        rows.append(html.Div(
            style={"display": "grid", "gridTemplateColumns": "90px 50px 55px 60px 70px 60px",
                   "gap": "5px", "padding": "3px 0", "borderBottom": "1px solid rgba(255,255,255,.03)"},
            children=[
                html.Span(pair.rstrip("m"), style={"fontSize": "10px", "fontWeight": "600", "color": TEXT}),
                html.Span(f"{s.wins}W/{s.losses}L", style={"fontSize": "9px", "color": MUTED2}),
                html.Span(f"{s.win_rate:.0f}%", style={"fontSize": "10px", "color": wr_col, "fontWeight": "700"}),
                html.Span(f"{s.min_score:.0f}", style={"fontSize": "10px", "color": YELLOW}),
                html.Span(f"{s.lot_multiplier:.2f}×", style={"fontSize": "10px", "color": ACCENT2}),
                html.Span(f"${s.avg_pnl:+.4f}", style={"fontSize": "9px", "color": wr_col,
                           "fontFamily": "'JetBrains Mono',monospace"}),
            ],
        ))
    return html.Div(rows)


# ── Sentiment panel ───────────────────────────────────────────────────────────

@app.callback(
    Output("sentiment-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_sentiment(n: int) -> html.Div:
    try:
        pairs   = get_pairs()
        scores  = {p: sentiment_cache.get_score(p) for p in pairs}
    except Exception:
        return html.Div("Sentiment unavailable", style={"color": MUTED, "fontSize": "11px"})

    if not any(v is not None for v in scores.values()):
        return html.Div("Sentiment loading (next refresh in ~2h)", style={"color": MUTED, "fontSize": "11px"})

    items = []
    for pair, score in scores.items():
        if score is None:
            continue
        col  = GREEN if score > 0 else (RED if score < 0 else MUTED)
        lbl  = "BULLISH" if score > 0.2 else ("BEARISH" if score < -0.2 else "NEUTRAL")
        items.append(html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                   "padding": "4px 0", "borderBottom": "1px solid rgba(255,255,255,.03)"},
            children=[
                html.Span(pair.rstrip("m"), style={"fontSize": "11px", "fontWeight": "600", "color": TEXT}),
                html.Span(f"{score:+.2f}", style={"fontSize": "10px", "color": col,
                           "fontWeight": "700", "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(lbl, style={"fontSize": "9px", "color": col, "fontWeight": "600"}),
            ],
        ))
    return html.Div(items) if items else html.Div("No sentiment data", style={"color": MUTED, "fontSize": "11px"})


# ── Hypothesis queue panel ────────────────────────────────────────────────────

@app.callback(
    Output("hypothesis-queue-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_hypothesis_queue(n: int) -> html.Div:
    try:
        from core.hypothesis_queue import get_pending, stats as hq_stats
        pending = get_pending()
        s = hq_stats()
    except Exception:
        return html.Div("No hypothesis data", style={"color": MUTED, "fontSize": "11px"})

    summary = html.Div(
        style={"fontSize": "9px", "color": MUTED2, "marginBottom": "8px"},
        children=f"Total: {s['total']} | Pending: {s['pending']} | Accepted: {s['accepted']} | Rejected: {s['rejected']}",
    )

    if not pending:
        return html.Div([summary, html.Div("Queue is empty — no pending hypotheses", style={"color": MUTED, "fontSize": "11px"})])

    rows = [summary]
    for h in pending[:8]:
        rows.append(html.Div(
            style={"padding": "5px 0", "borderBottom": "1px solid rgba(255,255,255,.03)"},
            children=[
                html.Div(style={"display": "flex", "gap": "8px", "alignItems": "baseline"}, children=[
                    html.Span(f"#{h['id']}", style={"fontSize": "9px", "color": ACCENT, "fontFamily": "'JetBrains Mono',monospace"}),
                    html.Span(h["pair"].rstrip("m"), style={"fontSize": "10px", "fontWeight": "600", "color": TEXT}),
                    html.Span(f"P{h['priority']}", style={"fontSize": "9px", "color": YELLOW}),
                    html.Span(h["source"], style={"fontSize": "9px", "color": MUTED2}),
                ]),
                html.Div(h["title"][:70], style={"fontSize": "10px", "color": MUTED, "marginTop": "2px"}),
            ],
        ))
    return html.Div(rows)


# ── Mistake detector panel ────────────────────────────────────────────────────

@app.callback(
    Output("mistake-detector-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_mistake_detector(n: int) -> html.Div:
    try:
        from core.mistake_detector import detect_mistakes
        mistakes = detect_mistakes(recent_n=50)
    except Exception:
        return html.Div("Mistake detector unavailable", style={"color": MUTED, "fontSize": "11px"})

    if not mistakes:
        return html.Div(
            "No systematic mistakes detected — need 10+ trades to analyse",
            style={"color": GREEN, "fontSize": "11px"},
        )

    rows = []
    for m in mistakes:
        sev     = m.get("severity", "medium")
        sev_col = RED if sev == "high" else YELLOW
        sev_lbl = "HIGH" if sev == "high" else "MED"
        rows.append(html.Div(
            style={"display": "flex", "gap": "8px", "alignItems": "flex-start",
                   "padding": "6px 0", "borderBottom": "1px solid rgba(255,255,255,.03)"},
            children=[
                html.Span(
                    sev_lbl,
                    style={"fontSize": "8px", "fontWeight": "700", "color": sev_col,
                           "background": f"rgba{sev_col[1:]},0.1)" if len(sev_col) == 7 else "transparent",
                           "border": f"1px solid {sev_col}44",
                           "borderRadius": "3px", "padding": "1px 5px",
                           "flexShrink": "0", "marginTop": "1px"},
                ),
                html.Div([
                    html.Div(m.get("message", ""), style={"fontSize": "10px", "color": TEXT, "lineHeight": "1.4"}),
                    html.Div(m.get("stat", ""),    style={"fontSize": "9px",  "color": MUTED2, "marginTop": "2px",
                                                           "fontFamily": "'JetBrains Mono',monospace"}),
                ]),
            ],
        ))

    high_count = sum(1 for m in mistakes if m.get("severity") == "high")
    summary = html.Div(
        f"{len(mistakes)} issues found  · {high_count} high severity",
        style={"fontSize": "9px", "color": RED if high_count else YELLOW,
               "marginBottom": "8px", "fontWeight": "600"},
    )
    return html.Div([summary] + rows)


# ── Performance attribution panel ─────────────────────────────────────────────

@app.callback(
    Output("perf-attribution-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_perf_attribution(n: int) -> html.Div:
    try:
        from core.performance_attribution import compute_attribution
        attrs = compute_attribution(min_samples=30)
    except Exception:
        return html.Div("Attribution unavailable", style={"color": MUTED, "fontSize": "11px"})

    if not attrs:
        return html.Div(
            "Need 30+ closed trades with ML samples to compute attribution",
            style={"color": MUTED, "fontSize": "11px"},
        )

    header = html.Div(
        style={"display": "grid",
               "gridTemplateColumns": "120px 60px 60px 55px 80px 55px 55px",
               "gap": "4px", "padding": "4px 0",
               "borderBottom": "1px solid rgba(255,255,255,.06)", "marginBottom": "4px"},
        children=[
            html.Span(c, style={"fontSize": "8px", "color": MUTED, "fontWeight": "600",
                                "textTransform": "uppercase"})
            for c in ["FEATURE", "HIGH WR", "LOW WR", "DELTA", "VERDICT", "N HIGH", "N LOW"]
        ],
    )
    rows = [header]
    for a in attrs[:15]:
        verdict = a.get("verdict", "noise")
        v_col   = GREEN if verdict == "predictive" else (RED if verdict == "inverse" else MUTED2)
        d_col   = GREEN if a["delta_pp"] > 0 else (RED if a["delta_pp"] < 0 else MUTED2)
        rows.append(html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "120px 60px 60px 55px 80px 55px 55px",
                   "gap": "4px", "padding": "3px 0",
                   "borderBottom": "1px solid rgba(255,255,255,.03)",
                   "alignItems": "center"},
            children=[
                html.Span(a["feature"][:18],
                          style={"fontSize": "9px", "color": TEXT, "fontWeight": "600",
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(f"{a['high_wr']:.0f}%",
                          style={"fontSize": "9px", "color": GREEN if a["high_wr"] >= 50 else RED}),
                html.Span(f"{a['low_wr']:.0f}%",
                          style={"fontSize": "9px", "color": GREEN if a["low_wr"] >= 50 else RED}),
                html.Span(f"{a['delta_pp']:+.0f}pp",
                          style={"fontSize": "9px", "color": d_col, "fontWeight": "700",
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(verdict.upper(),
                          style={"fontSize": "8px", "color": v_col, "fontWeight": "600"}),
                html.Span(str(a["n_high"]),
                          style={"fontSize": "9px", "color": MUTED2}),
                html.Span(str(a["n_low"]),
                          style={"fontSize": "9px", "color": MUTED2}),
            ],
        ))

    predictive_n = sum(1 for a in attrs if a["verdict"] == "predictive")
    inverse_n    = sum(1 for a in attrs if a["verdict"] == "inverse")
    summary = html.Div(
        f"{len(attrs)} features analysed  · {predictive_n} predictive  · {inverse_n} inverse  · {len(attrs)-predictive_n-inverse_n} noise",
        style={"fontSize": "9px", "color": MUTED2, "marginBottom": "6px"},
    )
    return html.Div([summary] + rows)


# ── Autonomous pipeline panel ─────────────────────────────────────────────────

_pipeline_thread_running = False
_pipeline_thread_lock    = threading.Lock()


def _run_pipeline_thread() -> None:
    global _pipeline_thread_running
    try:
        from core.autonomous_pipeline import run_pipeline_step
        from core.pipeline_log import record as log_pipeline
        result = run_pipeline_step()
        log_pipeline(result)
    except Exception:
        pass
    finally:
        with _pipeline_thread_lock:
            _pipeline_thread_running = False


@app.callback(
    Output("pipeline-panel",    "children"),
    Output("pipeline-run-btn",  "disabled"),
    Output("pipeline-run-btn",  "children"),
    Input("interval",           "n_intervals"),
    Input("pipeline-run-btn",   "n_clicks"),
    prevent_initial_call=False,
)
def update_pipeline_panel(n: int, run_clicks: int):
    global _pipeline_thread_running

    # Manual trigger
    if ctx.triggered_id == "pipeline-run-btn" and run_clicks:
        with _pipeline_thread_lock:
            if not _pipeline_thread_running:
                _pipeline_thread_running = True
                threading.Thread(target=_run_pipeline_thread, daemon=True).start()

    btn_disabled = _pipeline_thread_running
    btn_label    = "⏳ RUNNING…" if _pipeline_thread_running else "▶ RUN NOW"

    try:
        from core.pipeline_log import get_recent
        entries = get_recent(15)
    except Exception:
        entries = []

    if not entries:
        no_data = html.Div(
            "No pipeline runs yet — runs every 6 hours automatically, or click RUN NOW",
            style={"color": MUTED, "fontSize": "11px"},
        )
        return no_data, btn_disabled, btn_label

    header = html.Div(
        style={"display": "grid",
               "gridTemplateColumns": "130px 80px 55px 1fr",
               "gap": "5px", "padding": "4px 0",
               "borderBottom": "1px solid rgba(255,255,255,.06)", "marginBottom": "4px"},
        children=[
            html.Span(c, style={"fontSize": "8px", "color": MUTED, "fontWeight": "600",
                                "textTransform": "uppercase"})
            for c in ["TIME", "HYPOTHESIS", "RESULT", "REASON"]
        ],
    )
    rows = [header]
    for e in entries:
        ts_raw   = e.get("ts", "")
        try:
            ts_dt = datetime.fromisoformat(ts_raw)
            ts_str = ts_dt.strftime("%m-%d %H:%M UTC")
        except Exception:
            ts_str = ts_raw[:16]

        hyp      = e.get("hypothesis_processed") or "—"
        approved = e.get("approved", False)
        reason   = e.get("reason", "—")
        res_col  = GREEN if approved else (YELLOW if hyp == "—" else RED)
        res_txt  = "APPROVED" if approved else ("IDLE" if hyp == "—" else "REJECTED")

        rows.append(html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "130px 80px 55px 1fr",
                   "gap": "5px", "padding": "3px 0",
                   "borderBottom": "1px solid rgba(255,255,255,.03)",
                   "alignItems": "center"},
            children=[
                html.Span(ts_str,
                          style={"fontSize": "9px", "color": MUTED2,
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(hyp[:10],
                          style={"fontSize": "9px", "color": ACCENT,
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(res_txt,
                          style={"fontSize": "8px", "color": res_col, "fontWeight": "600"}),
                html.Span(reason[:60],
                          style={"fontSize": "9px", "color": MUTED2}),
            ],
        ))

    approved_count = sum(1 for e in entries if e.get("approved"))
    summary = html.Div(
        f"{len(entries)} runs shown  · {approved_count} approved  "
        f"· {len(entries) - approved_count} rejected/idle",
        style={"fontSize": "9px", "color": MUTED2, "marginBottom": "6px"},
    )
    return html.Div([summary] + rows), btn_disabled, btn_label


# ── A/B Tests panel ───────────────────────────────────────────────────────────

@app.callback(
    Output("ab-test-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_ab_tests(n: int) -> html.Div:
    try:
        from core.ab_testing import get_all_tests
        tests = get_all_tests()
    except Exception:
        return html.Div("A/B test data unavailable", style={"color": MUTED, "fontSize": "11px"})

    if not tests:
        return html.Div("No A/B tests running — create one via the hypothesis pipeline",
                        style={"color": MUTED, "fontSize": "11px"})

    # Sort: running first, then by start_time descending
    tests = sorted(tests, key=lambda t: (t.get("status") != "running", t.get("start_time", "")), reverse=False)

    header = html.Div(
        style={"display": "grid",
               "gridTemplateColumns": "60px 90px 90px 70px 70px 65px 65px 60px",
               "gap": "4px", "padding": "4px 0",
               "borderBottom": "1px solid rgba(255,255,255,.06)", "marginBottom": "4px"},
        children=[
            html.Span(c, style={"fontSize": "8px", "color": MUTED, "fontWeight": "600",
                                "textTransform": "uppercase"})
            for c in ["ID", "CONTROL (A)", "CHALLENGER (B)", "PAIR",
                      "A WR / PnL", "B WR / PnL", "STATUS", "WINNER"]
        ],
    )
    rows = [header]
    for t in tests[:10]:
        status     = t.get("status", "—")
        winner     = t.get("winner") or "—"
        sta_col    = GREEN if status == "complete" else YELLOW
        win_col    = ACCENT if winner != "—" else MUTED2

        n_a = t.get("trades_a", 0) or 1
        n_b = t.get("trades_b", 0) or 1
        wr_a = t.get("wins_a", 0) / n_a * 100
        wr_b = t.get("wins_b", 0) / n_b * 100
        pnl_a = t.get("pnl_a", 0.0)
        pnl_b = t.get("pnl_b", 0.0)

        rows.append(html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "60px 90px 90px 70px 70px 65px 65px 60px",
                   "gap": "4px", "padding": "3px 0",
                   "borderBottom": "1px solid rgba(255,255,255,.03)",
                   "alignItems": "center"},
            children=[
                html.Span(t.get("test_id", "—"),
                          style={"fontSize": "9px", "color": ACCENT,
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(t.get("strategy_a", "—"),
                          style={"fontSize": "9px", "color": TEXT, "fontWeight": "600"}),
                html.Span(t.get("strategy_b", "—"),
                          style={"fontSize": "9px", "color": PURPLE, "fontWeight": "600"}),
                html.Span(t.get("pair", "—").rstrip("m"),
                          style={"fontSize": "9px", "color": MUTED2}),
                html.Span(f"{wr_a:.0f}% ${pnl_a:+.2f}",
                          style={"fontSize": "9px", "color": GREEN if pnl_a >= 0 else RED,
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(f"{wr_b:.0f}% ${pnl_b:+.2f}",
                          style={"fontSize": "9px", "color": GREEN if pnl_b >= 0 else RED,
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(status.upper(),
                          style={"fontSize": "8px", "color": sta_col, "fontWeight": "600"}),
                html.Span(winner[:12] if winner != "—" else "—",
                          style={"fontSize": "8px", "color": win_col, "fontWeight": "600"}),
            ],
        ))

    summary = html.Div(
        f"{sum(1 for t in tests if t.get('status') == 'running')} running  "
        f"· {sum(1 for t in tests if t.get('status') == 'complete')} complete  "
        f"· {sum(1 for t in tests if t.get('winner'))} decided",
        style={"fontSize": "9px", "color": MUTED2, "marginBottom": "6px"},
    )
    return html.Div([summary] + rows)


# ── Paper trading panel ───────────────────────────────────────────────────────

@app.callback(
    Output("paper-trading-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_paper_trading(n: int) -> html.Div:
    try:
        from core.paper_trader import get_open_trades, get_all_trades, paper_performance
        open_trades = get_open_trades()
        all_trades  = get_all_trades()
    except Exception:
        return html.Div("Paper trading data unavailable", style={"color": MUTED, "fontSize": "11px"})

    if not all_trades:
        return html.Div("No paper trades yet — auto-approved hypotheses enter paper mode first",
                        style={"color": MUTED, "fontSize": "11px"})

    # Per-strategy performance summary
    strategies = list({t.get("strategy", "SMC_TREND") for t in all_trades})
    perf_rows = []
    for strat in sorted(strategies):
        try:
            p = paper_performance(strat, since_days=14)
        except Exception:
            continue
        if p["trades"] == 0:
            continue
        sh_col  = GREEN if p["sharpe"] > 0 else RED
        pnl_col = GREEN if p["pnl"] >= 0 else RED
        promo_ready = p["trades"] >= 10 and p["pnl"] > 0 and p["sharpe"] > 0
        perf_rows.append(html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "100px 50px 60px 75px 65px 60px",
                   "gap": "4px", "padding": "3px 0",
                   "borderBottom": "1px solid rgba(255,255,255,.03)",
                   "alignItems": "center"},
            children=[
                html.Span(strat.replace("_", " "),
                          style={"fontSize": "9px", "fontWeight": "600", "color": TEXT}),
                html.Span(str(p["trades"]),
                          style={"fontSize": "9px", "color": MUTED2}),
                html.Span(f"{p['win_rate']:.0f}%",
                          style={"fontSize": "9px", "color": GREEN if p["win_rate"] >= 50 else RED,
                                 "fontWeight": "700"}),
                html.Span(f"${p['pnl']:+.2f}",
                          style={"fontSize": "9px", "color": pnl_col, "fontWeight": "700",
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(f"{p['sharpe']:+.3f}",
                          style={"fontSize": "9px", "color": sh_col, "fontWeight": "700",
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span("PROMOTE?" if promo_ready else "WATCHING",
                          style={"fontSize": "8px", "fontWeight": "600",
                                 "color": GREEN if promo_ready else YELLOW}),
            ],
        ))

    perf_header = html.Div(
        style={"display": "grid",
               "gridTemplateColumns": "100px 50px 60px 75px 65px 60px",
               "gap": "4px", "padding": "4px 0",
               "borderBottom": "1px solid rgba(255,255,255,.06)", "marginBottom": "4px"},
        children=[
            html.Span(c, style={"fontSize": "8px", "color": MUTED, "fontWeight": "600",
                                "textTransform": "uppercase"})
            for c in ["STRATEGY", "TRADES", "WIN %", "14d PnL", "SHARPE", "STATUS"]
        ],
    )

    # Open paper positions
    open_section = []
    if open_trades:
        open_section = [
            html.Div("OPEN PAPER POSITIONS",
                     style={"fontSize": "8px", "color": MUTED2, "letterSpacing": "1px",
                            "textTransform": "uppercase", "marginTop": "10px", "marginBottom": "4px"}),
        ]
        for t in open_trades[:5]:
            d_col = GREEN if t.get("direction") == "long" else RED
            d_sym = "▲" if t.get("direction") == "long" else "▼"
            open_section.append(html.Div(
                style={"display": "flex", "gap": "10px", "alignItems": "center",
                       "padding": "3px 0", "borderBottom": "1px solid rgba(255,255,255,.03)",
                       "fontSize": "10px"},
                children=[
                    html.Span(f"#{t.get('id','')}", style={"color": ACCENT, "fontFamily": "'JetBrains Mono',monospace", "fontSize": "9px"}),
                    html.Span(t.get("pair","").rstrip("m"), style={"fontWeight": "600", "color": TEXT}),
                    html.Span(f"{d_sym} {t.get('direction','').upper()}", style={"color": d_col, "fontWeight": "700"}),
                    html.Span(f"E:{t.get('entry',0):.5f}", style={"color": MUTED2, "fontFamily": "'JetBrains Mono',monospace", "fontSize": "9px"}),
                    html.Span(t.get("strategy","").replace("_"," "), style={"color": PURPLE, "fontSize": "9px"}),
                ],
            ))

    summary = html.Div(
        f"{len(open_trades)} open  · {len(all_trades) - len(open_trades)} closed  · {len(strategies)} strategies",
        style={"fontSize": "9px", "color": MUTED2, "marginBottom": "8px"},
    )

    return html.Div([summary, perf_header] + perf_rows + open_section)


# ── Equity curve chart ────────────────────────────────────────────────────────

@app.callback(
    Output("equity-chart", "figure"),
    Input("interval", "n_intervals"),
)
def update_equity_chart(n: int) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
        margin=dict(l=50, r=10, t=8, b=30),
        xaxis=dict(color=MUTED, gridcolor="rgba(255,255,255,.04)", zeroline=False),
        yaxis=dict(color=MUTED, gridcolor="rgba(255,255,255,.04)", zeroline=False, side="right"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PANEL2, font_color=TEXT, font_size=11),
    )

    try:
        from db.session import get_session
        from db.models import DayLog
        with get_session() as db:
            logs = db.query(DayLog).order_by(DayLog.date).all()

        if not logs:
            fig.add_annotation(text="No trade history yet", xref="paper", yref="paper",
                               x=0.5, y=0.5, font=dict(color=MUTED, size=12), showarrow=False)
            return fig

        dates   = [l.date for l in logs]
        balance = [l.ending_balance if l.ending_balance else l.starting_balance for l in logs]
        pnl_cum = [sum(l.realized_pnl for l in logs[:i+1]) for i in range(len(logs))]

        fig.add_trace(go.Scatter(
            x=dates, y=balance,
            mode="lines+markers",
            name="Balance",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor=f"rgba(124,58,237,0.05)",
        ))
        fig.add_trace(go.Bar(
            x=dates, y=[l.realized_pnl for l in logs],
            name="Daily P&L",
            marker_color=[GREEN if l.realized_pnl >= 0 else RED for l in logs],
            opacity=0.6,
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="left", color=MUTED,
                        gridcolor="rgba(0,0,0,0)", zeroline=True,
                        zerolinecolor="rgba(255,255,255,.1)"),
        )
    except Exception as e:
        fig.add_annotation(text=f"Equity data error: {e}", xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(color=MUTED, size=11), showarrow=False)
    return fig


# ── Correlation panel ─────────────────────────────────────────────────────────

@app.callback(
    Output("correlation-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_correlation(n: int) -> html.Div:
    positions = feed.get_positions()
    if not positions or len(positions) < 2:
        return html.Div("Open 2+ positions to see correlation", style={"color": MUTED, "fontSize": "11px"})

    pairs = [p["pair"] for p in positions]
    rows  = []
    for i, p1 in enumerate(pairs):
        for j, p2 in enumerate(pairs):
            if j <= i:
                continue
            # Simple heuristic: shared currency = positive corr, opposing = negative
            p1u = p1.rstrip("m").upper()
            p2u = p2.rstrip("m").upper()
            shared = len(set(p1u[:3] + p1u[3:6]) & set(p2u[:3] + p2u[3:6])) > 0
            corr   = 0.7 if shared else -0.3
            col    = RED if abs(corr) > 0.5 else (YELLOW if abs(corr) > 0.3 else GREEN)
            rows.append(html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "padding": "3px 0", "borderBottom": "1px solid rgba(255,255,255,.03)"},
                children=[
                    html.Span(f"{p1u} ↔ {p2u}", style={"fontSize": "10px", "color": TEXT}),
                    html.Span(f"{corr:+.1f}", style={"fontSize": "10px", "fontWeight": "700",
                               "color": col, "fontFamily": "'JetBrains Mono',monospace"}),
                    html.Span("HIGH CORR" if abs(corr) > 0.5 else "OK",
                              style={"fontSize": "9px", "color": col}),
                ],
            ))

    return html.Div(rows) if rows else html.Div("No correlations to show", style={"color": MUTED, "fontSize": "11px"})


# ── WFO callbacks ─────────────────────────────────────────────────────────────

_wfo_result: dict = {}
_wfo_lock   = threading.Lock()
_mc_result:  dict = {}
_mc_lock    = threading.Lock()


def _run_wfo_thread(pair, total, is_d, oos_d, trials, anchored) -> None:
    global _wfo_result
    with _wfo_lock:
        _wfo_result = {"status": "running", "text": f"Running WFO {pair} …", "windows": []}
    try:
        from backtest.wfo import WalkForwardOptimizer
        opt = WalkForwardOptimizer(
            pair=pair, total_days=total, is_days=is_d, oos_days=oos_d,
            anchored=anchored, n_trials=trials,
        )
        summary = opt.run()
        text = (
            f"Pair     : {summary.pair}\n"
            f"Windows  : {len(summary.windows)}\n"
            f"Mean OOS PF: {summary.mean_oos_pf:.2f}\n"
            f"Mean OOS WR: {summary.mean_oos_wr:.1f}%\n"
            f"Mean OOS DD: {summary.mean_oos_dd:.1f}%\n"
            f"Stability  : {summary.stability_score:.2f}\n"
            f"Rec params : score={summary.recommended_params[0]:.0f} risk={summary.recommended_params[1]}%\n"
            f"\n{summary._oos_verdict()}"
        )
        windows = [
            {"window": w.window_idx, "pf": round(w.oos_result.profit_factor, 2),
             "wr": round(w.oos_result.win_rate, 1), "trades": w.oos_result.total_trades}
            for w in summary.windows
        ]
        with _wfo_lock:
            _wfo_result = {"status": "done", "text": text, "windows": windows}
        summary.save_to_obsidian()
    except Exception as e:
        with _wfo_lock:
            _wfo_result = {"status": "error", "text": f"Error: {e}", "windows": []}


@app.callback(
    Output("wfo-anchored-store", "data"),
    Output("wfo-anchored-btn",   "className"),
    Input("wfo-anchored-btn",    "n_clicks"),
    State("wfo-anchored-store",  "data"),
    prevent_initial_call=True,
)
def toggle_wfo_anchored(n, anchored):
    new_val = not anchored
    cls = "tf-btn active" if new_val else "tf-btn"
    return new_val, cls


@app.callback(
    Output("wfo-result",         "children"),
    Output("wfo-run-btn",        "disabled"),
    Output("wfo-run-btn",        "children"),
    Output("wfo-stability-chart","figure"),
    Input("wfo-run-btn",  "n_clicks"),
    Input("wfo-poll",     "n_intervals"),
    State("wfo-pair-select",  "value"),
    State("wfo-total-days",   "value"),
    State("wfo-is-days",      "value"),
    State("wfo-oos-days",     "value"),
    State("wfo-trials",       "value"),
    State("wfo-anchored-store","data"),
    prevent_initial_call=True,
)
def handle_wfo(n_clicks, poll, pair, total, is_d, oos_d, trials, anchored):
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
                             margin=dict(l=40, r=10, t=8, b=30))

    if ctx.triggered_id == "wfo-run-btn":
        triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
        if not triggered_value:
            raise dash.exceptions.PreventUpdate
        total = int(total or 365); is_d = int(is_d or 90)
        oos_d = int(oos_d or 30);  trials = int(trials or 20)
        t = threading.Thread(target=_run_wfo_thread,
                             args=(pair, total, is_d, oos_d, trials, bool(anchored)), daemon=True)
        t.start()
        return "Starting WFO …", True, "⏳ RUNNING…", empty_fig

    with _wfo_lock:
        state = dict(_wfo_result)

    if not state:
        raise dash.exceptions.PreventUpdate

    status  = state.get("status", "")
    text    = state.get("text",   "")
    windows = state.get("windows", [])

    if status == "running":
        return text, True, "⏳ RUNNING…", empty_fig

    # Build stability chart
    if windows:
        xs  = [w["window"] for w in windows]
        pfs = [w["pf"]     for w in windows]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=pfs, mode="lines+markers",
                                 line=dict(color=ACCENT, width=2), marker=dict(size=6)))
        fig.add_hline(y=1.2, line=dict(color=RED, dash="dot", width=1))
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
                          margin=dict(l=40, r=10, t=8, b=30),
                          xaxis=dict(title="Window", color=MUTED, gridcolor="rgba(255,255,255,.04)"),
                          yaxis=dict(title="OOS PF", color=MUTED, gridcolor="rgba(255,255,255,.04)", side="right"))
    else:
        fig = empty_fig

    if status in ("done", "error"):
        return text, False, "▶  RUN WFO", fig

    raise dash.exceptions.PreventUpdate


# ── Monte Carlo callbacks ─────────────────────────────────────────────────────

def _run_mc_thread(pair, sims, days) -> None:
    global _mc_result
    with _mc_lock:
        _mc_result = {"status": "running", "text": f"Running MC {pair} …", "bands": {}}
    try:
        from backtest.montecarlo import MonteCarlo
        from backtest.engine import BacktestEngine
        engine = BacktestEngine(pair=pair, days=days)
        bt_res = engine.run()
        if bt_res.total_trades < 5:
            with _mc_lock:
                _mc_result = {"status": "error", "text": f"Not enough trades ({bt_res.total_trades})", "bands": {}}
            return
        mc = MonteCarlo(bt_res.trades, n_simulations=sims)
        res = mc.run()
        text = (
            f"Pair          : {pair}\n"
            f"Simulations   : {sims}\n"
            f"Ruin prob     : {res.ruin_probability:.1f}%\n"
            f"Exp max DD    : {res.expected_max_drawdown:.1f}%\n"
            f"P95 max DD    : {res.p95_drawdown:.1f}%\n"
            f"\n{res.verdict()}"
        )
        with _mc_lock:
            _mc_result = {"status": "done", "text": text,
                          "bands": {"p05": res.p05_curve, "p50": res.p50_curve, "p95": res.p95_curve}}
        res.save_to_obsidian(pair, days)
    except Exception as e:
        with _mc_lock:
            _mc_result = {"status": "error", "text": f"Error: {e}", "bands": {}}


@app.callback(
    Output("mc-result",      "children"),
    Output("mc-run-btn",     "disabled"),
    Output("mc-run-btn",     "children"),
    Output("mc-bands-chart", "figure"),
    Input("mc-run-btn",   "n_clicks"),
    Input("mc-poll",      "n_intervals"),
    State("mc-pair-select",  "value"),
    State("mc-sims-input",   "value"),
    State("mc-days-input",   "value"),
    prevent_initial_call=True,
)
def handle_mc(n_clicks, poll, pair, sims, days):
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
                             margin=dict(l=40, r=10, t=8, b=30))

    if ctx.triggered_id == "mc-run-btn":
        triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
        if not triggered_value:
            raise dash.exceptions.PreventUpdate
        sims = int(sims or 500); days = int(days or 90)
        t = threading.Thread(target=_run_mc_thread, args=(pair, sims, days), daemon=True)
        t.start()
        return "Starting Monte Carlo …", True, "⏳ RUNNING…", empty_fig

    with _mc_lock:
        state = dict(_mc_result)

    if not state:
        raise dash.exceptions.PreventUpdate

    status = state.get("status", "")
    text   = state.get("text",   "")
    bands  = state.get("bands",  {})

    if status == "running":
        return text, True, "⏳ RUNNING…", empty_fig

    if bands:
        xs = list(range(len(bands.get("p50", []))))
        fig = go.Figure()
        if bands.get("p95"):
            fig.add_trace(go.Scatter(x=xs, y=bands["p95"], fill=None,
                                     line=dict(color="rgba(124,58,237,.3)", width=1), name="P95"))
        if bands.get("p05"):
            fig.add_trace(go.Scatter(x=xs, y=bands["p05"], fill="tonexty",
                                     fillcolor="rgba(124,58,237,.08)",
                                     line=dict(color="rgba(124,58,237,.3)", width=1), name="P05"))
        if bands.get("p50"):
            fig.add_trace(go.Scatter(x=xs, y=bands["p50"],
                                     line=dict(color=ACCENT, width=2), name="P50 Median"))
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
                          margin=dict(l=40, r=10, t=8, b=30),
                          xaxis=dict(title="Trade #", color=MUTED, gridcolor="rgba(255,255,255,.04)"),
                          yaxis=dict(title="Equity $", color=MUTED, gridcolor="rgba(255,255,255,.04)", side="right"))
    else:
        fig = empty_fig

    if status in ("done", "error"):
        return text, False, "▶  RUN MONTE CARLO", fig

    raise dash.exceptions.PreventUpdate


# ── Risk page callbacks ───────────────────────────────────────────────────────

@app.callback(
    Output("risk-gate-panel",     "children"),
    Output("risk-daily-label",    "children"),
    Output("risk-daily-bar",      "style"),
    Output("risk-weekly-label",   "children"),
    Output("risk-weekly-bar",     "style"),
    Output("risk-monthly-label",  "children"),
    Output("risk-monthly-bar",    "style"),
    Output("risk-leverage-label", "children"),
    Output("risk-leverage-bar",   "style"),
    Output("risk-exposure-panel", "children"),
    Output("risk-streak-panel",   "children"),
    Input("interval", "n_intervals"),
)
def update_risk_page(n: int):
    if _capital is None:
        empty = html.Div("Capital manager unavailable", style={"color": MUTED, "fontSize": "11px"})
        bar0 = {"height": "100%", "width": "0%", "background": MUTED2, "borderRadius": "3px"}
        return empty, "—", bar0, "—", bar0, "—", bar0, "—", bar0, empty, empty

    cap = _capital.status_dict

    # ── Gate status ──────────────────────────────────────────────────────
    halted    = cap.get("halted", False)
    halt_reason = cap.get("halt_reason", "")
    can_trade  = cap.get("can_trade", True)
    cooldown   = cap.get("cooldown_active", False)
    cooldown_until = cap.get("cooldown_until")

    if halted:
        gate_color, gate_label, gate_bg = RED, "HALTED", "rgba(239,68,68,.1)"
    elif cooldown:
        gate_color, gate_label, gate_bg = YELLOW, "COOLDOWN", "rgba(245,158,11,.1)"
    else:
        gate_color, gate_label, gate_bg = GREEN, "ACTIVE", "rgba(16,185,129,.1)"

    gate_detail = halt_reason or (f"Cooldown until {cooldown_until}" if cooldown else "All systems go")
    gate_panel = html.Div(style={"padding": "10px", "display": "flex", "alignItems": "center", "gap": "16px"}, children=[
        html.Div(style={"background": gate_bg, "border": f"1px solid {gate_color}",
                        "borderRadius": "6px", "padding": "6px 14px"}, children=[
            html.Span(gate_label, style={"color": gate_color, "fontSize": "12px", "fontWeight": "700", "letterSpacing": "2px"}),
        ]),
        html.Div(children=[
            html.Div(gate_detail, style={"fontSize": "11px", "color": TEXT}),
            html.Div(f"Balance: ${cap.get('balance', 0):,.2f}  |  Equity: ${cap.get('equity', 0):,.2f}",
                     style={"fontSize": "10px", "color": MUTED, "marginTop": "2px",
                            "fontFamily": "'JetBrains Mono',monospace"}),
        ]),
    ])

    # ── Daily PnL gauge ──────────────────────────────────────────────────
    day_pnl     = cap.get("realized_pnl", 0.0)
    max_loss    = settings.max_loss_amount
    day_target  = settings.daily_target_amount

    if day_pnl < 0:
        daily_pct  = min(abs(day_pnl) / max_loss * 100, 100) if max_loss > 0 else 0
        daily_col  = RED if daily_pct >= 80 else YELLOW if daily_pct >= 50 else GREEN
        daily_lbl  = f"${day_pnl:+.2f} / −${max_loss:.2f} ({daily_pct:.0f}% of limit)"
    else:
        daily_pct  = min(day_pnl / day_target * 100, 100) if day_target > 0 else 0
        daily_col  = GREEN
        daily_lbl  = f"${day_pnl:+.2f} / +${day_target:.2f} ({daily_pct:.0f}% of target)"
    daily_bar = {"height": "100%", "width": f"{daily_pct:.0f}%", "background": daily_col, "borderRadius": "3px", "transition": "width .5s"}

    # ── Weekly drawdown gauge ────────────────────────────────────────────
    weekly_dd   = max(cap.get("weekly_dd_pct", 0.0), 0.0)
    weekly_lim  = cap.get("weekly_dd_limit", 6.0)
    weekly_pct  = min(weekly_dd / weekly_lim * 100, 100) if weekly_lim > 0 else 0
    weekly_col  = RED if weekly_pct >= 80 else YELLOW if weekly_pct >= 50 else GREEN
    weekly_lbl  = f"−{weekly_dd:.2f}% / −{weekly_lim:.0f}% ({weekly_pct:.0f}% of limit)"
    weekly_bar  = {"height": "100%", "width": f"{weekly_pct:.0f}%", "background": weekly_col, "borderRadius": "3px", "transition": "width .5s"}

    # ── Monthly drawdown gauge ───────────────────────────────────────────
    monthly_dd  = max(cap.get("monthly_dd_pct", 0.0), 0.0)
    monthly_lim = cap.get("monthly_dd_limit", 10.0)
    monthly_pct = min(monthly_dd / monthly_lim * 100, 100) if monthly_lim > 0 else 0
    monthly_col = RED if monthly_pct >= 80 else YELLOW if monthly_pct >= 50 else GREEN
    monthly_lbl = f"−{monthly_dd:.2f}% / −{monthly_lim:.0f}% ({monthly_pct:.0f}% of limit)"
    monthly_bar = {"height": "100%", "width": f"{monthly_pct:.0f}%", "background": monthly_col, "borderRadius": "3px", "transition": "width .5s"}

    # ── Leverage gauge ───────────────────────────────────────────────────
    lev_cur     = cap.get("current_leverage", 0.0)
    lev_max     = cap.get("max_leverage", 500.0)
    lev_pct     = min(lev_cur / lev_max * 100, 100) if lev_max > 0 else 0
    lev_col     = RED if lev_pct >= 80 else YELLOW if lev_pct >= 50 else ACCENT2
    lev_lbl     = f"{lev_cur:.1f}× / {lev_max:.0f}× ({lev_pct:.0f}% of max)"
    lev_bar     = {"height": "100%", "width": f"{lev_pct:.0f}%", "background": lev_col, "borderRadius": "3px", "transition": "width .5s"}

    # ── Open exposure table ──────────────────────────────────────────────
    open_pos = cap.get("open_positions", [])
    if not open_pos:
        exposure_panel = html.Div("No open positions", style={"fontSize": "11px", "color": MUTED, "padding": "6px 0"})
    else:
        col_w = "80px 70px 55px 70px 80px"
        hdr = html.Div(style={"display": "grid", "gridTemplateColumns": col_w, "gap": "5px",
                               "padding": "4px 0", "borderBottom": f"1px solid {BORDER}", "marginBottom": "4px"},
                       children=[html.Span(c, style={"fontSize": "8px", "color": MUTED2, "fontWeight": "600", "textTransform": "uppercase"})
                                  for c in ["PAIR", "DIR", "LOTS", "ENTRY", "UNREAL PNL"]])
        rows = [hdr]
        for pos in open_pos:
            upnl = pos.get("pnl", 0.0)
            upnl_col = GREEN if upnl >= 0 else RED
            rows.append(html.Div(
                style={"display": "grid", "gridTemplateColumns": col_w, "gap": "5px",
                       "padding": "3px 0", "borderBottom": f"1px solid rgba(255,255,255,.02)"},
                children=[
                    html.Span(pos.get("pair", "—").rstrip("m"), style={"fontSize": "10px", "fontWeight": "600", "color": TEXT}),
                    html.Span(pos.get("direction", "—").upper(), style={"fontSize": "9px",
                              "color": GREEN if pos.get("direction") == "long" else RED}),
                    html.Span(f'{pos.get("lots", 0):.2f}', style={"fontSize": "9px", "color": MUTED2,
                              "fontFamily": "'JetBrains Mono',monospace"}),
                    html.Span(f'{pos.get("entry", 0):.5f}', style={"fontSize": "9px", "color": MUTED2,
                              "fontFamily": "'JetBrains Mono',monospace"}),
                    html.Span(f'${upnl:+.2f}', style={"fontSize": "10px", "color": upnl_col,
                              "fontWeight": "700", "fontFamily": "'JetBrains Mono',monospace"}),
                ],
            ))
        exposure_panel = html.Div(rows)

    # ── Loss streak ──────────────────────────────────────────────────────
    streak      = cap.get("consecutive_losses", 0)
    bypass_ct   = cap.get("bypass_count", 0)
    streak_col  = RED if streak >= 4 else YELLOW if streak >= 2 else GREEN
    streak_panel = html.Div(style={"display": "flex", "gap": "24px", "padding": "4px 0", "flexWrap": "wrap"}, children=[
        html.Div(children=[
            html.Div("CONSECUTIVE LOSSES", style={"fontSize": "8px", "color": MUTED2, "letterSpacing": "1px", "textTransform": "uppercase"}),
            html.Div(str(streak), style={"fontSize": "28px", "fontWeight": "700", "color": streak_col,
                                          "fontFamily": "'JetBrains Mono',monospace", "lineHeight": "1.1"}),
            html.Div("(cooldown triggers at 3)" if streak < 3 else "⚠ cooldown active" if cooldown else "resolved",
                     style={"fontSize": "9px", "color": MUTED}),
        ]),
        html.Div(children=[
            html.Div("BYPASSES (LIFETIME)", style={"fontSize": "8px", "color": MUTED2, "letterSpacing": "1px", "textTransform": "uppercase"}),
            html.Div(str(bypass_ct), style={"fontSize": "28px", "fontWeight": "700", "color": YELLOW if bypass_ct > 0 else MUTED2,
                                             "fontFamily": "'JetBrains Mono',monospace", "lineHeight": "1.1"}),
            html.Div("unauthorized open attempts", style={"fontSize": "9px", "color": MUTED}),
        ]),
        html.Div(children=[
            html.Div("WEEKLY PNL", style={"fontSize": "8px", "color": MUTED2, "letterSpacing": "1px", "textTransform": "uppercase"}),
            html.Div(f'${cap.get("weekly_pnl", 0):+.2f}',
                     style={"fontSize": "20px", "fontWeight": "700", "fontFamily": "'JetBrains Mono',monospace",
                            "lineHeight": "1.1", "color": GREEN if cap.get("weekly_pnl", 0) >= 0 else RED}),
            html.Div(f'Monthly: ${cap.get("monthly_pnl", 0):+.2f}', style={"fontSize": "9px", "color": MUTED}),
        ]),
    ])

    return (gate_panel,
            daily_lbl,   daily_bar,
            weekly_lbl,  weekly_bar,
            monthly_lbl, monthly_bar,
            lev_lbl,     lev_bar,
            exposure_panel, streak_panel)


# ── Pair profiles panel ───────────────────────────────────────────────────────

@app.callback(
    Output("pair-profiles-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_pair_profiles(n: int) -> html.Div:
    try:
        from core.adaptive_learning import adaptive
        from core.pattern_library import pattern_library
        pairs = get_pairs()
    except Exception:
        return html.Div("Pair profiles unavailable", style={"color": MUTED, "fontSize": "11px"})

    if not pairs:
        return html.Div("No pairs configured", style={"color": MUTED, "fontSize": "11px"})

    header = html.Div(
        style={"display": "grid", "gridTemplateColumns": "90px 60px 55px 80px 80px 70px",
               "gap": "5px", "padding": "4px 0", "borderBottom": "1px solid rgba(255,255,255,.06)",
               "marginBottom": "6px"},
        children=[html.Span(c, style={"fontSize": "8px", "color": MUTED, "fontWeight": "600", "textTransform": "uppercase"})
                  for c in ["PAIR", "W/L", "WIN%", "MIN SCORE", "LOT MULT", "STATUS"]],
    )
    rows = [header]
    for pair in pairs:
        s = adaptive.get_stats(pair)
        if not s:
            rows.append(html.Div(
                style={"display": "grid", "gridTemplateColumns": "90px 60px 55px 80px 80px 70px", "gap": "5px", "padding": "3px 0"},
                children=[html.Span(pair.rstrip("m"), style={"fontSize": "10px", "fontWeight": "600", "color": TEXT}),
                           html.Span("—", style={"fontSize": "9px", "color": MUTED2}),
                           html.Span("—", style={"fontSize": "9px", "color": MUTED2}),
                           html.Span("—", style={"fontSize": "9px", "color": MUTED2}),
                           html.Span("—", style={"fontSize": "9px", "color": MUTED2}),
                           html.Span("NO DATA", style={"fontSize": "9px", "color": MUTED2})],
            ))
            continue
        wr_col  = GREEN if s.win_rate >= 50 else RED
        status  = "LEARNING" if s.total_trades < 20 else ("STRONG" if s.win_rate >= 55 else "WEAK")
        st_col  = YELLOW if status == "LEARNING" else (GREEN if status == "STRONG" else RED)
        rows.append(html.Div(
            style={"display": "grid", "gridTemplateColumns": "90px 60px 55px 80px 80px 70px",
                   "gap": "5px", "padding": "3px 0", "borderBottom": "1px solid rgba(255,255,255,.03)"},
            children=[
                html.Span(pair.rstrip("m"), style={"fontSize": "10px", "fontWeight": "600", "color": TEXT}),
                html.Span(f"{s.wins}W/{s.losses}L", style={"fontSize": "9px", "color": MUTED2}),
                html.Span(f"{s.win_rate:.0f}%", style={"fontSize": "10px", "color": wr_col, "fontWeight": "700"}),
                html.Span(f"{s.min_score:.0f}", style={"fontSize": "10px", "color": YELLOW}),
                html.Span(f"{s.lot_multiplier:.2f}×", style={"fontSize": "10px", "color": ACCENT2}),
                html.Span(status, style={"fontSize": "9px", "color": st_col, "fontWeight": "600"}),
            ],
        ))
    return html.Div(rows)


# ── Settings page callbacks ───────────────────────────────────────────────────

@app.callback(
    Output("cfg-risk-pct",        "value"),
    Output("cfg-daily-target",    "value"),
    Output("cfg-max-loss",        "value"),
    Output("cfg-emergency-dd",    "value"),
    Output("cfg-max-concurrent",  "value"),
    Output("cfg-max-daily-trades","value"),
    Output("cfg-min-score",       "value"),
    Output("cfg-max-spread",      "value"),
    Output("cfg-news-buffer",     "value"),
    Output("cfg-news-filter",     "value"),
    Output("settings-sysinfo",    "children"),
    Input("interval", "n_intervals"),
)
def populate_settings(n: int):
    info_rows = [
        ("MT5 Server",    settings.mt5_server),
        ("MT5 Enabled",   str(settings.mt5_enabled)),
        ("Strategy",      settings.strategy),
        ("Entry TF",      settings.entry_tf),
        ("HTF Bias TF",   settings.htf_bias_tf),
        ("Dashboard",     f"{settings.dashboard_host}:{settings.dashboard_port}"),
        ("DB Path",       str(settings.db_path)),
        ("Dry Run",       str(settings.dry_run)),
        ("Log Level",     settings.log_level),
        ("Model Brief",   settings.model_brief),
        ("Model Report",  settings.model_daily_report),
        ("Telegram",      "configured" if settings.telegram_bot_token else "not configured"),
    ]
    sysinfo = html.Div(style={"display": "grid", "gridTemplateColumns": "140px 1fr", "gap": "4px 12px"}, children=[
        child
        for label, val in info_rows
        for child in [
            html.Span(label, style={"fontSize": "9px", "color": MUTED2, "textTransform": "uppercase",
                                    "letterSpacing": "1px", "padding": "3px 0"}),
            html.Span(val, style={"fontSize": "10px", "color": TEXT,
                                   "fontFamily": "'JetBrains Mono',monospace", "padding": "3px 0"}),
        ]
    ])
    return (
        settings.risk_per_trade_pct,
        settings.daily_target_pct,
        settings.max_daily_loss_pct,
        settings.emergency_drawdown_pct,
        settings.max_concurrent_trades,
        settings.max_trades_per_day,
        settings.min_signal_score,
        settings.max_spread_pips,
        settings.news_buffer_minutes,
        ["on"] if settings.news_filter_enabled else [],
        sysinfo,
    )


@app.callback(
    Output("settings-save-status", "children"),
    Output("settings-save-status", "style"),
    Input("settings-save-btn", "n_clicks"),
    State("cfg-risk-pct",         "value"),
    State("cfg-daily-target",     "value"),
    State("cfg-max-loss",         "value"),
    State("cfg-emergency-dd",     "value"),
    State("cfg-max-concurrent",   "value"),
    State("cfg-max-daily-trades", "value"),
    State("cfg-min-score",        "value"),
    State("cfg-max-spread",       "value"),
    State("cfg-news-buffer",      "value"),
    State("cfg-news-filter",      "value"),
    prevent_initial_call=True,
)
def save_settings(n_clicks, risk_pct, daily_tgt, max_loss, emerg_dd,
                  max_conc, max_daily, min_score, max_spread, news_buf, news_filter):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    ok_style = {"fontSize": "11px", "color": GREEN}
    err_style = {"fontSize": "11px", "color": RED}

    try:
        changes: list[tuple[str, str, object]] = [
            ("RISK_PER_TRADE_PCT",    "risk_per_trade_pct",    risk_pct),
            ("DAILY_TARGET_PCT",      "daily_target_pct",      daily_tgt),
            ("MAX_DAILY_LOSS_PCT",    "max_daily_loss_pct",    max_loss),
            ("EMERGENCY_DRAWDOWN_PCT","emergency_drawdown_pct",emerg_dd),
            ("MAX_CONCURRENT_TRADES", "max_concurrent_trades", max_conc),
            ("MAX_TRADES_PER_DAY",    "max_trades_per_day",    max_daily),
            ("MIN_SIGNAL_SCORE",      "min_signal_score",      min_score),
            ("MAX_SPREAD_PIPS",       "max_spread_pips",       max_spread),
            ("NEWS_BUFFER_MINUTES",   "news_buffer_minutes",   news_buf),
            ("NEWS_FILTER_ENABLED",   "news_filter_enabled",   bool(news_filter)),
        ]

        applied = []
        for env_key, attr, val in changes:
            if val is None:
                continue
            # Coerce types
            if attr in ("max_concurrent_trades", "max_trades_per_day", "news_buffer_minutes"):
                val = int(val)
            elif attr == "news_filter_enabled":
                pass  # already bool
            else:
                val = float(val)

            setattr(settings, attr, val)
            _write_env_key(env_key, str(val).lower() if isinstance(val, bool) else str(val))
            applied.append(env_key)

        logger.info(f"[Settings] Applied {len(applied)} changes: {', '.join(applied)}")
        return f"✓ Saved {len(applied)} settings ({datetime.now().strftime('%H:%M:%S')})", ok_style

    except Exception as e:
        logger.error(f"[Settings] Save failed: {e}")
        return f"✗ Error: {e}", err_style


# ── Knowledge base search ─────────────────────────────────────────────────────

@app.callback(
    Output("kb-search-results", "children"),
    Input("kb-search-btn",   "n_clicks"),
    Input("kb-search-input", "n_submit"),
    State("kb-search-input", "value"),
    prevent_initial_call=True,
)
def kb_search(n_clicks, n_submit, query: str) -> html.Div:
    triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
    if not triggered_value or not query or not query.strip():
        raise dash.exceptions.PreventUpdate

    try:
        from config.settings import settings
        from pathlib import Path
        vault = settings.obsidian_vault_path
        folder = settings.obsidian_aria_folder
        if not vault or not folder:
            return html.Div("Vault not configured", style={"color": RED, "fontSize": "11px"})

        base = Path(vault) / folder
        if not base.exists():
            return html.Div("Vault folder not found", style={"color": RED, "fontSize": "11px"})

        q = query.strip().lower()
        results = []
        for md_file in base.rglob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                if q in text.lower() or q in md_file.name.lower():
                    preview = ""
                    for line in text.splitlines():
                        if q in line.lower():
                            preview = line.strip()[:120]
                            break
                    rel = md_file.relative_to(base)
                    results.append((str(rel), preview))
                    if len(results) >= 15:
                        break
            except Exception:
                pass

        if not results:
            return html.Div(f"No results for '{query}'", style={"color": MUTED, "fontSize": "11px"})

        items = [html.Div(f"{len(results)} results for '{query}'",
                          style={"fontSize": "9px", "color": MUTED2, "marginBottom": "6px"})]
        for path, preview in results:
            items.append(html.Div(
                style={"padding": "5px 0", "borderBottom": "1px solid rgba(255,255,255,.03)"},
                children=[
                    html.Div(path, style={"fontSize": "10px", "fontWeight": "600", "color": ACCENT}),
                    html.Div(preview or "—", style={"fontSize": "10px", "color": MUTED2, "marginTop": "2px"}),
                ],
            ))
        return html.Div(items)
    except Exception as e:
        return html.Div(f"Search error: {e}", style={"color": RED, "fontSize": "11px"})


# ── Returns heatmap ───────────────────────────────────────────────────────────

@app.callback(
    Output("returns-heatmap", "figure"),
    Input("interval", "n_intervals"),
)
def update_returns_heatmap(n: int) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
                      margin=dict(l=40, r=10, t=30, b=30))
    try:
        from db.session import get_session
        from db.models import DayLog
        with get_session() as db:
            logs = db.query(DayLog).order_by(DayLog.date).all()

        if not logs:
            fig.add_annotation(text="No trade history", xref="paper", yref="paper",
                               x=0.5, y=0.5, font=dict(color=MUTED, size=12), showarrow=False)
            return fig

        import pandas as pd
        dates = [l.date for l in logs]
        pnls  = [l.realized_pnl for l in logs]
        s     = pd.Series(pnls, index=pd.to_datetime(dates))

        # Monthly grouping for heatmap (week of month vs month)
        s.index = pd.to_datetime(s.index)
        months  = s.resample("ME").sum()
        month_labels = [d.strftime("%b %Y") for d in months.index]
        values = months.values.tolist()

        fig.add_trace(go.Bar(
            x=month_labels,
            y=values,
            marker_color=[GREEN if v >= 0 else RED for v in values],
            name="Monthly P&L",
        ))
        fig.update_layout(
            title="Monthly Returns",
            xaxis=dict(color=MUTED, gridcolor="rgba(255,255,255,.04)"),
            yaxis=dict(color=MUTED, gridcolor="rgba(255,255,255,.04)", side="right"),
        )
    except Exception as e:
        fig.add_annotation(text=f"Returns data error: {e}", xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(color=MUTED, size=11), showarrow=False)
    return fig


# ── Strategy regime indicator ─────────────────────────────────────────────────

@app.callback(
    Output("regime-indicator-panel", "children"),
    Input("interval", "n_intervals"),
)
def update_regime_indicator(n: int) -> html.Div:
    try:
        from core.regime_classifier import classify_trade_regime
        pairs = get_pairs()
    except Exception:
        return html.Div("Regime classification unavailable", style={"color": MUTED, "fontSize": "11px"})

    items = []
    for pair in pairs:
        regime = classify_trade_regime(pair)
        col    = (GREEN if regime == "TRENDING"
                  else (YELLOW if regime == "RANGING"
                        else (RED if regime == "VOLATILE" else MUTED)))
        icon   = ("▲" if regime == "TRENDING"
                  else ("↔" if regime == "RANGING"
                        else ("⚡" if regime == "VOLATILE" else "?")))
        items.append(html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                   "padding": "4px 0", "borderBottom": "1px solid rgba(255,255,255,.03)"},
            children=[
                html.Span(pair.rstrip("m"), style={"fontSize": "11px", "fontWeight": "600", "color": TEXT}),
                html.Span(f"{icon} {regime}", style={"fontSize": "10px", "color": col, "fontWeight": "700",
                                                      "fontFamily": "'JetBrains Mono',monospace"}),
            ],
        ))
    return html.Div(items) if items else html.Div("No pairs", style={"color": MUTED, "fontSize": "11px"})


# ── Strategy equity curves ────────────────────────────────────────────────────

_STRATEGY_COLORS = {
    "SMC_TREND":        ACCENT,   # purple
    "SESSION_BREAKOUT": GREEN,
    "MEAN_REVERSION":   YELLOW,
    "RANGE_TRADING":    ACCENT2,  # blue
}

@app.callback(
    Output("strategy-equity-chart", "figure"),
    Output("strategy-equity-stats", "children"),
    Input("interval", "n_intervals"),
)
def update_strategy_equity(n: int):
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
        margin=dict(l=50, r=10, t=8, b=30),
        xaxis=dict(color=MUTED, gridcolor="rgba(255,255,255,.04)", zeroline=False),
        yaxis=dict(color=MUTED, gridcolor="rgba(255,255,255,.04)", zeroline=False, side="right"),
    )

    try:
        from core.strategy_equity import get_all_curves, get_summary
        curves  = get_all_curves()
        summary = get_summary()
    except Exception:
        curves  = {}
        summary = []

    if not curves:
        empty_fig.add_annotation(
            text="No strategy trades recorded yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(color=MUTED, size=12), showarrow=False,
        )
        no_data = html.Div("No data — trades will appear here after the first close",
                           style={"color": MUTED, "fontSize": "11px"})
        return empty_fig, no_data

    fig = go.Figure()
    for name, curve in curves.items():
        color = _STRATEGY_COLORS.get(name, MUTED)
        xs    = list(range(len(curve.equity)))
        fig.add_trace(go.Scatter(
            x=xs, y=curve.equity,
            mode="lines",
            name=name.replace("_", " "),
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{name}</b><br>Trade #%{{x}}<br>Equity: $%{{y:+.2f}}<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
        margin=dict(l=50, r=10, t=8, b=30),
        xaxis=dict(title="Trade #", color=MUTED, gridcolor="rgba(255,255,255,.04)", zeroline=False),
        yaxis=dict(title="Net P&L $", color=MUTED, gridcolor="rgba(255,255,255,.04)",
                   zeroline=True, zerolinecolor="rgba(255,255,255,.1)", side="right"),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=9, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PANEL2, font_color=TEXT, font_size=11),
    )

    # Stats table
    header = html.Div(
        style={"display": "grid",
               "gridTemplateColumns": "120px 50px 60px 70px 65px 70px",
               "gap": "5px", "padding": "4px 0",
               "borderBottom": "1px solid rgba(255,255,255,.06)", "marginBottom": "4px"},
        children=[
            html.Span(c, style={"fontSize": "8px", "color": MUTED, "fontWeight": "600",
                                "textTransform": "uppercase"})
            for c in ["STRATEGY", "TRADES", "WIN %", "NET P&L", "SHARPE", "MAX DD"]
        ],
    )
    rows = [header]
    for r in summary:
        color    = _STRATEGY_COLORS.get(r["strategy"], MUTED)
        wr_col   = GREEN if r["win_rate"] >= 50 else RED
        sh_col   = GREEN if r["sharpe"] > 0 else RED
        pnl_col  = GREEN if r["net_pnl"] >= 0 else RED
        rows.append(html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "120px 50px 60px 70px 65px 70px",
                   "gap": "5px", "padding": "3px 0",
                   "borderBottom": "1px solid rgba(255,255,255,.03)"},
            children=[
                html.Span(r["strategy"].replace("_", " "),
                          style={"fontSize": "10px", "fontWeight": "600", "color": color}),
                html.Span(str(r["trades"]),
                          style={"fontSize": "10px", "color": MUTED2}),
                html.Span(f"{r['win_rate']:.0f}%",
                          style={"fontSize": "10px", "color": wr_col, "fontWeight": "700"}),
                html.Span(f"${r['net_pnl']:+.2f}",
                          style={"fontSize": "10px", "color": pnl_col, "fontWeight": "700",
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(f"{r['sharpe']:+.3f}",
                          style={"fontSize": "10px", "color": sh_col, "fontWeight": "700",
                                 "fontFamily": "'JetBrains Mono',monospace"}),
                html.Span(f"{r['max_drawdown']:.1f}%",
                          style={"fontSize": "9px", "color": MUTED2,
                                 "fontFamily": "'JetBrains Mono',monospace"}),
            ],
        ))
    return fig, html.Div(rows)




# ── Entry point ────────────────────────────────────────────────────────────────

def run_dashboard(debug: bool = False) -> None:
    logger.info(f"Dashboard at http://{settings.dashboard_host}:{settings.dashboard_port}")
    app.run(
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        debug=debug,
        use_reloader=False,
    )
