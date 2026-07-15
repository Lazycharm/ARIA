"""
Confluence scorer — aggregates all signals into a 0-100 score.

Score breakdown:
  MTF alignment (D1+H4+H1+M15)   → 0-35 pts
  At key SMC level (OB/FVG)       → 0-20 pts
  RSI zone confirmation           → 0-15 pts
  Trend filter (EMA stack)        → 0-15 pts
  Session weight                  → 0-10 pts
  Spread filter                   → 0-5 pts

Score >= 65 → signal emitted to dashboard
Score >= 80 → auto-execution eligible (if session + capital allow)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from analysis.mtf import MTFBias
from analysis.smc import SMCResult


@dataclass
class ConfluenceScore:
    pair: str
    direction: str              # "long" | "short" | "wait"
    score: float                # 0-100
    breakdown: dict[str, float] = field(default_factory=dict)
    entry_reason: str = ""
    nearest_ob_level: Optional[float] = None
    nearest_fvg_level: Optional[float] = None

    @property
    def tradeable(self) -> bool:
        return self.score >= 65 and self.direction != "wait"

    @property
    def auto_executable(self) -> bool:
        return self.score >= 70 and self.direction != "wait"

    @property
    def signal_pct(self) -> float:
        """Dashboard gauge value: 0-100 for buy, negative for sell."""
        if self.direction == "long":
            return self.score
        elif self.direction == "short":
            return -self.score
        return 0.0

    def label(self) -> str:
        if self.direction == "long":
            return f"BUY {self.score:.0f}%"
        elif self.direction == "short":
            return f"SELL {self.score:.0f}%"
        return f"WAIT {self.score:.0f}%"


def score(
    pair: str,
    mtf: MTFBias,
    smc: SMCResult,
    spread_pips: float,
    news_blocked: bool,
    current_price: float,
    session_active: bool = True,   # kept for backward compat
    session_pts: float = 10.0,     # explicit session weight (0 / 5 / 10)
    df_m15: "Optional[pd.DataFrame]" = None,  # for ADX/MACD/Stochastic
    ml_boost: float = 0.0,         # Phase 6: ML win-probability adjustment
    sentiment_pts: float = 0.0,    # Phase 7: Reddit sentiment adjustment
) -> ConfluenceScore:
    """
    Calculate total confluence score for a trade opportunity.
    Direction is determined by MTF + SMC agreement.

    Score components (max ~100 after clamping):
      MTF alignment     0-35
      SMC zone (OB/FVG) 0-20
      RSI confirmation  0-15
      EMA trend filter  0-15
      ADX regime        -20 to +10  ← NEW: kills ranging-market trades
      BOS confirmation  0-10        ← NEW: trend break alignment
      MACD alignment    0-5         ← NEW: momentum confirmation
      Session weight    0-10
      News filter       -20 to +5
      Spread filter     -10 to +5
    """
    import pandas as pd

    breakdown: dict[str, float] = {}
    total = 0.0
    reasons: list[str] = []

    # ── 1. MTF Alignment (up to 35 pts) ──────────────────────────
    if mtf.direction == "ranging" or mtf.signal_direction == "wait":
        breakdown["mtf"] = 0.0
    else:
        mtf_pts = 0.0
        if mtf.aligned:
            mtf_pts = 35.0
            reasons.append(f"Full MTF alignment {mtf.direction}")
        elif mtf.confidence >= 0.6:
            mtf_pts = 22.0
            reasons.append(f"Strong MTF bias {mtf.direction}")
        elif mtf.confidence >= 0.4:
            mtf_pts = 12.0
            reasons.append(f"Weak MTF bias {mtf.direction}")
        breakdown["mtf"] = mtf_pts
        total += mtf_pts

    direction = mtf.signal_direction  # "long" | "short" | "wait"

    # ── 2. SMC Level (up to 20 pts) ──────────────────────────────
    smc_pts = 0.0
    ob_level = None
    fvg_level = None

    if direction == "long":
        ob  = smc.nearest_ob(current_price, "bullish")
        fvg = smc.nearest_fvg(current_price, "bullish")
        if ob and ob.price_inside(current_price):
            smc_pts += 12.0
            ob_level = ob.mid
            reasons.append(f"Inside bullish OB @ {ob.mid:.5f}")
        elif ob and abs(ob.mid - current_price) / current_price < 0.001:
            smc_pts += 6.0
            ob_level = ob.mid
            reasons.append(f"Near bullish OB @ {ob.mid:.5f}")
        if fvg and fvg.price_inside(current_price):
            smc_pts += 8.0
            fvg_level = (fvg.top + fvg.bottom) / 2
            reasons.append(f"Inside bullish FVG @ {fvg_level:.5f}")

    elif direction == "short":
        ob  = smc.nearest_ob(current_price, "bearish")
        fvg = smc.nearest_fvg(current_price, "bearish")
        if ob and ob.price_inside(current_price):
            smc_pts += 12.0
            ob_level = ob.mid
            reasons.append(f"Inside bearish OB @ {ob.mid:.5f}")
        elif ob and abs(ob.mid - current_price) / current_price < 0.001:
            smc_pts += 6.0
            ob_level = ob.mid
            reasons.append(f"Near bearish OB @ {ob.mid:.5f}")
        if fvg and fvg.price_inside(current_price):
            smc_pts += 8.0
            fvg_level = (fvg.top + fvg.bottom) / 2
            reasons.append(f"Inside bearish FVG @ {fvg_level:.5f}")

    breakdown["smc"] = min(smc_pts, 20.0)
    total += breakdown["smc"]

    # ── 3. RSI Confirmation (up to 15 pts) ───────────────────────
    rsi = mtf.m15.rsi
    rsi_pts = 0.0
    if direction == "long" and 30 <= rsi <= 55:
        rsi_pts = 15.0
        reasons.append(f"RSI favors long ({rsi:.0f})")
    elif direction == "short" and 45 <= rsi <= 70:
        rsi_pts = 15.0
        reasons.append(f"RSI favors short ({rsi:.0f})")
    elif direction == "long" and rsi < 30:
        rsi_pts = 8.0
    elif direction == "short" and rsi > 70:
        rsi_pts = 8.0
    elif direction == "long" and 55 < rsi <= 65:
        rsi_pts = 8.0   # momentum building, partial credit
    elif direction == "short" and 35 <= rsi < 45:
        rsi_pts = 8.0
    breakdown["rsi"] = rsi_pts
    total += rsi_pts

    # ── 4. EMA Trend Filter (up to 15 pts) ───────────────────────
    ema_pts = 0.0
    m15 = mtf.m15
    if direction == "long" and m15.direction == "bullish":
        ema_pts = 15.0
        reasons.append("M15 EMA trend aligned long")
    elif direction == "short" and m15.direction == "bearish":
        ema_pts = 15.0
        reasons.append("M15 EMA trend aligned short")
    elif m15.direction == "ranging":
        ema_pts = 5.0
    breakdown["ema"] = ema_pts
    total += ema_pts

    # ── 5. ADX Regime Filter (-20 to +10 pts) ────────────────────
    # The single biggest win-rate lever: don't trade in choppy markets
    adx_pts = 0.0
    if df_m15 is not None and "adx" in df_m15.columns:
        adx = df_m15["adx"].iloc[-1]
        di_p = df_m15["di_p"].iloc[-1] if "di_p" in df_m15.columns else 0
        di_m = df_m15["di_m"].iloc[-1] if "di_m" in df_m15.columns else 0
        if not pd.isna(adx):
            if adx >= 30:
                adx_pts = 10.0
                reasons.append(f"Strong trend ADX={adx:.0f}")
                # DI alignment: extra confirmation
                if direction == "long" and di_p > di_m:
                    adx_pts += 3.0
                elif direction == "short" and di_m > di_p:
                    adx_pts += 3.0
            elif adx >= 20:
                adx_pts = 5.0
            elif adx >= 15:
                adx_pts = 0.0   # neutral — neither good nor bad
            else:
                # ADX < 15 = ranging/choppy → hard penalty
                adx_pts = -20.0
                reasons.append(f"Ranging market ADX={adx:.0f} (skip)")
    breakdown["adx"] = adx_pts
    total += adx_pts

    # ── 6. BOS Confirmation (0-10 pts) ───────────────────────────
    # Break of Structure in same direction = trend continuation confirmed
    bos_pts = 0.0
    if smc.bos:
        last_bos = smc.bos[-1]
        if (direction == "long"  and last_bos["direction"] == "bullish") or \
           (direction == "short" and last_bos["direction"] == "bearish"):
            bos_pts = 10.0
            reasons.append(f"BOS {last_bos['direction'].upper()} confirmed")
        else:
            # Opposing BOS — price just broke the other way
            bos_pts = -8.0
            reasons.append("Opposing BOS — counter-trend risk")
    breakdown["bos"] = bos_pts
    total += bos_pts

    # ── 7. MACD Alignment (0-5 pts) ──────────────────────────────
    macd_pts = 0.0
    if df_m15 is not None and "macd_hist" in df_m15.columns:
        hist = df_m15["macd_hist"].iloc[-1]
        prev_hist = df_m15["macd_hist"].iloc[-2] if len(df_m15) > 1 else hist
        if not pd.isna(hist):
            aligned = (direction == "long" and hist > 0) or (direction == "short" and hist < 0)
            rising  = (direction == "long" and hist > prev_hist) or (direction == "short" and hist < prev_hist)
            if aligned and rising:
                macd_pts = 5.0
                reasons.append("MACD momentum aligned")
            elif aligned:
                macd_pts = 3.0
            else:
                macd_pts = -3.0   # opposing MACD
    breakdown["macd"] = macd_pts
    total += macd_pts

    # ── 8. Stochastic Guard (0 or -10 pts) ───────────────────────
    # Avoid entering at extremes in wrong direction
    stoch_pts = 0.0
    if df_m15 is not None and "stoch_k" in df_m15.columns:
        sk = df_m15["stoch_k"].iloc[-1]
        if not pd.isna(sk):
            if direction == "long" and sk > 85:
                stoch_pts = -10.0
                reasons.append(f"Stochastic overbought ({sk:.0f}) — risky long")
            elif direction == "short" and sk < 15:
                stoch_pts = -10.0
                reasons.append(f"Stochastic oversold ({sk:.0f}) — risky short")
            elif (direction == "long" and 20 <= sk <= 60) or (direction == "short" and 40 <= sk <= 80):
                stoch_pts = 3.0   # clean entry zone
    breakdown["stoch"] = stoch_pts
    total += stoch_pts

    # ── 9. Session Weight (0 / 5 / 10 pts) ───────────────────────
    effective_session_pts = session_pts if session_pts > 0 else (10.0 if session_active else 0.0)
    breakdown["session"] = effective_session_pts
    total += effective_session_pts
    if effective_session_pts == 0:
        reasons.append("Outside active session")
    elif effective_session_pts < 10:
        reasons.append("Off-session pair (partial weight)")

    # ── 10. News Filter (-20 or +5 pts) ──────────────────────────
    if news_blocked:
        total -= 20.0
        breakdown["news"] = -20.0
    else:
        breakdown["news"] = 5.0
        total += 5.0

    # ── 11. Spread Filter (-10 or +5 pts) ────────────────────────
    from config.settings import settings
    if spread_pips <= settings.max_spread_pips:
        breakdown["spread"] = 5.0
        total += 5.0
    else:
        breakdown["spread"] = -10.0
        total -= 10.0

    # ── 12. ML Boost (-15 to +12 pts) — Phase 6 ─────────────────
    if ml_boost != 0.0:
        breakdown["ml"] = ml_boost
        total += ml_boost
        if ml_boost >= 6.0:
            reasons.append(f"ML confident win ({ml_boost:+.0f}pts)")
        elif ml_boost <= -8.0:
            reasons.append(f"ML expects loss ({ml_boost:+.0f}pts)")

    # ── 13. Sentiment (-10 to +10 pts) — Phase 7 ─────────────────
    if sentiment_pts != 0.0:
        breakdown["sentiment"] = sentiment_pts
        total += sentiment_pts
        if abs(sentiment_pts) >= 5.0:
            direction_word = "bullish" if sentiment_pts > 0 else "bearish"
            reasons.append(f"Reddit {direction_word} ({sentiment_pts:+.0f}pts)")

    total = max(0.0, min(100.0, total))

    return ConfluenceScore(
        pair=pair,
        direction=direction,
        score=round(total, 1),
        breakdown=breakdown,
        entry_reason=" | ".join(reasons) if reasons else "No strong confluence",
        nearest_ob_level=ob_level,
        nearest_fvg_level=fvg_level,
    )
