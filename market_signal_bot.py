"""
Market Signal Bot — MAX Edition
Runs once per invocation via GitHub Actions (every 10 min).
Keys are read from environment variables — set them as GitHub Secrets.

Upgrades:
  1.  Live intraday prices (today's open vs now, signed move)
  2.  AI sentiment/impact score (1-10) factored into signal strength
  3.  AI entry, stop loss, and target price levels
  4.  Economic calendar — high-impact events injected into AI context
  5.  Duplicate story filter — similar headlines from different sources skipped
  6.  Weekly summary — Monday recap of all signals from the past week
  7.  MACD indicator — momentum confirmation alongside RSI
  8.  Volume spike detection — 2x average volume = confirmed move
  9.  Multi-source confirmation — 3+ outlets reporting = score boost
  10. Support & Resistance — auto-detected from recent price data
  11. Real order flow via Polygon.io — delta, POC, L2 imbalances, absorption
  12. Daily trend context — big-picture RSI + EMA from daily bars
  13. DXY (Dollar Index) — shown as context for macro/forex signals
  14. Signal confidence rating — X/5 indicators aligned
  15. Streak tracking — consecutive BUY/SELL signals per symbol
"""

import re
import feedparser
import requests
import anthropic
import os
import time
import json
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo
import yfinance as yf

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")
CHAT_ID_2      = os.environ.get("CHAT_ID_2", "")   # optional second recipient
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_KEY", "")
POLYGON_KEY    = os.environ.get("POLYGON_KEY", "")  # Polygon.io API key for real order flow

SCORE_THRESHOLD      = 55
MAX_HEADLINE_AGE_MIN = 45
SYMBOL_COOLDOWN_MIN  = 15
SIMILARITY_THRESHOLD = 0.78
SEEN_FILE            = "seen_headlines.json"
MEMORY_FILE          = "signal_memory.json"
OUTCOME_EXPIRY_DAYS  = 3   # signals are marked EXPIRED if neither TP nor SL hit within 3 days
MAX_SIGNALS_PER_RUN  = 5   # Cap signals per 5-min run — prevents Telegram spam during news floods

if not all([TELEGRAM_TOKEN, CHAT_ID, ANTHROPIC_KEY]):
    raise SystemExit("ERROR: TELEGRAM_TOKEN, CHAT_ID, and ANTHROPIC_KEY must be set.")

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ── Session-level endpoint-failure tracker ────────────────────────────────
# Polygon endpoints that are not in the subscription return a quick non-200
# and cost almost no time.  But if they time out (server slow / firewall),
# repeated 5-second waits per signal add up fast.  After two consecutive
# failures the endpoint is skipped for the rest of this GitHub Actions job.
_ENDPOINT_SKIP: dict = {}   # endpoint_name -> consecutive fail count

def _ep_ok(name: str) -> bool:
    """True unless this endpoint has failed ≥2 times this session."""
    return _ENDPOINT_SKIP.get(name, 0) < 2

def _ep_fail(name: str) -> None:
    """Increment failure counter; at 2 the endpoint is skipped this session."""
    _ENDPOINT_SKIP[name] = _ENDPOINT_SKIP.get(name, 0) + 1

def _ep_reset(name: str) -> None:
    """Clear failure counter on success so a later recovery is honoured."""
    _ENDPOINT_SKIP.pop(name, None)

# ─────────────────────────────
# FEEDS
# ─────────────────────────────
NEWS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/UKBusinessNews",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://www.investing.com/rss/news.rss",
    "https://www.investing.com/rss/news_8.rss",    # Stock markets
    "https://www.investing.com/rss/news_14.rss",   # Commodities
    "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
    "https://www.aljazeera.com/xml/rss/all.xml",   # Live geopolitical/war updates
    "https://www.kitco.com/rss/kitco-news.xml",    # Gold & precious metals focus
    "https://oilprice.com/rss/main",               # Oil & energy focus
]

ANALYSIS_FEEDS = [
    "https://www.investing.com/rss/news_25.rss",   # General financial analysis
]

# Weekend: gold and crude oil trade Sunday evening onward — keep minimal feeds
WEEKEND_FEEDS = [
    ("https://www.kitco.com/rss/kitco-news.xml", "📰 NEWS"),
    ("https://oilprice.com/rss/main",             "📰 NEWS"),
    ("https://feeds.reuters.com/reuters/businessNews", "📰 NEWS"),
]

ALL_FEEDS = (
    [(url, "📰 NEWS") for url in NEWS_FEEDS] +
    [(url, "📊 ANALYSIS") for url in ANALYSIS_FEEDS]
)

# ─────────────────────────────
# ASSET MAP & FRIENDLY NAMES
# ─────────────────────────────
ASSET_MAP = {
    "GC=F":  ["gold", "xau", "bullion", "precious metal"],
    "SI=F":  ["silver", "xag", "silver futures", "silver price", "comex silver"],
    "HG=F":  ["copper", "hg", "comex copper", "copper price", "copper futures",
              "copper supply", "copper demand", "copper mine", "freeport", "southern copper"],
    "ALI=F": ["aluminium", "aluminum", "alcoa", "bauxite", "rusal", "aluminium smelter",
              "aluminum smelter", "norsk hydro", "hindalco", "aluminium tariff",
              "aluminum tariff", "aluminium supply", "aluminum supply"],
    "CL=F":  ["oil", "crude", "wti", "brent", "opec", "petroleum", "energy"],
    "^GSPC": ["s&p", "spx", "spy", "sp500", "s&p 500", "equities", "wall street"],
    "QQQ":   ["nasdaq", "qqq", "nq", "us100", "tech 100"],
}

ASSET_NAMES = {
    "GC=F":  "Gold (XAU/USD)",
    "SI=F":  "Silver (XAG/USD)",
    "HG=F":  "Copper Futures (HG)",
    "ALI=F": "Aluminium Futures",
    "CL=F":  "Crude Oil (WTI)",
    "^GSPC": "S&P 500",
    "QQQ":   "US Tech 100 (Nasdaq)",
}

# Context-only symbols — fetched for data, never trigger signals
CONTEXT_SYMBOLS = {
    "DX-Y.NYB": "US Dollar Index (DXY)",
    "^VIX":     "CBOE Volatility Index (VIX)",
}

# Crypto symbols — trade 24/7 including weekends
# Crypto and forex removed — bot now covers futures + US indices only
CRYPTO_SYMBOLS: set = set()
FOREX_SYMBOLS:  set = set()

# Polygon/Massive symbol mapping (yfinance symbol → Massive ticker)
# Confirmed working on Futures Developer + Stocks Basic plans.
# Gold/Aluminium continuous futures (GC, ALI) are not in the Massive plan —
# GLD and DBB are highly liquid ETF proxies that give real Polygon order flow data.
POLYGON_SYMBOLS = {
    "^GSPC": "SPY",    # S&P 500 → SPDR S&P 500 ETF (direct proxy)
    "QQQ":   "QQQ",    # Nasdaq 100 ETF (direct)
    "CL=F":  "CL",     # Crude Oil → NYMEX continuous futures (real futures data) ✅
    "GC=F":  "GLD",    # Gold → SPDR Gold ETF (tracks gold 1:1; GC futures not in plan) ✅
    "SI=F":  "SI",     # Silver → COMEX continuous futures (confirmed real data) ✅
    "HG=F":  "HG",     # Copper → COMEX continuous futures (confirmed real data) ✅
    "ALI=F": "DBB",    # Aluminium → Invesco DB Base Metals ETF (33% alu; ALI not in plan) ✅
}

def friendly(symbol: str) -> str:
    return ASSET_NAMES.get(symbol, CONTEXT_SYMBOLS.get(symbol, symbol))

# ─────────────────────────────
# KEYWORDS
# ─────────────────────────────
KEYWORDS = [
    "cpi", "inflation", "fed", "fomc", "interest rate",
    "rate hike", "rate cut", "nfp", "jobs report",
    "gdp", "recession", "powell", "treasury", "yield", "dollar", "dxy",
    "gold", "xau", "bullion", "precious metal",
    "silver", "xag", "comex silver",
    "copper", "comex copper", "freeport", "southern copper", "copper mine",
    "aluminium", "aluminum", "alcoa", "bauxite", "rusal", "norsk hydro", "hindalco",
    "oil", "crude", "wti", "brent", "opec", "petroleum", "energy",
    "natural gas", "lng", "pipeline",
    "s&p", "spx", "spy", "sp500", "s&p 500", "wall street", "equities",
    "nasdaq", "qqq", "nq", "us100", "tech 100",
    "tariff", "trade war", "sanctions", "geopolit", "conflict", "war",
]

# Crypto and forex keywords no longer needed — assets removed from bot
CRYPTO_KEYWORDS: list = []

MACRO_KEYWORDS = [
    "fed", "fomc", "powell", "interest rate", "rate hike", "rate cut",
    "cpi", "inflation", "nfp", "jobs report", "gdp", "recession",
    "treasury", "yield", "dollar", "dxy",
]

def is_important(text: str) -> bool:
    return any(k in text.lower() for k in KEYWORDS)

def is_macro(text: str) -> bool:
    return any(k in text.lower() for k in MACRO_KEYWORDS)

def is_crypto_headline(text: str) -> bool:
    return any(k in text.lower() for k in CRYPTO_KEYWORDS)

# ─────────────────────────────
# NEWS SENTIMENT SCORING
# Pre-screens headlines before we burn AI tokens on them. Headlines with no
# directional language (or perfectly balanced) get the same baseline score
# they always did; headlines with strong bullish/bearish language get a
# small score nudge so genuinely directional news rises faster.
# ─────────────────────────────
BULLISH_KEYWORDS = [
    "rally", "surge", "soar", "jump", "spike", "climb", "rise", "gain",
    "boost", "beat", "beats", "exceed", "exceeds", "outperform", "upgrade",
    "bullish", "breakout", "record high", "all-time high", "ath", "strong",
    "robust", "solid", "approve", "approved", "accelerate", "expand",
    "growth", "profit", "earnings beat", "buy rating", "long",
    "rate cut", "stimulus", "easing", "dovish", "soft inflation",
    "recovery", "rebound", "upbeat", "optimism", "positive",
]

BEARISH_KEYWORDS = [
    "plunge", "crash", "tumble", "slump", "drop", "fall", "decline", "sink",
    "miss", "misses", "underperform", "downgrade", "bearish", "breakdown",
    "record low", "all-time low", "weak", "fragile", "reject", "rejected",
    "decelerate", "contract", "loss", "losses", "earnings miss", "sell rating",
    "short", "rate hike", "tightening", "hawkish", "hot inflation",
    "recession", "slowdown", "downturn", "pessimism", "negative",
    "war", "sanctions", "default", "bankruptcy", "fraud", "investigation",
    "ban", "banned", "halt", "halts", "concerns", "fears", "panic",
]

def score_headline_sentiment(title: str) -> tuple:
    """Return (sentiment_score, label) where score ranges -100..+100.

    >  20: bullish-leaning headline
    < -20: bearish-leaning headline
    else: neutral / mixed
    """
    t           = title.lower()
    bull_hits   = sum(1 for k in BULLISH_KEYWORDS if k in t)
    bear_hits   = sum(1 for k in BEARISH_KEYWORDS if k in t)
    total_hits  = bull_hits + bear_hits
    if total_hits == 0:
        return 0, "neutral"
    raw = (bull_hits - bear_hits) / total_hits * 100
    raw = round(raw)
    if raw >= 30:
        return raw, "bullish"
    if raw <= -30:
        return raw, "bearish"
    return raw, "mixed"

def is_weekend() -> bool:
    """True on Saturday and Sunday.

    Gold and crude oil futures trade Sunday evening onward, so Sunday after
    22:00 UTC is treated as 'market open' by is_market_open() — weekend mode
    only applies to Saturday and most of Sunday.
    """
    now = datetime.now(timezone.utc)
    return now.weekday() >= 5  # Saturday=5, Sunday=6

# ─────────────────────────────
# TIME HELPERS
# ─────────────────────────────
ZURICH_TZ = ZoneInfo("Europe/Zurich")

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def zurich_now() -> datetime:
    return datetime.now(ZURICH_TZ)

def is_zurich_quiet_hours() -> bool:
    """True between 23:00 and 06:30 Zürich time — no signals sent during this window."""
    zh = zurich_now()
    h, m = zh.hour, zh.minute
    if h == 23:
        return True
    if h < 6 or (h == 6 and m < 30):
        return True
    return False

def should_send_morning_recap(state: dict) -> bool:
    """True once per day when the clock crosses 06:30 Zürich and recap not yet sent."""
    zh = zurich_now()
    # Only trigger from 06:30 onwards
    if zh.hour < 6 or (zh.hour == 6 and zh.minute < 30):
        return False
    today = zh.date().isoformat()
    return state.get("__last_recap_date__") != today

def is_market_open() -> bool:
    now  = now_utc()
    day  = now.weekday()
    hour = now.hour
    if day == 5:
        return False
    if day == 6:
        return hour >= 22
    if day == 4 and hour >= 22:
        return False
    return True

def get_session_label() -> str:
    hour = now_utc().hour
    day  = now_utc().weekday()
    if day >= 5:
        return "Weekend"
    if hour >= 22 or hour < 7:
        return "Asia Session"
    if 7 <= hour < 13:
        return "London Session"
    if 13 <= hour < 17:
        return "London/NY Overlap"
    if 17 <= hour < 22:
        return "NY Session"
    return "Off Hours"

# ─────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────
def calculate_rsi(closes: list, period: int = 14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(closes) - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)

def calculate_ema(closes: list, period: int):
    if len(closes) < period:
        return None
    k   = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return round(ema, 4)

def calculate_macd(closes: list, fast: int = 12, slow: int = 26, signal_period: int = 9):
    """Returns (macd_line, signal_line, histogram) or (None, None, None)."""
    if len(closes) < slow + signal_period:
        return None, None, None
    k_fast = 2 / (fast + 1)
    k_slow = 2 / (slow + 1)
    k_sig  = 2 / (signal_period + 1)

    ef = sum(closes[:fast]) / fast
    es = sum(closes[:slow]) / slow
    for i in range(fast, slow):
        ef = closes[i] * k_fast + ef * (1 - k_fast)

    macd_series = []
    for i in range(slow, len(closes)):
        ef = closes[i] * k_fast + ef * (1 - k_fast)
        es = closes[i] * k_slow + es * (1 - k_slow)
        macd_series.append(ef - es)

    if len(macd_series) < signal_period:
        return None, None, None

    sig = sum(macd_series[:signal_period]) / signal_period
    for m in macd_series[signal_period:]:
        sig = m * k_sig + sig * (1 - k_sig)

    return round(macd_series[-1], 6), round(sig, 6), round(macd_series[-1] - sig, 6)

def calculate_atr(highs: list, lows: list, closes: list, period: int = 14):
    """Wilder's Average True Range — measures actual volatility."""
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 6)

def rsi_label(rsi) -> str:
    if rsi is None:
        return "n/a"
    if rsi >= 70:
        return f"{rsi} (overbought)"
    if rsi <= 30:
        return f"{rsi} (oversold)"
    return f"{rsi} (neutral)"

def macd_label(macd, signal) -> str:
    if macd is None or signal is None:
        return "n/a"
    diff = abs(macd - signal)
    if macd > signal:
        return f"Bullish (above signal by {diff:.5f})"
    return f"Bearish (below signal by {diff:.5f})"

def trend_label(price: float, ema20, ema50) -> str:
    if ema20 is None or ema50 is None:
        return "n/a"
    if price > ema20 > ema50:
        return "Uptrend (price > EMA20 > EMA50)"
    if price < ema20 < ema50:
        return "Downtrend (price < EMA20 < EMA50)"
    return "Mixed / ranging"

def volume_label(ratio) -> str:
    if ratio is None:
        return "n/a"
    if ratio >= 3.0:
        return f"{ratio:.1f}x avg (strong spike ⚡)"
    if ratio >= 2.0:
        return f"{ratio:.1f}x avg (spike)"
    if ratio >= 1.5:
        return f"{ratio:.1f}x avg (elevated)"
    return f"{ratio:.1f}x avg (normal)"

# ─────────────────────────────
# SIGNAL CONFIDENCE
# Counts how many indicators agree with the bias (out of up to 7 now —
# RSI, MACD, 15m trend, 1h trend, daily trend, volume, RSI divergence).
# ─────────────────────────────
def signal_confidence(data: dict, bias: str):
    """Returns (score, total) where score = indicators agreeing with bias.

    15 checks across three layers — each one earns its place:

    LAYER A — Technical / macro (yfinance + context)
      1. Multi-TF trend alignment  — 3+/4 timeframes agree
      2. RSI extremes only         — ≤35 oversold (BUY), ≥65 overbought (SELL)
      3. Volume confirmation       — last bar ≥ 1.5× average

    LAYER B — Core Polygon order flow
      4. Cumulative delta bias     — net buy/sell pressure direction
      5. Buy%                      — Polygon buy-side pressure ≥60% / ≤40%
      6. VWAP position             — price above/below session-anchored VWAP
      7. VPOC position             — price above/below most-traded price level
      8. Delta flip                — momentum reversal confirming direction

    LAYER C — Institutional order flow events (Polygon)
      9.  Market structure         — HH+HL bullish / LH+LL bearish
      10. Imbalance                — 7+/10 or 8+/10 bars one-sided
      11. Opening range breakout   — price broke above/below 15-min OR
      12. Block trade direction    — institutional prints align with bias
      13. Bar streak               — 3+ consecutive bars in bias direction
      14. Exhaustion (opposing)    — OPPOSING side exhausted → fuel for bias
      15. Absorption               — DEFENDING side absorbing → holding level
    """
    if bias not in ("BUY", "SELL"):
        return 0, 0

    score, total = 0, 0
    poly_flow = data.get("polygon_flow") or {}
    price     = data.get("price")

    # ── LAYER A: Technical / macro ────────────────────────────────────────────

    # 1. Multi-timeframe trend alignment (counts as ONE vote)
    trends = [
        data.get("trend", ""),
        data.get("hourly_trend", ""),
        data.get("4h_trend", ""),
        data.get("daily_trend", ""),
    ]
    up_count   = sum(1 for t in trends if t and "Uptrend"   in t)
    down_count = sum(1 for t in trends if t and "Downtrend" in t)
    if (up_count + down_count) >= 2:
        total += 1
        if bias == "BUY"  and up_count   >= 3: score += 1
        elif bias == "SELL" and down_count >= 3: score += 1

    # 2. RSI at genuine extremes only
    rsi = data.get("rsi")
    if rsi is not None:
        total += 1
        if bias == "BUY"  and rsi <= 35: score += 1
        elif bias == "SELL" and rsi >= 65: score += 1

    # 3. Volume confirmation
    vol = data.get("vol_ratio")
    if vol is not None:
        total += 1
        if vol >= 1.5: score += 1

    # ── LAYER B: Core Polygon order flow ─────────────────────────────────────

    # 4. Cumulative delta direction
    of_src = poly_flow if poly_flow else (data.get("order_flow") or {})
    if of_src:
        total += 1
        if bias == "BUY"  and of_src.get("of_bias") == "bullish": score += 1
        elif bias == "SELL" and of_src.get("of_bias") == "bearish": score += 1

    # 5. Buy% pressure
    buy_pct = of_src.get("buy_pct")
    if buy_pct is not None:
        total += 1
        if bias == "BUY"  and buy_pct >= 60: score += 1
        elif bias == "SELL" and buy_pct <= 40: score += 1

    # 6. VWAP position — prefer Polygon real VWAP, fall back to yfinance
    vwap_val = poly_flow.get("vwap")
    if vwap_val is None:
        vwap_d   = data.get("vwap") or {}
        vwap_val = vwap_d.get("vwap") if isinstance(vwap_d, dict) else None
    if vwap_val and price:
        total += 1
        if bias == "BUY"  and price < vwap_val: score += 1
        elif bias == "SELL" and price > vwap_val: score += 1

    # 7. VPOC position — price above VPOC = buyers in control, below = sellers
    vpoc = poly_flow.get("vpoc")
    if vpoc and price:
        total += 1
        if bias == "BUY"  and price > vpoc: score += 1
        elif bias == "SELL" and price < vpoc: score += 1

    # 8. Delta flip — momentum reversal confirming direction
    flip = of_src.get("delta_flip", "")
    if flip:
        total += 1
        if bias == "BUY"  and "bullish" in flip.lower(): score += 1
        elif bias == "SELL" and "bearish" in flip.lower(): score += 1

    # ── LAYER C: Institutional order flow events ──────────────────────────────

    # 9. Market structure alignment
    mkt = poly_flow.get("mkt_structure", "")
    if mkt and mkt != "ranging":
        total += 1
        if bias == "BUY"  and "bullish" in mkt: score += 1
        elif bias == "SELL" and "bearish" in mkt: score += 1

    # 10. Imbalance — directional dominance
    imbalance = poly_flow.get("imbalance", "")
    if imbalance:
        total += 1
        if bias == "BUY"  and "buy" in imbalance.lower(): score += 1
        elif bias == "SELL" and "sell" in imbalance.lower(): score += 1

    # 11. Opening range breakout
    or_st = poly_flow.get("or_status", "")
    if or_st and "inside" not in or_st:
        total += 1
        if bias == "BUY"  and "bullish breakout" in or_st: score += 1
        elif bias == "SELL" and "bearish breakout" in or_st: score += 1

    # 12. Block trade direction — institutional print aligns with bias
    block = poly_flow.get("block_trade", "")
    if block:
        total += 1
        if bias == "BUY"  and "buy" in block.lower(): score += 1
        elif bias == "SELL" and "sell" in block.lower(): score += 1

    # 13. Bar streak — consecutive bars in bias direction
    streak = poly_flow.get("bar_streak", "")
    if streak:
        total += 1
        if bias == "BUY"  and "buying" in streak: score += 1
        elif bias == "SELL" and "selling" in streak: score += 1

    # 14. Exhaustion — of the OPPOSING side = fuel still available for bias
    #     "Selling exhaustion" + BUY  = sellers running out → good for buyers
    #     "Buying exhaustion"  + SELL = buyers running out → good for sellers
    #     Exhaustion of the SAME side as bias = warning → adds to total, no score
    exhaustion = poly_flow.get("exhaustion", "")
    if exhaustion:
        total += 1
        if bias == "BUY"  and "selling" in exhaustion.lower(): score += 1
        elif bias == "SELL" and "buying"  in exhaustion.lower(): score += 1

    # 15. Absorption — of the defending side = confirms level is holding
    #     "Buyer absorption"  = buyers absorbing sellers → floor holding → BUY
    #     "Seller absorption" = sellers absorbing buyers → ceiling holding → SELL
    absorption = poly_flow.get("absorption", "")
    if absorption:
        total += 1
        if bias == "BUY"  and "buyer absorption"  in absorption.lower(): score += 1
        elif bias == "SELL" and "seller absorption" in absorption.lower(): score += 1

    # ── LAYER D: New supplementary data ──────────────────────────────────────

    # 16. Options PCR directional bias
    opts = data.get("options_flow") or {}
    pcr  = opts.get("pcr")
    if pcr is not None:
        total += 1
        if bias == "BUY"  and pcr < 0.7:  score += 1   # call-heavy = bullish options positioning
        elif bias == "SELL" and pcr > 1.2: score += 1   # put-heavy = bearish options positioning

    # 17. Unusual options activity direction
    unusual_sum = opts.get("unusual_summary", "")
    if unusual_sum and "both sides" not in unusual_sum.lower():
        total += 1
        if bias == "BUY"  and "call" in unusual_sum.lower(): score += 1
        elif bias == "SELL" and "put"  in unusual_sum.lower(): score += 1

    # 18. Treasury yield macro signal (equities only)
    yields  = data.get("treasury_yields") or {}
    eq_sig  = yields.get("equity_signal", "")
    if eq_sig:
        total += 1
        if bias == "BUY"  and eq_sig == "bullish_equities": score += 1
        elif bias == "SELL" and eq_sig == "bearish_equities": score += 1

    # 19. Short squeeze setup (BUY signal on heavily shorted instrument)
    short = data.get("short_data") or {}
    if short.get("squeeze_setup") and bias == "BUY":
        total += 1
        score += 1

    # 19b. Short volume direction — intraday short sale ratio
    #   SELL: heavy short volume (>55%) = active bearish pressure, agrees with SELL
    #   BUY:  light short volume (<40%) = bears stepping back, agrees with BUY
    svr = short.get("short_vol_ratio")
    if svr is not None:
        total += 1
        if bias == "SELL" and svr > 55: score += 1
        elif bias == "BUY"  and svr < 40: score += 1

    # 20. Pre-market / after-hours gap direction
    pm       = data.get("premarket_data") or {}
    gap_bias = pm.get("gap_bias", "")
    if gap_bias and gap_bias != "neutral":
        total += 1
        if bias == "BUY"  and gap_bias == "bullish": score += 1
        elif bias == "SELL" and gap_bias == "bearish": score += 1

    return score, total

def confidence_bar(score: int, total: int) -> str:
    filled = "●" * score
    empty  = "○" * (total - score)
    return f"{filled}{empty}"

# ─────────────────────────────
# STREAK TRACKING
# ─────────────────────────────
def get_streak_label(state: dict, symbol: str, current_bias: str = "") -> str:
    """Show streak only when current signal extends it.

    Old version showed e.g. 'BUY streak: 3' even when current signal was SELL,
    which was misleading. Now we only show the streak when the current bias
    matches, and project the count to include the current signal.
    """
    streaks = state.get("__streaks__", {})
    s       = streaks.get(symbol, {})
    count   = s.get("count", 0)
    bias    = s.get("bias", "")
    if not current_bias or bias != current_bias:
        return ""
    new_count = count + 1
    if new_count < 2:
        return ""
    emoji = "✅" if current_bias == "BUY" else "🔴" if current_bias == "SELL" else "⚠️"
    return f"{emoji} {current_bias} streak: {new_count} signals in a row\n"

def update_streak(state: dict, symbol: str, bias: str):
    if "__streaks__" not in state:
        state["__streaks__"] = {}
    s = state["__streaks__"].get(symbol, {"bias": "", "count": 0})
    if s.get("bias") == bias:
        s["count"] += 1
    else:
        s = {"bias": bias, "count": 1}
    state["__streaks__"][symbol] = s

# ─────────────────────────────
# MULTI-SOURCE COUNTER
# ─────────────────────────────
def count_sources(title: str, story_counts: dict) -> int:
    """Return how many outlets have reported a similar story."""
    key = title.lower()[:150]
    for existing_key, count in story_counts.items():
        if SequenceMatcher(None, key, existing_key).ratio() >= 0.70:
            return count
    return 1

# ─────────────────────────────
# TRADINGVIEW CHART LINKS
# ─────────────────────────────
TRADINGVIEW_SYMBOLS = {
    "GC=F":  "COMEX:GC1!",
    "SI=F":  "COMEX:SI1!",
    "HG=F":  "COMEX:HG1!",
    "ALI=F": "COMEX:ALI1!",
    "CL=F":  "NYMEX:CL1!",
    "^GSPC": "SP:SPX",
    "QQQ":   "NASDAQ:QQQ",
}

def chart_url(symbol: str) -> str:
    tv = TRADINGVIEW_SYMBOLS.get(symbol, "")
    if not tv:
        return ""
    return f"https://www.tradingview.com/chart/?symbol={tv}"

# ─────────────────────────────
# PRICE CACHE — 15-min bars
# ─────────────────────────────
_price_cache: dict = {}
_fear_greed_cache = None

def fetch_symbol_data(symbol: str, ticker=None):
    """Return price + RSI + EMA + MACD + volume + S/R from 15-min bars."""
    try:
        df = (ticker or yf.Ticker(symbol)).history(period="5d", interval="15m", auto_adjust=True)
        if df.empty or len(df) < 2:
            return None

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        # Only filter zero-volume bars if the instrument actually has volume.
        # Indexes like ^VIX always report 0 volume and would be wiped entirely.
        if df["Volume"].sum() > 0:
            df = df[df["Volume"] > 0]
        if len(df) < 2:
            return None

        opens   = df["Open"].tolist()
        highs   = df["High"].tolist()
        lows    = df["Low"].tolist()
        closes  = df["Close"].tolist()
        volumes = df["Volume"].tolist()

        current    = float(closes[-1])
        # Use vectorised date comparison — faster than per-row lambda
        today_date = df.index[-1].date()
        prev_bars  = df[df.index.normalize().date < today_date]
        prev_close = float(prev_bars["Close"].iloc[-1]) if not prev_bars.empty else float(closes[0])

        move = round((current - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0.0

        rsi   = calculate_rsi(closes)
        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)
        trend = trend_label(current, ema20, ema50)

        macd_val, macd_sig, macd_hist = calculate_macd(closes)

        vol_ratio = None
        if len(volumes) >= 2:
            recent_vols = volumes[:-1][-20:]
            avg_vol     = sum(recent_vols) / len(recent_vols) if recent_vols else 0
            vol_ratio   = round(volumes[-1] / avg_vol, 2) if avg_vol > 0 else None

        support    = round(min(lows[-30:]),  4) if len(lows)   >= 2 else None
        resistance = round(max(highs[-30:]), 4) if len(highs)  >= 2 else None

        # RSI divergence over the last 20 bars — flags weakening momentum
        # before price actually reverses.
        divergence = detect_divergence(closes)
        atr        = calculate_atr(highs, lows, closes)
        fib        = detect_fibonacci(highs, lows, closes)
        ob         = detect_order_blocks(opens, highs, lows, closes)
        vwap       = calculate_vwap(highs, lows, closes, volumes)
        mkt_str    = detect_market_structure(highs, lows, closes)
        candle     = detect_candle_pattern(opens, highs, lows, closes)
        liq_sweep  = detect_liquidity_sweep(highs, lows, closes)
        order_flow = calculate_order_flow(opens, closes, volumes)

        return {
            "move":             move,
            "price":            round(current, 4),
            "rsi":              rsi,
            "ema20":            ema20,
            "ema50":            ema50,
            "trend":            trend,
            "macd":             macd_val,
            "macd_signal":      macd_sig,
            "macd_hist":        macd_hist,
            "vol_ratio":        vol_ratio,
            "support":          support,
            "resistance":       resistance,
            "divergence":       divergence,
            "atr":              atr,
            "fib":              fib,
            "ob":               ob,
            "vwap":             vwap,
            "market_structure": mkt_str,
            "candle_pattern":   candle,
            "liquidity_sweep":  liq_sweep,
            "order_flow":       order_flow,
        }
    except Exception as e:
        print(f"  Data fetch failed for {symbol}: {e}")
        return None

def fetch_daily_context(symbol: str, ticker=None) -> dict:
    """Return daily RSI + trend from 3-month daily bars (big-picture context)."""
    try:
        df = (ticker or yf.Ticker(symbol)).history(period="3mo", interval="1d", auto_adjust=True)
        if df.empty:
            return {}
        df     = df.dropna(subset=["Open", "High", "Low", "Close"])
        opens  = df["Open"].tolist()
        closes = df["Close"].tolist()
        highs  = df["High"].tolist()
        lows   = df["Low"].tolist()
        if not closes:
            return {}
        current       = closes[-1]
        d_ema20       = calculate_ema(closes, 20)
        d_ema50       = calculate_ema(closes, 50)
        d_rsi         = calculate_rsi(closes)
        d_trend       = trend_label(current, d_ema20, d_ema50)
        d_support     = round(min(lows[-30:]),  4) if len(lows)  >= 2 else None
        d_resistance  = round(max(highs[-30:]), 4) if len(highs) >= 2 else None
        volumes       = df["Volume"].tolist()
        return {
            "daily_trend":            d_trend,
            "daily_rsi":              d_rsi,
            "daily_support":          d_support,
            "daily_resistance":       d_resistance,
            "daily_fib":              detect_fibonacci(highs, lows, closes),
            "daily_market_structure": detect_market_structure(highs, lows, closes),
            "daily_order_flow":       calculate_order_flow(opens, closes, volumes, session_bars=20),
        }
    except Exception as e:
        print(f"  Daily context failed for {symbol}: {e}")
        return {}

def fetch_hourly_context(symbol: str, ticker=None) -> dict:
    """Return 1-hour timeframe trend + RSI for multi-timeframe confirmation.

    The 15-minute timeframe used elsewhere catches every short-term wiggle and
    fires false signals when the larger trend disagrees. Adding the 1h check
    lets us require both timeframes to agree before classifying a signal as
    high-confidence — eliminating most fakeouts.
    """
    try:
        df = (ticker or yf.Ticker(symbol)).history(period="1mo", interval="60m", auto_adjust=True)
        if df.empty:
            return {}
        df     = df.dropna(subset=["Close"])
        closes = df["Close"].tolist()
        if not closes:
            return {}
        current = closes[-1]
        h_ema20 = calculate_ema(closes, 20)
        h_ema50 = calculate_ema(closes, 50)
        h_rsi   = calculate_rsi(closes)
        h_trend = trend_label(current, h_ema20, h_ema50)
        return {
            "hourly_trend": h_trend,
            "hourly_rsi":   h_rsi,
        }
    except Exception as e:
        print(f"  Hourly context failed for {symbol}: {e}")
        return {}

def fetch_5m_context(symbol: str, ticker=None) -> dict:
    """Return 5-minute timeframe data: RSI, trend, S/R, FVG, Fib, and Order Blocks."""
    try:
        df = (ticker or yf.Ticker(symbol)).history(period="5d", interval="5m", auto_adjust=True)
        if df.empty:
            return {}
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if len(df) < 10:
            return {}
        opens   = df["Open"].tolist()
        highs   = df["High"].tolist()
        lows    = df["Low"].tolist()
        closes  = df["Close"].tolist()
        volumes = df["Volume"].tolist()
        current = closes[-1]
        ema20   = calculate_ema(closes, 20)
        ema50   = calculate_ema(closes, 50)
        return {
            "5m_rsi":              calculate_rsi(closes),
            "5m_trend":            trend_label(current, ema20, ema50),
            "5m_support":          round(min(lows[-30:]),  4) if len(lows)  >= 2 else None,
            "5m_resistance":       round(max(highs[-30:]), 4) if len(highs) >= 2 else None,
            "5m_fvg":              detect_fvg(highs, lows, closes),
            "5m_fib":              detect_fibonacci(highs, lows, closes),
            "5m_ob":               detect_order_blocks(opens, highs, lows, closes),
            "5m_market_structure": detect_market_structure(highs, lows, closes),
            "5m_candle_pattern":   detect_candle_pattern(opens, highs, lows, closes),
            "5m_liquidity_sweep":  detect_liquidity_sweep(highs, lows, closes),
            "5m_order_flow":       calculate_order_flow(opens, closes, volumes),
        }
    except Exception as e:
        print(f"  5m context failed for {symbol}: {e}")
        return {}

def fetch_4h_context(symbol: str, ticker=None) -> dict:
    """Return 4-hour timeframe data aggregated from 1h bars: RSI, trend, S/R, and FVG.

    Yahoo Finance has no native 4h interval, so we fetch 1h data over 3 months
    and group every 4 consecutive bars into one 4h bar.
    """
    try:
        df = (ticker or yf.Ticker(symbol)).history(period="3mo", interval="60m", auto_adjust=True)
        if df.empty:
            return {}
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if len(df) < 8:
            return {}

        h1_opens   = df["Open"].tolist()
        h1_highs   = df["High"].tolist()
        h1_lows    = df["Low"].tolist()
        h1_closes  = df["Close"].tolist()
        h1_volumes = df["Volume"].tolist()

        # Trim to multiple of 4 so grouping is clean
        rem = len(h1_closes) % 4
        if rem:
            h1_opens   = h1_opens[rem:]
            h1_highs   = h1_highs[rem:]
            h1_lows    = h1_lows[rem:]
            h1_closes  = h1_closes[rem:]
            h1_volumes = h1_volumes[rem:]

        n4      = len(h1_closes) // 4
        opens   = [h1_opens[i*4]                   for i in range(n4)]
        highs   = [max(h1_highs[i*4 : i*4+4])     for i in range(n4)]
        lows    = [min(h1_lows[i*4  : i*4+4])     for i in range(n4)]
        closes  = [h1_closes[i*4+3]                for i in range(n4)]
        volumes = [sum(h1_volumes[i*4 : i*4+4])   for i in range(n4)]

        if len(closes) < 10:
            return {}

        current = closes[-1]
        ema20   = calculate_ema(closes, 20)
        ema50   = calculate_ema(closes, 50)
        return {
            "4h_rsi":              calculate_rsi(closes),
            "4h_trend":            trend_label(current, ema20, ema50),
            "4h_support":          round(min(lows[-30:]),  4) if len(lows)  >= 2 else None,
            "4h_resistance":       round(max(highs[-30:]), 4) if len(highs) >= 2 else None,
            "4h_fvg":              detect_fvg(highs, lows, closes),
            "4h_fib":              detect_fibonacci(highs, lows, closes),
            "4h_ob":               detect_order_blocks(opens, highs, lows, closes),
            "4h_market_structure": detect_market_structure(highs, lows, closes),
            "4h_liquidity_sweep":  detect_liquidity_sweep(highs, lows, closes),
            "4h_order_flow":       calculate_order_flow(opens, closes, volumes, session_bars=30),
        }
    except Exception as e:
        print(f"  4h context failed for {symbol}: {e}")
        return {}

def detect_divergence(closes: list, period: int = 14, lookback: int = 20) -> str:
    """Detect bullish/bearish RSI divergence over the recent window.

    Bullish divergence: price made a lower low but RSI made a higher low →
    selling pressure weakening, possible reversal up.
    Bearish divergence: price made a higher high but RSI made a lower high →
    buying pressure weakening, possible reversal down.

    Returns "bullish", "bearish", or "" (no divergence / insufficient data).
    """
    if len(closes) < period + lookback + 2:
        return ""

    # Build RSI series
    rsi_series = []
    for end in range(period + 1, len(closes) + 1):
        sub = closes[:end]
        gains, losses = [], []
        for i in range(1, len(sub)):
            diff = sub[i] - sub[i - 1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(sub) - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_val = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100 - (100 / (1 + rs))
        rsi_series.append(rsi_val)

    # Compare last `lookback` bars
    if len(rsi_series) < lookback:
        return ""

    recent_closes = closes[-lookback:]
    recent_rsi    = rsi_series[-lookback:]

    # Find the two extreme points (newest vs older) in the window
    # Split the window in half to find a "previous" and "current" pivot
    half = lookback // 2
    older_lows_idx  = min(range(half), key=lambda i: recent_closes[i])
    newer_lows_idx  = half + min(range(lookback - half), key=lambda i: recent_closes[half + i])
    older_highs_idx = max(range(half), key=lambda i: recent_closes[i])
    newer_highs_idx = half + max(range(lookback - half), key=lambda i: recent_closes[half + i])

    # Bullish: lower low in price, higher low in RSI
    if (recent_closes[newer_lows_idx] < recent_closes[older_lows_idx] and
        recent_rsi[newer_lows_idx]    > recent_rsi[older_lows_idx] and
        recent_rsi[newer_lows_idx]    < 50):  # divergence in oversold zone is more meaningful
        return "bullish"

    # Bearish: higher high in price, lower high in RSI
    if (recent_closes[newer_highs_idx] > recent_closes[older_highs_idx] and
        recent_rsi[newer_highs_idx]    < recent_rsi[older_highs_idx] and
        recent_rsi[newer_highs_idx]    > 50):  # divergence in overbought zone is more meaningful
        return "bearish"

    return ""

def detect_fvg(highs: list, lows: list, closes: list, lookback: int = 60) -> dict:
    """Detect the most recent unfilled Fair Value Gap (imbalance zone).

    A FVG is a 3-candle pattern where price moved so fast it left a gap that
    wasn't traded through — the market tends to return and "fill" it later.

    Bullish FVG: candle[i-1].high < candle[i+1].low
      → price surged up, left a gap below (potential support / pullback magnet)
    Bearish FVG: candle[i-1].low > candle[i+1].high
      → price crashed down, left a gap above (potential resistance / rally magnet)

    Returns dict with type/low/high/status, or {} if none found.
    Status values:
      "open"   — price hasn't returned to the gap yet (most tradeable)
      "inside" — price is currently inside the gap (filling in progress)
      "filled" — price traded fully through the gap (skip)
    """
    n = len(closes)
    if n < 3 or len(highs) < 3 or len(lows) < 3:
        return {}

    current = closes[-1]
    limit   = max(1, n - lookback)

    for i in range(n - 2, limit - 1, -1):
        h_prev = highs[i - 1]
        l_prev = lows[i - 1]
        h_next = highs[i + 1]
        l_next = lows[i + 1]

        # Bullish FVG: gap between previous high and next candle's low
        if l_next > h_prev:
            gap_lo = round(h_prev, 6)
            gap_hi = round(l_next, 6)
            if gap_hi <= gap_lo:
                continue
            if current < gap_lo:
                status = "filled"
            elif current <= gap_hi:
                status = "inside"
            else:
                status = "open"
            if status != "filled":
                return {"type": "bullish", "low": gap_lo, "high": gap_hi, "status": status}

        # Bearish FVG: gap between next candle's high and previous low
        elif h_next < l_prev:
            gap_lo = round(h_next, 6)
            gap_hi = round(l_prev, 6)
            if gap_hi <= gap_lo:
                continue
            if current > gap_hi:
                status = "filled"
            elif current >= gap_lo:
                status = "inside"
            else:
                status = "open"
            if status != "filled":
                return {"type": "bearish", "low": gap_lo, "high": gap_hi, "status": status}

    return {}

def fvg_label(fvg: dict) -> str:
    """Human-readable FVG description for AI prompt and Telegram messages."""
    if not fvg:
        return ""
    t = fvg["type"].capitalize()
    notes = {
        "open":   "unfilled — price likely drawn back",
        "inside": "price currently inside gap (filling)",
    }
    note = notes.get(fvg["status"], fvg["status"])
    return f"{t} imbalance {fvg['low']} – {fvg['high']} ({note})"

def detect_fibonacci(highs: list, lows: list, closes: list, lookback: int = 100) -> dict:
    """Detect Fibonacci retracement levels from the most recent significant swing.

    Scans the last `lookback` bars for the highest high and lowest low.
    Whichever extreme is more recent defines the impulse direction:
      - High after low  → bullish impulse → retracement levels count DOWN from top
      - Low after high  → bearish impulse → retracement levels count UP from bottom

    Key levels returned: 0.236, 0.382, 0.5 (Equilibrium), 0.618, 0.786
    Extension targets:   1.272, 1.618 (for TP projection)

    Returns {} if data is insufficient or the swing is flat.
    """
    n = len(closes)
    if n < 10 or len(highs) < 10 or len(lows) < 10:
        return {}

    window   = min(n, lookback)
    h_w      = highs[-window:]
    l_w      = lows[-window:]
    sh_idx   = h_w.index(max(h_w))
    sl_idx   = l_w.index(min(l_w))
    sw_high  = max(h_w)
    sw_low   = min(l_w)
    diff     = sw_high - sw_low

    if diff <= 0:
        return {}

    current = closes[-1]

    if sh_idx > sl_idx:
        # Bullish impulse: low formed first, then price ran up
        # Retracement levels measured DOWN from swing high
        direction = "bullish"
        lvl = lambda pct: round(sw_high - diff * pct, 4)
        ext = lambda pct: round(sw_low  - diff * pct, 4)
    else:
        # Bearish impulse: high formed first, then price fell
        # Retracement levels measured UP from swing low
        direction = "bearish"
        lvl = lambda pct: round(sw_low  + diff * pct, 4)
        ext = lambda pct: round(sw_high + diff * pct, 4)

    levels = {
        "0.236": lvl(0.236),
        "0.382": lvl(0.382),
        "0.5":   lvl(0.500),  # Equilibrium
        "0.618": lvl(0.618),
        "0.786": lvl(0.786),
        "1.272": ext(0.272),  # extension target 1
        "1.618": ext(0.618),  # extension target 2
    }

    # Nearest standard retracement level to current price
    std_keys = ["0.236", "0.382", "0.5", "0.618", "0.786"]
    nearest  = min(std_keys, key=lambda k: abs(levels[k] - current))

    return {
        "direction":     direction,
        "swing_high":    round(sw_high, 4),
        "swing_low":     round(sw_low,  4),
        "equilibrium":   levels["0.5"],
        "fib_0236":      levels["0.236"],
        "fib_0382":      levels["0.382"],
        "fib_0618":      levels["0.618"],
        "fib_0786":      levels["0.786"],
        "fib_1272":      levels["1.272"],
        "fib_1618":      levels["1.618"],
        "nearest_level": nearest,
        "nearest_price": levels[nearest],
        "proximity_pct": round(abs(levels[nearest] - current) / current * 100, 2),
    }

def detect_order_blocks(opens: list, highs: list, lows: list, closes: list,
                        lookback: int = 60) -> dict:
    """Detect the most recent active bullish and bearish order blocks.

    An order block (OB) is the last opposing candle before a strong impulse move.
    Institutions leave unfilled orders there — price tends to return and react.

      Bullish OB: last bearish candle before a strong bullish impulse
        → zone = body of that candle, acts as support on a pullback
      Bearish OB: last bullish candle before a strong bearish impulse
        → zone = body of that candle, acts as resistance on a rally

    Status:
      "active"  — price hasn't returned to the zone (cleanest setup)
      "inside"  — price is retesting the zone right now (entry opportunity)
      "mitigated" — price traded through the zone (OB consumed, skip)

    Returns {"bullish": {...}, "bearish": {...}} with only active/inside blocks.
    """
    n = len(closes)
    if n < 6 or len(opens) < 6:
        return {}

    ranges    = [highs[i] - lows[i] for i in range(n)]
    avg_range = sum(ranges[max(0, n-30):]) / min(30, n)
    if avg_range == 0:
        return {}

    current = closes[-1]
    result  = {"bullish": None, "bearish": None}
    end     = max(1, n - lookback)

    for i in range(n - 2, end, -1):
        if result["bullish"] and result["bearish"]:
            break
        if i + 1 >= n:
            continue

        # ── Bullish OB ────────────────────────────────────────────────────────
        # Bearish candle[i] → bullish impulse at candle[i+1]
        if (result["bullish"] is None
                and closes[i] < opens[i]             # candle i is bearish
                and closes[i+1] > opens[i+1]         # candle i+1 is bullish
                and (highs[i+1] - lows[i+1]) >= avg_range * 1.2):  # impulse

            ob_lo = min(opens[i], closes[i])
            ob_hi = max(opens[i], closes[i])
            if ob_hi > ob_lo and current > ob_lo:   # price still above OB
                status = "inside" if current <= ob_hi else "active"
                result["bullish"] = {
                    "low": round(ob_lo, 6), "high": round(ob_hi, 6),
                    "status": status,
                }

        # ── Bearish OB ────────────────────────────────────────────────────────
        # Bullish candle[i] → bearish impulse at candle[i+1]
        if (result["bearish"] is None
                and closes[i] > opens[i]             # candle i is bullish
                and closes[i+1] < opens[i+1]         # candle i+1 is bearish
                and (highs[i+1] - lows[i+1]) >= avg_range * 1.2):  # impulse

            ob_lo = min(opens[i], closes[i])
            ob_hi = max(opens[i], closes[i])
            if ob_hi > ob_lo and current < ob_hi:   # price still below OB
                status = "inside" if current >= ob_lo else "active"
                result["bearish"] = {
                    "low": round(ob_lo, 6), "high": round(ob_hi, 6),
                    "status": status,
                }

    return {k: v for k, v in result.items() if v is not None}

def calculate_vwap(highs: list, lows: list, closes: list, volumes: list,
                   session_bars: int = 40) -> dict:
    """Calculate VWAP over the most recent session window.

    VWAP = Σ(typical_price × volume) / Σ(volume)
    Typical price = (high + low + close) / 3

    Uses last `session_bars` bars as a session approximation (≈40 bars covers
    a full US equities session on 15m, or several forex hours). Also returns
    ±1σ and ±2σ bands so the AI can identify overextension.
    """
    n = min(len(closes), session_bars)
    if n < 2:
        return {}

    cum_vol   = 0.0
    cum_tpvol = 0.0
    cum_tp2vol = 0.0

    for h, l, c, v in zip(highs[-n:], lows[-n:], closes[-n:], volumes[-n:]):
        if v <= 0:
            continue
        tp = (h + l + c) / 3.0
        cum_vol   += v
        cum_tpvol += tp * v
        cum_tp2vol += tp * tp * v

    if cum_vol == 0:
        return {}

    vwap     = cum_tpvol / cum_vol
    variance = max(0.0, (cum_tp2vol / cum_vol) - vwap ** 2)
    std      = variance ** 0.5

    current = closes[-1]
    if vwap > 0:
        dist_pct = round((current - vwap) / vwap * 100, 2)
        if dist_pct > 0:
            position = f"above VWAP (+{dist_pct}%)"
        elif dist_pct < 0:
            position = f"below VWAP ({dist_pct}%)"
        else:
            position = "at VWAP"
    else:
        position = "n/a"

    return {
        "vwap":      round(vwap, 4),
        "vwap_u1":   round(vwap + std,     4),
        "vwap_l1":   round(vwap - std,     4),
        "vwap_u2":   round(vwap + 2 * std, 4),
        "vwap_l2":   round(vwap - 2 * std, 4),
        "position":  position,  # human-readable: "above VWAP (+0.3%)"
    }

def calculate_order_flow(opens: list, closes: list, volumes: list,
                         session_bars: int = 40) -> dict:
    """Estimate order flow delta from OHLCV bars (Bookmap-style approximation).

    Since Yahoo Finance has no L2 bid/ask data, we use the standard retail
    approximation:
      Bullish bar (close > open) → all volume = buy pressure
      Bearish bar (close < open) → all volume = sell pressure
      Doji (close == open)       → neutral (0)

    Delta     = buy_vol − sell_vol for the last bar
    Cum delta = Σ delta over the last session_bars window

    Delta flip: the last 3-bar block changed sign vs the prior 3-bar block →
    buyers overwhelmed sellers (or vice versa) = potential reversal setup.
    """
    n = min(len(closes), session_bars)
    if n < 3:
        return {}

    o_w = opens[-n:]
    c_w = closes[-n:]
    v_w = volumes[-n:]

    deltas = []
    for o, c, v in zip(o_w, c_w, v_w):
        if c > o:
            deltas.append(v)
        elif c < o:
            deltas.append(-v)
        else:
            deltas.append(0)

    cum_delta  = sum(deltas)
    last_delta = deltas[-1] if deltas else 0

    # Delta flip: last 3 bars vs prior 3 bars
    flip = ""
    if len(deltas) >= 6:
        recent = sum(deltas[-3:])
        prior  = sum(deltas[-6:-3])
        if prior < 0 and recent > 0:
            flip = "Bullish delta flip — sellers exhausted, buyers taking over"
        elif prior > 0 and recent < 0:
            flip = "Bearish delta flip — buyers exhausted, sellers taking over"

    if cum_delta > 0:
        bias = "bullish"
    elif cum_delta < 0:
        bias = "bearish"
    else:
        bias = "neutral"

    return {
        "delta":      round(last_delta),
        "cum_delta":  round(cum_delta),
        "delta_flip": flip,
        "of_bias":    bias,
    }

# ─────────────────────────────
# POLYGON.IO — REAL ORDER FLOW
# Uses 1-minute aggregate bars from Polygon (available on current plan).
# Calculates real delta, buy %, POC, VWAP, volume spikes, and delta flips
# from bar-direction logic (close > open = buy bar, close < open = sell bar).
#
# Coverage:
#   SPY (^GSPC proxy) and QQQ  → real Polygon 1-min bars ✅
#   GC=F, CL=F, ALI=F (futures) → not on this Polygon plan, yfinance fallback
#
# Falls back gracefully to {} on any error or missing POLYGON_KEY.
# ─────────────────────────────

def _massive_get_aggs(ticker: str, minutes: int = 300) -> list:
    """
    Fetch the last `minutes` worth of 1-minute aggregate bars from api.massive.com.
    Returns list of bar dicts (v, vw, o, c, h, l, t, n) or [] on failure.
    Each bar: v=volume, vw=VWAP, o=open, c=close, h=high, l=low, t=timestamp_ms, n=trades

    Confirmed working tickers on Massive (Futures Developer + Stocks Basic plans):
      CL  → Crude Oil futures (NYMEX, real futures)       ✅
      SPY → S&P 500 ETF proxy                             ✅
      QQQ → Nasdaq 100 ETF                                ✅
      GLD → SPDR Gold ETF (tracks gold 1:1, highly liquid) ✅  ← GC futures not in plan
      DBB → Invesco DB Base Metals ETF (33% aluminium)    ✅  ← ALI futures not in plan
    Also includes DELAYED status (equivalent to real data for our use).
    """
    try:
        from datetime import date
        today     = date.today().isoformat()
        from_date = (date.today() - timedelta(days=3)).isoformat()  # covers weekends
        url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{today}"
        params = {
            "adjusted": "true",
            "sort":     "asc",
            "limit":    minutes,
            "apiKey":   POLYGON_KEY,
        }
        r = requests.get(url, params=params, timeout=12)
        if r.status_code == 200:
            data = r.json()
            # Accept both OK and DELAYED — DELAYED just means 15-min lag, data is real
            if data.get("status") in ("OK", "DELAYED"):
                results = data.get("results", [])
                if results:
                    return results
    except Exception as e:
        print(f"  Massive aggs error ({ticker}): {e}")
    return []


def _calc_rich_flow(bars: list, proxy_note: str = "") -> dict:
    """
    Comprehensive order flow analysis from Polygon 1-minute OHLCV bars.

    Bar classification (close-tick rule):
      close > open  → market buy pressure   (+volume)
      close < open  → market sell pressure  (-volume)
      close == open → neutral (split 50/50)

    Computed metrics (all from real Polygon data):
      DELTA      : cum_delta, buy_pct, of_bias, per-bar delta
      VWAP       : vwap price, price position vs VWAP, % distance
      VOLUME PROFILE: vpoc (most-traded price), value area high/low (VAH/VAL, 70% vol)
      EVENTS     : delta_flip, exhaustion, absorption, imbalance
      STRUCTURE  : bar_streak, market structure (HH/HL or LH/LL), swing S/R levels
      VOLUME     : vol_spike (last bar vs session average)
    """
    if len(bars) < 10:
        return {}

    # ── Per-bar pass ──────────────────────────────────────────────────────────
    buy_vol = sell_vol = 0.0
    deltas: list = []
    price_vol: dict = {}          # price → cumulative volume (volume profile)
    total_vol_all = 0.0
    vols = [b.get("v") or 0 for b in bars]
    avg_vol = sum(vols) / len(vols) if vols else 1.0

    bar_data: list = []

    for b in bars:
        v  = b.get("v") or 0
        o  = b.get("o") or 0
        c  = b.get("c") or 0
        h  = b.get("h") or c
        l  = b.get("l") or c
        vw = b.get("vw") or c

        total_vol_all += v

        # Volume profile: bucket by typical price (H+L+C)/3 rounded to 2dp
        tp = round((h + l + c) / 3, 2)
        price_vol[tp] = price_vol.get(tp, 0) + v

        if c > o:
            bar_delta = v
            buy_vol  += v
        elif c < o:
            bar_delta = -v
            sell_vol += v
        else:
            bar_delta = 0
            buy_vol  += v / 2
            sell_vol += v / 2

        deltas.append(bar_delta)
        bar_data.append({"v": v, "o": o, "c": c, "h": h, "l": l,
                         "vw": vw, "delta": bar_delta, "range": h - l})

    total_vol = buy_vol + sell_vol
    cum_delta = round(buy_vol - sell_vol)
    buy_pct   = round(buy_vol / total_vol * 100) if total_vol > 0 else 50
    bias      = "bullish" if cum_delta > 0 else "bearish" if cum_delta < 0 else "neutral"

    # ── Session-anchored VWAP (∑(typical_price × volume) / ∑volume) ─────────
    # More accurate than the per-bar 'vw' field because it anchors to bar[0].
    # Falls back to Polygon's per-bar vw if volume data is missing.
    last_bar   = bar_data[-1]
    current_px = last_bar.get("c") or 0

    tpv_sum = sum((b["h"] + b["l"] + b["c"]) / 3 * b["v"] for b in bar_data if b["v"])
    v_sum   = sum(b["v"] for b in bar_data if b["v"])
    if v_sum > 0 and tpv_sum > 0:
        vwap_val = round(tpv_sum / v_sum, 4)
    else:
        vwap_val = last_bar.get("vw")   # fallback

    price_vs_vwap = "n/a"
    vwap_dist_pct = 0.0
    if vwap_val and current_px and vwap_val > 0:
        vwap_dist_pct = round((current_px - vwap_val) / vwap_val * 100, 2)
        if vwap_dist_pct > 0.05:
            price_vs_vwap = f"above (+{vwap_dist_pct}%)"
        elif vwap_dist_pct < -0.05:
            price_vs_vwap = f"below ({vwap_dist_pct}%)"
        else:
            price_vs_vwap = "at VWAP"

    # ── Volume Profile: VPOC + Value Area (70% of volume) ────────────────────
    vpoc = max(price_vol, key=price_vol.get) if price_vol else None
    vah = val = None
    if price_vol and total_vol_all > 0:
        target = total_vol_all * 0.70
        accumulated = 0.0
        va_prices: list = []
        for px, vol in sorted(price_vol.items(), key=lambda x: x[1], reverse=True):
            accumulated += vol
            va_prices.append(px)
            if accumulated >= target:
                break
        if va_prices:
            vah = round(max(va_prices), 4)
            val = round(min(va_prices), 4)

    # ── Delta flip ────────────────────────────────────────────────────────────
    delta_flip = ""
    if len(deltas) >= 6:
        recent_d = sum(deltas[-3:])
        prior_d  = sum(deltas[-6:-3])
        if prior_d < 0 and recent_d > 0:
            delta_flip = "Bullish delta flip — sellers exhausted, buyers stepping in"
        elif prior_d > 0 and recent_d < 0:
            delta_flip = "Bearish delta flip — buyers exhausted, sellers pressing"

    # ── Exhaustion ────────────────────────────────────────────────────────────
    # Trend bars losing volume + delta → one side running out of orders
    exhaustion = ""
    if len(bar_data) >= 8:
        first4 = bar_data[-8:-4]
        last4  = bar_data[-4:]
        f_vol  = sum(b["v"]     for b in first4)
        l_vol  = sum(b["v"]     for b in last4)
        f_dlt  = sum(b["delta"] for b in first4)
        l_dlt  = sum(b["delta"] for b in last4)
        if f_vol > 0 and l_vol < f_vol * 0.6:      # volume fading ≥40%
            if f_dlt > 0 and l_dlt < f_dlt * 0.4:
                exhaustion = "Buying exhaustion — volume and buy delta both fading"
            elif f_dlt < 0 and l_dlt > f_dlt * 0.4:
                exhaustion = "Selling exhaustion — volume and sell delta both fading"

    # ── Absorption ────────────────────────────────────────────────────────────
    # High volume + tiny price range → opposing side absorbing all orders
    absorption = ""
    if len(bar_data) >= 3:
        last3     = bar_data[-3:]
        avg_rng3  = sum(b["range"] for b in last3) / 3
        avg_vol3  = sum(b["v"]     for b in last3) / 3
        ref_price = last3[-1]["c"] or 1
        if avg_vol3 > avg_vol * 1.8 and avg_rng3 < ref_price * 0.0015:
            net3 = sum(b["delta"] for b in last3)
            if net3 > 0:
                absorption = "Seller absorption — buyers driving hard but price contained; sellers absorbing supply"
            else:
                absorption = "Buyer absorption — sellers pressing but price not dropping; buyers absorbing selling"

    # ── Imbalance ─────────────────────────────────────────────────────────────
    # Consecutive bars dominated by one side
    imbalance = ""
    if len(deltas) >= 10:
        last10 = deltas[-10:]
        buys   = sum(1 for d in last10 if d > 0)
        sells  = sum(1 for d in last10 if d < 0)
        if buys >= 8:
            imbalance = f"Strong buy imbalance — {buys}/10 recent bars bullish"
        elif sells >= 8:
            imbalance = f"Strong sell imbalance — {sells}/10 recent bars bearish"
        elif buys >= 7:
            imbalance = f"Moderate buy imbalance — {buys}/10 recent bars bullish"
        elif sells >= 7:
            imbalance = f"Moderate sell imbalance — {sells}/10 recent bars bearish"

    # ── Bar streak ────────────────────────────────────────────────────────────
    streak = 0
    streak_dir = None
    for d in reversed(deltas):
        if d > 0:
            if streak_dir in (None, "buy"):
                streak += 1; streak_dir = "buy"
            else: break
        elif d < 0:
            if streak_dir in (None, "sell"):
                streak += 1; streak_dir = "sell"
            else: break
        else:
            break

    bar_streak = ""
    if streak >= 3 and streak_dir:
        bar_streak = f"{streak} consecutive {'buying' if streak_dir == 'buy' else 'selling'} bars"

    # ── Volume spike ──────────────────────────────────────────────────────────
    last_vol  = bar_data[-1]["v"]
    vol_spike = round(last_vol / avg_vol, 1) if avg_vol > 0 else 1.0

    # ── Block trade detection (using Polygon 'n' field = trades per bar) ─────
    # A block trade bar has very few large individual transactions:
    # high volume but low trade count → average trade size is abnormally large.
    # Threshold: avg_trade_size >= 5× the session average trade size.
    block_trades: list = []
    n_values = [b.get("n") or 0 for b in bars if b.get("n")]
    if n_values:
        avg_n    = sum(n_values) / len(n_values)
        avg_size = (total_vol_all / sum(n_values)) if sum(n_values) > 0 else 0
        for b in bars[-20:]:   # scan last 20 bars for recency
            bv = b.get("v") or 0
            bn = b.get("n") or 0
            if bn > 0 and bv > 0:
                bar_avg_size = bv / bn
                # Block trade: high volume + very few individual transactions
                # avg_size * 5 = abnormally large trade size; bn < avg_n * 0.5 = unusually few trades
                if bar_avg_size >= avg_size * 5 and bv >= avg_vol * 1.5 and bn < avg_n * 0.5:
                    direction = "buy" if (b.get("c", 0) >= b.get("o", 0)) else "sell"
                    block_trades.append(direction)
    block_trade_signal = ""
    if block_trades:
        buy_blk  = block_trades.count("buy")
        sell_blk = block_trades.count("sell")
        if buy_blk > sell_blk:
            block_trade_signal = f"🐳 Block buy prints detected ({buy_blk} bars) — institutional accumulation"
        elif sell_blk > buy_blk:
            block_trade_signal = f"🐳 Block sell prints detected ({sell_blk} bars) — institutional distribution"
        else:
            block_trade_signal = f"🐳 Block prints detected ({len(block_trades)} bars) — large player activity"

    # ── Session high / low ────────────────────────────────────────────────────
    session_high = round(max(b["h"] for b in bar_data), 4)
    session_low  = round(min(b["l"] for b in bar_data), 4)

    # ── Opening range (first 15 bars = first 15 minutes) ─────────────────────
    # Shows where price opened and whether it has broken out of the early range.
    # Meaningful for equities (RTH open) and highly liquid futures.
    or_high = or_low = None
    or_status = ""
    if len(bar_data) >= 15:
        or_bars   = bar_data[:15]
        or_high   = round(max(b["h"] for b in or_bars), 4)
        or_low    = round(min(b["l"] for b in or_bars), 4)
        if current_px > or_high:
            or_status = f"above OR high ({or_low}–{or_high}) ↑ bullish breakout"
        elif current_px < or_low:
            or_status = f"below OR low ({or_low}–{or_high}) ↓ bearish breakout"
        else:
            or_status = f"inside OR ({or_low}–{or_high}) — no breakout yet"

    # ── Market structure from swing highs/lows (last 30 bars) ────────────────
    window    = bar_data[-30:] if len(bar_data) >= 30 else bar_data
    swg_highs: list = []
    swg_lows:  list = []
    for i in range(1, len(window) - 1):
        if window[i]["h"] > window[i-1]["h"] and window[i]["h"] > window[i+1]["h"]:
            swg_highs.append(round(window[i]["h"], 4))
        if window[i]["l"] < window[i-1]["l"] and window[i]["l"] < window[i+1]["l"]:
            swg_lows.append(round(window[i]["l"], 4))

    mkt_structure = "ranging"
    if len(swg_highs) >= 2 and len(swg_lows) >= 2:
        if swg_highs[-1] > swg_highs[-2] and swg_lows[-1] > swg_lows[-2]:
            mkt_structure = "bullish (HH + HL)"
        elif swg_highs[-1] < swg_highs[-2] and swg_lows[-1] < swg_lows[-2]:
            mkt_structure = "bearish (LH + LL)"

    key_resistance = round(max(swg_highs[-3:]), 4) if swg_highs else None
    key_support    = round(min(swg_lows[-3:]),  4) if swg_lows  else None

    return {
        # Core delta
        "cum_delta":       cum_delta,
        "buy_pct":         buy_pct,
        "of_bias":         bias,
        # VWAP
        "vwap":            round(vwap_val, 4) if vwap_val else None,
        "price_vs_vwap":   price_vs_vwap,
        "vwap_dist_pct":   vwap_dist_pct,
        # Volume profile
        "vpoc":            vpoc,
        "vah":             vah,
        "val":             val,
        # Events
        "delta_flip":      delta_flip,
        "exhaustion":      exhaustion,
        "absorption":      absorption,
        "imbalance":       imbalance,
        "bar_streak":      bar_streak,
        "vol_spike":       vol_spike,
        "block_trade":     block_trade_signal,
        # Session levels
        "session_high":    session_high,
        "session_low":     session_low,
        "or_high":         or_high,
        "or_low":          or_low,
        "or_status":       or_status,
        # Structure
        "mkt_structure":   mkt_structure,
        "key_resistance":  key_resistance,
        "key_support":     key_support,
        # Meta
        "data_source":     "polygon_real",
        "bar_count":       len(bars),
        "proxy_note":      proxy_note,
    }


def fetch_polygon_order_flow(symbol: str) -> dict:
    """
    Fetch comprehensive real order flow from api.massive.com (1-min bars).

    Ticker mapping (all confirmed working on current Massive plan):
      ^GSPC → SPY  ✅  S&P 500 ETF (direct proxy)
      QQQ   → QQQ  ✅  Nasdaq 100 ETF (direct)
      CL=F  → CL   ✅  Crude Oil futures — real NYMEX continuous contract
      GC=F  → GLD  ✅  Gold: SPDR Gold ETF (tracks gold price 1:1, highly liquid)
                        GC continuous futures not in plan; GLD is the best proxy
      ALI=F → DBB  ✅  Aluminium: Invesco DB Base Metals ETF (33% aluminium,
                        33% copper, 33% zinc) — ALI futures not in plan
    Falls back gracefully to {} on any failure.
    """
    if not POLYGON_KEY:
        return {}
    poly_sym = POLYGON_SYMBOLS.get(symbol)
    if not poly_sym:
        return {}

    proxy_notes = {
        "GC=F":  "via GLD (SPDR Gold ETF — tracks gold 1:1)",
        "ALI=F": "via DBB (Base Metals ETF — 33% aluminium)",
    }
    proxy_note = proxy_notes.get(symbol, "")

    # SPY/QQQ: use full trading session window (390 min = 6.5 h) for richer
    # volume profile and market structure vs. 300 min for futures.
    bar_minutes = 390 if symbol in ("^GSPC", "QQQ") else 300

    try:
        bars = _massive_get_aggs(poly_sym, minutes=bar_minutes)
        if not bars:
            print(f"  Massive: no bars returned for {poly_sym} ({symbol})")
            return {}
        print(f"  Massive: {len(bars)} bars for {poly_sym} ({symbol}){' ' + proxy_note if proxy_note else ''}")
        return _calc_rich_flow(bars, proxy_note=proxy_note)
    except Exception as e:
        print(f"  Massive order flow error ({symbol}): {e}")
        return {}

def detect_market_structure(highs: list, lows: list, closes: list,
                            lookback: int = 60) -> dict:
    """Detect market structure: trend bias, BOS, or CHoCH.

    Scans recent swing highs/lows to classify the market:
      HH + HL → Bullish structure
      LH + LL → Bearish structure
      Mixed   → Ranging

    BOS  (Break of Structure): current price broke a prior swing in the
         direction of the existing trend → trend continuation.
    CHoCH (Change of Character): current price broke a prior swing AGAINST
         the existing trend → potential reversal, high-probability setup.
    """
    n = len(closes)
    if n < 10:
        return {}

    window = min(n, lookback)
    h_w = highs[-window:]
    l_w = lows[-window:]

    swing_highs, swing_lows = [], []
    for i in range(1, len(h_w) - 1):
        if h_w[i] > h_w[i - 1] and h_w[i] > h_w[i + 1]:
            swing_highs.append(h_w[i])
        if l_w[i] < l_w[i - 1] and l_w[i] < l_w[i + 1]:
            swing_lows.append(l_w[i])

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {}

    sh1, sh2 = swing_highs[-1], swing_highs[-2]
    sl1, sl2 = swing_lows[-1],  swing_lows[-2]

    if sh1 > sh2 and sl1 > sl2:
        structure, bias = "Bullish (HH + HL)", "bullish"
    elif sh1 < sh2 and sl1 < sl2:
        structure, bias = "Bearish (LH + LL)", "bearish"
    elif sh1 > sh2:
        structure, bias = "Ranging (HH + LL — expanding)", "neutral"
    else:
        structure, bias = "Ranging (LH + HL — contracting)", "neutral"

    current = closes[-1]
    event = ""
    if current > sh1:
        event = "BOS Bullish (trend continuation)" if bias == "bullish" \
                else "CHoCH Bullish ⚡ (reversal signal — structure flipped)"
    elif current < sl1:
        event = "BOS Bearish (trend continuation)" if bias == "bearish" \
                else "CHoCH Bearish ⚡ (reversal signal — structure flipped)"

    return {
        "structure":  structure,
        "bias":       bias,
        "swing_high": round(sh1, 4),
        "swing_low":  round(sl1, 4),
        "event":      event,
    }

def detect_candle_pattern(opens: list, highs: list, lows: list,
                          closes: list) -> str:
    """Detect the most recent significant candlestick pattern.

    Checks the last 2 candles for:
      Pin Bar     — long wick rejection (hammer / shooting star)
      Engulfing   — body fully engulfs prior candle (continuation or reversal)
      Doji        — tiny body, indecision / exhaustion
    Returns a plain-English description, or "" if no pattern.
    """
    if len(closes) < 2 or len(opens) < 2:
        return ""
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    po, _ph, _pl, pc = opens[-2], highs[-2], lows[-2], closes[-2]

    total   = h - l
    if total <= 0:
        return ""
    body    = abs(c - o)
    body_hi = max(o, c)
    body_lo = min(o, c)
    upper   = h - body_hi
    lower   = body_lo - l

    # Pin bars first — a hammer/shooting star naturally has a tiny body, so
    # checking Doji first would misclassify them. Specific beats generic.
    # Bullish pin bar (hammer): lower wick ≥ 2× body, lower wick ≥ 2× upper wick
    if lower >= body * 2.0 and lower >= upper * 2.0 and lower >= total * 0.5:
        return "Bullish Pin Bar (hammer) — strong rejection of lows, potential reversal up"

    # Bearish pin bar (shooting star): upper wick ≥ 2× body, upper wick ≥ 2× lower wick
    if upper >= body * 2.0 and upper >= lower * 2.0 and upper >= total * 0.5:
        return "Bearish Pin Bar (shooting star) — strong rejection of highs, potential reversal down"

    # Doji — body is < 10% of range (only after ruling out pin bars above)
    if body / total < 0.10:
        return "Doji — indecision / possible exhaustion"

    # Engulfing — current body fully engulfs prior body
    prev_body = abs(pc - po)
    if prev_body > 0 and body > prev_body and body_lo <= min(po, pc) and body_hi >= max(po, pc):
        if c > o and pc > po:
            return "Bullish Engulfing — buyers overwhelmed prior sellers"
        if c < o and pc < po:
            return "Bearish Engulfing — sellers overwhelmed prior buyers"

    return ""

def detect_liquidity_sweep(highs: list, lows: list, closes: list,
                           lookback: int = 30) -> str:
    """Detect a recent liquidity sweep (stop hunt) on the last closed candle.

    A sweep occurs when price briefly exceeds a prior key level (triggering
    stop-losses) but then closes back inside — signalling a reversal.

    Bullish sweep: wicked below the recent swing low then closed above it
      → sells were triggered, now buyers absorb → long setup
    Bearish sweep: wicked above the recent swing high then closed below it
      → buys were triggered, now sellers absorb → short setup
    """
    n = len(closes)
    if n < lookback + 2:
        return ""

    prior_h = highs[-(lookback + 1):-1]
    prior_l = lows[-(lookback + 1):-1]
    if not prior_h or not prior_l:
        return ""

    sw_high = max(prior_h)
    sw_low  = min(prior_l)

    last_h, last_l, last_c = highs[-1], lows[-1], closes[-1]

    if last_h > sw_high and last_c < sw_high:
        return (f"Bearish sweep of highs at {round(sw_high, 4)} — "
                f"buy-side liquidity grabbed, potential reversal down ⚡")

    if last_l < sw_low and last_c > sw_low:
        return (f"Bullish sweep of lows at {round(sw_low, 4)} — "
                f"sell-side liquidity grabbed, potential reversal up ⚡")

    return ""

def premium_discount_label(price: float, fib: dict) -> str:
    """Classify current price as Premium, Discount, or at Equilibrium.

    Smart Money Concepts framework:
      Above EQ (0.5 fib) = PREMIUM — institutions prefer to sell here
      Below EQ (0.5 fib) = DISCOUNT — institutions prefer to buy here
      Near EQ            = EQUILIBRIUM — decision zone, wait for confirmation
    """
    if not fib or not price:
        return ""
    eq = fib.get("equilibrium")
    if not eq or eq <= 0:
        return ""
    pct = (price - eq) / eq * 100
    if pct > 1.0:
        return f"PREMIUM (+{pct:.2f}% above EQ {eq}) — SM sells, look for shorts"
    if pct < -1.0:
        return f"DISCOUNT ({pct:.2f}% below EQ {eq}) — SM buys, look for longs"
    return f"EQUILIBRIUM ({pct:+.2f}% from EQ {eq}) — fair value zone, wait for confirmation"

def fetch_fear_greed():
    """Fetch Crypto Fear & Greed Index from alternative.me (free, no key)."""
    global _fear_greed_cache
    if _fear_greed_cache is not None:
        return _fear_greed_cache
    try:
        r    = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=6,
        ).json()
        data = r["data"][0]
        _fear_greed_cache = {
            "value": int(data["value"]),
            "label": data["value_classification"],
        }
        return _fear_greed_cache
    except Exception as e:
        print(f"  Fear & Greed fetch failed: {e}")
        return None

def fg_emoji(value: int) -> str:
    if value <= 20:  return "😱"
    if value <= 40:  return "😨"
    if value <= 60:  return "😐"
    if value <= 80:  return "😀"
    return "🤑"

def vix_label(vix: float) -> str:
    """Human-readable VIX reading with emoji."""
    if vix >= 35:  return f"😱 {vix:.1f} — Extreme Fear (markets panicking)"
    if vix >= 25:  return f"😨 {vix:.1f} — Elevated Fear (high volatility)"
    if vix >= 18:  return f"😐 {vix:.1f} — Neutral"
    if vix >= 13:  return f"😀 {vix:.1f} — Low Fear (calm markets)"
    return             f"🤑 {vix:.1f} — Extreme Complacency (danger zone)"

def vix_signal_note(vix: float, bias: str) -> str:
    """Return a short contrarian note when VIX is at an extreme and agrees with bias."""
    if vix >= 30 and bias == "BUY":
        return " — extreme fear often marks bottoms ✅"
    if vix <= 13 and bias == "SELL":
        return " — extreme complacency often precedes drops ✅"
    if vix >= 30 and bias == "SELL":
        return " — selling into panic, watch for reversal ⚠️"
    if vix <= 13 and bias == "BUY":
        return " — buying into complacency, tighter stops advised ⚠️"
    return ""

def refresh_price_cache():
    global _price_cache
    _price_cache = {}

    symbols = list(ASSET_MAP.keys()) + list(CONTEXT_SYMBOLS.keys())

    for symbol in symbols:
        # Create ONE Ticker object per symbol — reused across all timeframe
        # fetches so yfinance only sets up the session/crumb once per symbol
        # instead of 5 times. Same data, ~5x fewer HTTP handshakes.
        ticker = yf.Ticker(symbol)
        data = fetch_symbol_data(symbol, ticker=ticker)
        if data is not None:
            if symbol in ASSET_MAP:
                data.update(fetch_5m_context(symbol,    ticker=ticker))
                data.update(fetch_daily_context(symbol, ticker=ticker))
                data.update(fetch_hourly_context(symbol,ticker=ticker))
                data.update(fetch_4h_context(symbol,    ticker=ticker))

                # Polygon real order flow — overrides yfinance OHLC approximation
                # when available. Falls back silently if key missing or API fails.
                # 13-second pause between calls keeps us under the ~5 req/min
                # rate limit so all 7 assets get real data every run.
                if POLYGON_KEY and symbol in POLYGON_SYMBOLS:
                    poly_flow = fetch_polygon_order_flow(symbol)
                    if poly_flow:
                        data["order_flow"]   = poly_flow   # replaces yfinance approx
                        data["polygon_flow"] = poly_flow   # also store separately
                    time.sleep(13)   # rate limit: ~5 req/min → 1 per 13s = safe

            _price_cache[symbol] = data
        time.sleep(0.3)

    summary = {s: f"{d['move']:+.2f}% @ {d['price']}" for s, d in _price_cache.items()}
    print(f"  Prices: {summary or 'all unavailable'}")

def get_cached_moves(title: str) -> dict:
    title_lower = title.lower()

    # Macro events affect all tradeable assets
    if is_macro(title_lower):
        return {s: d for s, d in _price_cache.items() if s in ASSET_MAP}

    # Try to match a specific asset from the headline keywords
    for sym, keywords in ASSET_MAP.items():
        if any(k in title_lower for k in keywords):
            return {sym: _price_cache[sym]} if sym in _price_cache else {}

    # Fallback — S&P 500 as general market proxy
    return {"^GSPC": _price_cache["^GSPC"]} if "^GSPC" in _price_cache else {}

# ─────────────────────────────
# HEADLINE AGE FILTER
# ─────────────────────────────
def is_fresh(entry) -> bool:
    try:
        raw = entry.get("published") or entry.get("updated")
        if not raw:
            return True
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (now_utc() - dt) <= timedelta(minutes=MAX_HEADLINE_AGE_MIN)
    except Exception:
        return True

# ─────────────────────────────
# ECONOMIC CALENDAR
# ─────────────────────────────
_calendar_cache = None

def fetch_calendar() -> list:
    global _calendar_cache
    if _calendar_cache is not None:
        return _calendar_cache
    try:
        url    = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r      = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8).json()
        today  = now_utc().strftime("%Y-%m-%d")
        events = []
        for e in r:
            if e.get("impact") == "High" and e.get("date", "").startswith(today):
                events.append({
                    "title":    e.get("title", ""),
                    "currency": e.get("currency", ""),
                    "time":     e.get("date", "")[-8:-3] + " UTC",
                })
        _calendar_cache = events
        if events:
            print(f"  Calendar: {len(events)} high-impact events today")
        return events
    except Exception as e:
        print(f"  Calendar fetch failed: {e}")
        _calendar_cache = []
        return []

# ─────────────────────────────
# DUPLICATE STORY FILTER
# ─────────────────────────────
def is_duplicate(title: str, recent: list) -> bool:
    t = title.lower()[:200]
    for s in recent[-50:]:
        if SequenceMatcher(None, t, s.lower()[:200]).ratio() >= SIMILARITY_THRESHOLD:
            return True
    return False

# ─────────────────────────────
# STATE
# ─────────────────────────────
def load_state() -> dict:
    try:
        with open(SEEN_FILE) as f:
            data = json.load(f)
        if isinstance(data, list):
            return {
                "seen_headlines":       data,
                "__cooldowns__":        {},
                "__weekly_signals__":   [],
                "__weekly_sent_week__": 0,
                "__heartbeat_day__":    0,
                "__streaks__":          {},
            }
        if isinstance(data, dict) and "headlines" in data:
            data["seen_headlines"] = data.pop("headlines", [])
        data.setdefault("seen_headlines",           [])
        data.setdefault("__cooldowns__",            {})
        data.setdefault("__weekly_signals__",       [])
        data.setdefault("__weekly_sent_week__",     0)
        data.setdefault("__heartbeat_day__",        0)
        data.setdefault("__streaks__",              {})
        data.setdefault("__event_warnings_sent__",  {})
        data.setdefault("__last_breakouts__",       {})
        data.setdefault("__last_prices__",          {})
        data.setdefault("__last_recap_date__",      "")
        return data
    except Exception:
        return {
            "seen_headlines":           [],
            "__cooldowns__":            {},
            "__weekly_signals__":       [],
            "__weekly_sent_week__":     0,
            "__heartbeat_day__":        0,
            "__streaks__":              {},
            "__event_warnings_sent__":  {},
            "__last_breakouts__":       {},
            "__last_prices__":          {},
            "__last_recap_date__":      "",
        }

def save_state(state: dict):
    try:
        state["seen_headlines"]     = state["seen_headlines"][-500:]
        state["__weekly_signals__"] = state["__weekly_signals__"][-200:]
        # Prune cooldowns that have already expired so the dict stays small
        cutoff = now_utc() - timedelta(minutes=SYMBOL_COOLDOWN_MIN * 2)
        pruned = {}
        for sym, ts in state.get("__cooldowns__", {}).items():
            try:
                if datetime.fromisoformat(ts) > cutoff:
                    pruned[sym] = ts
            except Exception:
                pass
        state["__cooldowns__"] = pruned
        with open(SEEN_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"  Could not save state: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL MEMORY — Phase 1: outcome tracking + lessons
# ═════════════════════════════════════════════════════════════════════════════
# Every BUY/SELL signal is saved to signal_memory.json with its entry, stop,
# and target. On each subsequent run the bot checks whether open signals have
# hit TP or SL (or expired after OUTCOME_EXPIRY_DAYS). Resolved signals feed
# a lessons-learned digest that is injected into the AI prompt so Claude can
# see the bot's own historical accuracy and which indicators predicted wins.
# ─────────────────────────────

def load_memory() -> dict:
    """Load signal_memory.json — returns a clean skeleton on first run."""
    try:
        with open(MEMORY_FILE) as f:
            data = json.load(f)
        data.setdefault("signals",          [])
        data.setdefault("poly_hist_cache",  {})
        data.setdefault("options_cache",    {})
        data.setdefault("yields_cache",     {})
        data.setdefault("short_cache",      {})
        data.setdefault("premarket_cache",  {})
        return data
    except Exception:
        return {"signals": [], "poly_hist_cache": {},
                "options_cache": {}, "yields_cache": {},
                "short_cache": {}, "premarket_cache": {}}


def save_memory(memory: dict):
    """Persist memory — keep only the most recent 300 signals."""
    try:
        memory["signals"] = memory["signals"][-300:]
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f)
    except Exception as e:
        print(f"  Could not save memory: {e}")


def record_signal(memory: dict, symbol: str, bias: str, price: float,
                  stop_str: str, target_str: str,
                  conf_score: int, conf_total: int,
                  poly_flow: dict, headline: str, base_score: int):
    """Save a sent BUY/SELL signal so its outcome can be tracked later."""
    if bias not in ("BUY", "SELL"):
        return

    # Parse stop to float
    stop_f = None
    try:
        stop_f = float(stop_str)
    except Exception:
        pass

    # Parse target — may be a range like "3050.0-3080.0"
    target_lo = target_hi = None
    try:
        tparts = str(target_str).split("-")
        if len(tparts) == 2:
            t1, t2 = float(tparts[0]), float(tparts[1])
            target_lo, target_hi = min(t1, t2), max(t1, t2)
        else:
            target_lo = target_hi = float(tparts[0])
    except Exception:
        pass

    # Record which Polygon events were present at signal time
    poly_events = [
        k for k in ("block_trade", "delta_flip", "exhaustion",
                    "absorption", "imbalance", "bar_streak")
        if poly_flow.get(k)
    ]

    memory["signals"].append({
        "ts":          now_utc().isoformat(),
        "symbol":      symbol,
        "bias":        bias,
        "price":       price,
        "stop":        stop_f,
        "target_lo":   target_lo,
        "target_hi":   target_hi,
        "conf_score":  conf_score,
        "conf_total":  conf_total,
        "score":       base_score,
        "poly_bias":   poly_flow.get("of_bias", ""),
        "buy_pct":     poly_flow.get("buy_pct"),
        "poly_events": poly_events,
        "headline":    headline[:100],
        "outcome":     None,   # filled in by check_and_update_outcomes()
        "outcome_ts":  None,
    })


def check_and_update_outcomes(memory: dict):
    """
    For every open signal (outcome=None) compare current cached price against
    the recorded stop and target.  Marks outcome as:
      TP_HIT   — price reached target (BUY: price >= target_lo | SELL: price <= target_hi)
      SL_HIT   — price hit the stop   (BUY: price <= stop      | SELL: price >= stop)
      EXPIRED  — signal older than OUTCOME_EXPIRY_DAYS with no hit
    Already-resolved signals are left untouched.
    """
    open_sigs = [s for s in memory["signals"] if s.get("outcome") is None]
    if not open_sigs:
        return

    expiry_cutoff = now_utc() - timedelta(days=OUTCOME_EXPIRY_DAYS)

    for sig in open_sigs:
        # Expire stale signals first
        try:
            if datetime.fromisoformat(sig["ts"]) < expiry_cutoff:
                sig["outcome"]    = "EXPIRED"
                sig["outcome_ts"] = now_utc().isoformat()
                continue
        except Exception:
            pass

        symbol  = sig.get("symbol", "")
        bias    = sig.get("bias",   "")
        stop_f  = sig.get("stop")
        t_lo    = sig.get("target_lo")
        t_hi    = sig.get("target_hi")

        # Get current price from live cache
        current = (_price_cache.get(symbol) or {}).get("price")
        if not current:
            continue

        now_ts = now_utc().isoformat()
        if bias == "BUY":
            if stop_f is not None and current <= stop_f:
                sig["outcome"] = "SL_HIT"; sig["outcome_ts"] = now_ts
            elif t_lo is not None and current >= t_lo:
                sig["outcome"] = "TP_HIT"; sig["outcome_ts"] = now_ts
        elif bias == "SELL":
            if stop_f is not None and current >= stop_f:
                sig["outcome"] = "SL_HIT"; sig["outcome_ts"] = now_ts
            elif t_hi is not None and current <= t_hi:
                sig["outcome"] = "TP_HIT"; sig["outcome_ts"] = now_ts


def build_lessons_digest(memory: dict, symbol: str) -> str:
    """
    Build a compact performance digest for the AI prompt.
    Shows win-rate, which Polygon indicators correlated with wins vs losses,
    and recent form — all based on *resolved* signals only.
    Returns "" if fewer than 4 resolved signals exist for this symbol.
    """
    settled = [
        s for s in memory.get("signals", [])[-200:]
        if s.get("symbol") == symbol
        and s.get("outcome") in ("TP_HIT", "SL_HIT")
    ]
    if len(settled) < 4:
        return ""

    wins   = [s for s in settled if s["outcome"] == "TP_HIT"]
    losses = [s for s in settled if s["outcome"] == "SL_HIT"]
    total  = len(settled)
    win_rate = len(wins) / total * 100

    def avg_conf(sigs):
        ratios = [s["conf_score"] / s["conf_total"]
                  for s in sigs
                  if (s.get("conf_total") or 0) > 0]
        return sum(ratios) / len(ratios) * 100 if ratios else None

    w_conf = avg_conf(wins)
    l_conf = avg_conf(losses)

    lines = [
        f"📚 SIGNAL MEMORY — {symbol} ({total} resolved signals):",
        f"  Win rate: {win_rate:.0f}%  ({len(wins)} wins / {len(losses)} losses)",
    ]
    if w_conf is not None and l_conf is not None:
        lines.append(f"  Avg confidence → wins: {w_conf:.0f}%  |  losses: {l_conf:.0f}%")

    # ── Which Polygon events predicted wins vs losses ─────────────────────────
    insights = []
    for ev in ("block_trade", "delta_flip", "imbalance", "absorption", "exhaustion"):
        with_ev    = [s for s in settled if ev in (s.get("poly_events") or [])]
        without_ev = [s for s in settled if ev not in (s.get("poly_events") or [])]
        if len(with_ev) < 3:
            continue
        w_with    = sum(1 for s in with_ev    if s["outcome"] == "TP_HIT")
        w_without = sum(1 for s in without_ev if s["outcome"] == "TP_HIT") if without_ev else 0
        wr_with   = w_with / len(with_ev)    * 100
        wr_without = w_without / len(without_ev) * 100 if without_ev else 0
        diff = wr_with - wr_without
        if abs(diff) >= 15:
            tag = "↑ better when present" if diff > 0 else "↓ worse when present"
            insights.append(
                f"{ev.replace('_', ' ')}: {wr_with:.0f}% win w/ it vs "
                f"{wr_without:.0f}% without ({tag})"
            )

    # ── buy% effectiveness ────────────────────────────────────────────────────
    for direction, threshold, bp_label in [("BUY", 65, "buy%≥65"), ("SELL", 35, "buy%≤35")]:
        dir_sigs = [s for s in settled if s.get("bias") == direction]
        if len(dir_sigs) < 4:
            continue
        if direction == "BUY":
            strong = [s for s in dir_sigs if (s.get("buy_pct") or 50) >= threshold]
            weak   = [s for s in dir_sigs if (s.get("buy_pct") or 50) <  threshold]
        else:
            strong = [s for s in dir_sigs if (s.get("buy_pct") or 50) <= threshold]
            weak   = [s for s in dir_sigs if (s.get("buy_pct") or 50) >  threshold]
        if len(strong) >= 2 and len(weak) >= 2:
            ws = sum(1 for s in strong if s["outcome"] == "TP_HIT")
            ww = sum(1 for s in weak   if s["outcome"] == "TP_HIT")
            rs = ws / len(strong) * 100
            rw = ww / len(weak)   * 100
            if abs(rs - rw) >= 15:
                insights.append(
                    f"{direction} signals — {bp_label}: {rs:.0f}% win "
                    f"vs {rw:.0f}% when not ({'+' if rs > rw else ''}{rs - rw:.0f}pp)"
                )

    if insights:
        lines.append("  Pattern analysis:")
        for ins in insights[:4]:
            lines.append(f"    • {ins}")

    # ── Recent form ───────────────────────────────────────────────────────────
    recent_5 = settled[-5:]
    r_wins   = sum(1 for s in recent_5 if s["outcome"] == "TP_HIT")
    lines.append(f"  Recent form: {r_wins}/{len(recent_5)} won (last {len(recent_5)} signals)")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# POLYGON HISTORICAL DATA — Phase 2: 2-year daily bar context
# ═════════════════════════════════════════════════════════════════════════════
# Fetches up to 2 years of daily bars for each primary symbol using the same
# Massive/Polygon API.  Cached in memory['poly_hist_cache'] for 6 hours so
# the 5-req/min rate limit is not stressed on routine 10-minute runs.
#
# Provides:
#   52-week high/low     → macro range context for the AI + extra SL/TP levels
#   200-day EMA          → long-term trend direction
#   Historical volatility (20-day ATR %) → context for stop sizing
#   Historical S/R levels (swing cluster) → additional confluence levels
#   at_hist_level        → flags when price is right at a historical key level
# ─────────────────────────────

def fetch_polygon_historical(symbol: str, memory: dict) -> dict:
    """
    Fetch 2 years of daily bars for macro/historical context.
    Cached in memory['poly_hist_cache'] for 6 hours.
    Returns {} on any failure or if POLYGON_KEY is not set.
    Caller must add time.sleep(13) ONLY when this returns fresh data
    (use the 'fetched_fresh' key in the return dict to decide).
    """
    if not POLYGON_KEY:
        return {}
    poly_sym = POLYGON_SYMBOLS.get(symbol)
    if not poly_sym:
        return {}

    cache_key       = f"{poly_sym}_hist"
    poly_hist_cache = memory.setdefault("poly_hist_cache", {})
    cached          = poly_hist_cache.get(cache_key, {})

    # Return cached data if it is less than 6 hours old
    if cached:
        try:
            cached_ts = datetime.fromisoformat(cached["fetched_at"])
            if (now_utc() - cached_ts) < timedelta(hours=6):
                data = cached.get("data", {})
                data["fetched_fresh"] = False
                return data
        except Exception:
            pass

    # ── Fetch from Polygon (daily bars, up to 2 years) ───────────────────────
    try:
        from datetime import date as _date
        to_date   = _date.today().isoformat()
        from_date = (_date.today() - timedelta(days=730)).isoformat()
        url = (
            f"https://api.massive.com/v2/aggs/ticker/{poly_sym}"
            f"/range/1/day/{from_date}/{to_date}"
        )
        params = {
            "adjusted": "true",
            "sort":     "asc",
            "limit":    750,    # 2 years ≈ 504 trading days; 750 gives headroom
            "apiKey":   POLYGON_KEY,
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            print(f"  Polygon hist: HTTP {r.status_code} for {poly_sym}")
            return {}
        resp = r.json()
        if resp.get("status") not in ("OK", "DELAYED"):
            return {}
        bars = resp.get("results", [])
        if len(bars) < 50:
            return {}

        closes = [b.get("c") or 0 for b in bars]
        highs  = [b.get("h") or 0 for b in bars]
        lows   = [b.get("l") or 0 for b in bars]
        current = closes[-1]
        if current <= 0:
            return {}

        # ── 52-week high / low ────────────────────────────────────────────────
        w52      = bars[-252:] if len(bars) >= 252 else bars
        high_52w = max((b.get("h") or 0) for b in w52)
        low_52w  = min((b.get("l") or float("inf")) for b in w52)
        if low_52w == float("inf"):
            low_52w = None

        pct_from_52h = round((current - high_52w) / high_52w * 100, 2) if high_52w else 0
        pct_from_52l = round((current - low_52w)  / low_52w  * 100, 2) if low_52w  else 0

        # ── 200-day EMA ───────────────────────────────────────────────────────
        ema_200 = None
        ema_200_trend = ""
        if len(closes) >= 200:
            ema_200 = calculate_ema(closes, 200)
        elif len(closes) >= 50:
            ema_200 = calculate_ema(closes, max(50, len(closes) // 2))
        if ema_200 and current:
            if current > ema_200:
                ema_200_trend = f"above 200d EMA ({ema_200}) — long-term uptrend"
            else:
                ema_200_trend = f"below 200d EMA ({ema_200}) — long-term downtrend"

        # ── 20-day ATR as % of price (normalized historical volatility) ───────
        hist_vol_pct = None
        if len(highs) >= 22 and len(lows) >= 22 and len(closes) >= 22:
            atr_20 = calculate_atr(highs[-22:], lows[-22:], closes[-22:], period=20)
            if atr_20 and current:
                hist_vol_pct = round(atr_20 / current * 100, 2)

        # ── Swing-based historical key S/R levels ─────────────────────────────
        sig_levels = []
        for i in range(2, len(closes) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2]
                    and highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                sig_levels.append(("R", round(highs[i], 4)))
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2]
                    and lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                sig_levels.append(("S", round(lows[i], 4)))

        # Cluster nearby levels (within 0.5% of each other → same zone)
        clustered: list = []
        for lvl_type, lvl_price in sig_levels[-200:]:
            if lvl_price <= 0:
                continue
            merged = False
            for cg in clustered:
                if abs(cg["price"] - lvl_price) / lvl_price < 0.005:
                    n = cg["count"]
                    cg["price"] = round((cg["price"] * n + lvl_price) / (n + 1), 4)
                    cg["count"] += 1
                    merged = True
                    break
            if not merged:
                clustered.append({"price": lvl_price, "type": lvl_type, "count": 1})

        # Top 6 most-tested zones
        key_levels = sorted(clustered, key=lambda x: x["count"], reverse=True)[:6]

        # ── Is price near a historical key level? ─────────────────────────────
        at_hist_level = None
        for kl in key_levels:
            if kl["price"] > 0:
                dist_pct = abs(current - kl["price"]) / current * 100
                if dist_pct <= 0.5:
                    lbl = "resistance" if kl["type"] == "R" else "support"
                    at_hist_level = (
                        f"Price at historical {lbl} zone {kl['price']} "
                        f"({kl['count']} historical touches) — high-confluence level"
                    )
                    break

        hist_data = {
            "high_52w":      round(high_52w, 4) if high_52w else None,
            "low_52w":       round(low_52w,  4) if low_52w  else None,
            "pct_from_52h":  pct_from_52h,
            "pct_from_52l":  pct_from_52l,
            "ema_200":       ema_200,
            "ema_200_trend": ema_200_trend,
            "hist_vol_pct":  hist_vol_pct,
            "key_levels":    key_levels,
            "at_hist_level": at_hist_level,
            "bar_count":     len(bars),
            "fetched_fresh": True,
        }

        # Store in cache
        poly_hist_cache[cache_key] = {
            "fetched_at": now_utc().isoformat(),
            "data":       {k: v for k, v in hist_data.items() if k != "fetched_fresh"},
        }
        print(f"  Polygon hist: {len(bars)} daily bars for {poly_sym} ({symbol})")
        return hist_data

    except Exception as e:
        print(f"  Polygon hist error ({symbol}): {e}")
        return {}


# ─────────────────────────────
# OPTIONS FLOW
# ─────────────────────────────
def fetch_options_flow(symbol: str, memory: dict) -> dict:
    """Fetch options chain snapshot for SPY/QQQ from Polygon.

    Returns PCR, IV, unusual activity, and gamma exposure.
    Only works for equity ETFs (SPY, QQQ) — commodity ETFs excluded.
    """
    if not POLYGON_KEY:
        return {}
    poly_sym = POLYGON_SYMBOLS.get(symbol, "")
    if poly_sym not in ("SPY", "QQQ"):
        return {}
    if not _ep_ok("options"):
        return {}

    options_cache = memory.setdefault("options_cache", {})
    cache_key = f"{poly_sym}_options"
    cached = options_cache.get(cache_key)
    if cached:
        try:
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if (now_utc() - fetched_at) < timedelta(minutes=15):
                result = dict(cached["data"])
                result["fetched_fresh"] = False
                return result
        except Exception:
            pass

    try:
        url  = f"https://api.massive.com/v3/snapshot/options/{poly_sym}"
        resp = requests.get(url, params={"limit": 250, "order": "desc",
                                          "sort": "open_interest", "apiKey": POLYGON_KEY},
                            timeout=5)
        if resp.status_code != 200:
            _ep_fail("options")
            return {}
        results = resp.json().get("results") or []

        total_call_oi = 0
        total_put_oi  = 0
        call_items    = []
        put_items     = []

        for item in results:
            details = item.get("details") or {}
            ctype   = (details.get("contract_type") or "").lower()
            oi      = item.get("open_interest") or 0
            vol     = (item.get("day") or {}).get("volume")
            iv      = item.get("implied_volatility")
            greeks  = item.get("greeks") or {}
            delta   = greeks.get("delta")
            gamma   = greeks.get("gamma")
            strike  = details.get("strike_price") or 0

            if ctype == "call":
                total_call_oi += oi
                call_items.append({"oi": oi, "vol": vol, "iv": iv, "gamma": gamma,
                                   "delta": delta, "strike": strike})
            elif ctype == "put":
                total_put_oi += oi
                put_items.append({"oi": oi, "vol": vol, "iv": iv, "gamma": gamma,
                                  "delta": delta, "strike": strike})

        # PCR
        pcr = None
        pcr_label = ""
        if total_call_oi > 0:
            pcr = round(total_put_oi / total_call_oi, 2)
            if pcr < 0.6:
                pcr_label = "extreme call buying (bullish sentiment)"
            elif pcr < 0.8:
                pcr_label = "call-heavy (bullish bias)"
            elif pcr < 1.1:
                pcr_label = "balanced"
            elif pcr < 1.4:
                pcr_label = "put-heavy (bearish bias)"
            else:
                pcr_label = "extreme put buying (bearish sentiment)"

        # IV average from top 50 by OI
        all_items_sorted = sorted(results, key=lambda x: x.get("open_interest") or 0, reverse=True)
        iv_vals = [item.get("implied_volatility") for item in all_items_sorted[:50]
                   if item.get("implied_volatility") is not None]
        iv_avg = round(sum(iv_vals) / len(iv_vals), 4) if iv_vals else None

        # Unusual activity
        unusual_calls = []
        for c in sorted(call_items, key=lambda x: x["oi"], reverse=True):
            if c["vol"] is not None and c["oi"] > 0 and c["vol"] / c["oi"] > 1.5:
                ratio = c["vol"] / c["oi"]
                unusual_calls.append(f"Call {c['strike']} (vol/OI={ratio:.1f}x)")
                if len(unusual_calls) >= 3:
                    break

        unusual_puts = []
        for p in sorted(put_items, key=lambda x: x["oi"], reverse=True):
            if p["vol"] is not None and p["oi"] > 0 and p["vol"] / p["oi"] > 1.5:
                ratio = p["vol"] / p["oi"]
                unusual_puts.append(f"Put {p['strike']} (vol/OI={ratio:.1f}x)")
                if len(unusual_puts) >= 3:
                    break

        if unusual_calls and unusual_puts:
            unusual_summary = "Unusual activity on BOTH sides — directional uncertainty"
        elif unusual_calls:
            unusual_summary = f"{len(unusual_calls)} unusual call sweeps (bullish positioning)"
        elif unusual_puts:
            unusual_summary = f"{len(unusual_puts)} unusual put sweeps (bearish positioning)"
        else:
            unusual_summary = ""

        # Gamma exposure
        net_gamma = 0.0
        for c in call_items:
            if c["gamma"] is not None:
                net_gamma += c["gamma"] * c["oi"] * 100
        for p in put_items:
            if p["gamma"] is not None:
                net_gamma -= p["gamma"] * p["oi"] * 100

        if net_gamma > 0:
            gamma_bias = "positive gamma (dealer hedging = dampens moves)"
        elif net_gamma < 0:
            gamma_bias = "negative gamma (dealer hedging = amplifies moves)"
        else:
            gamma_bias = ""

        data = {
            "total_call_oi":  total_call_oi,
            "total_put_oi":   total_put_oi,
            "pcr":            pcr,
            "pcr_label":      pcr_label,
            "iv_avg":         iv_avg,
            "unusual_calls":  unusual_calls,
            "unusual_puts":   unusual_puts,
            "unusual_summary": unusual_summary,
            "gamma_bias":     gamma_bias,
            "net_gamma":      round(net_gamma, 2),
        }
        options_cache[cache_key] = {
            "fetched_at": now_utc().isoformat(),
            "data":       data,
        }
        result = dict(data)
        result["fetched_fresh"] = True
        _ep_reset("options")
        print(f"  Options flow fetched for {poly_sym}: PCR={pcr}, IV={iv_avg}")
        return result

    except Exception as e:
        _ep_fail("options")
        print(f"  Options flow error ({symbol}): {e}")
        return {}


# ─────────────────────────────
# TREASURY YIELDS
# ─────────────────────────────
def fetch_treasury_yields(memory: dict) -> dict:
    """Fetch US Treasury yield curve from Polygon economy endpoint.

    Returns 2Y, 10Y, 30Y yields and curve interpretation.
    """
    if not POLYGON_KEY:
        return {}
    if not _ep_ok("yields"):
        return {}

    yields_cache = memory.setdefault("yields_cache", {})
    cached = yields_cache.get("yields")
    if cached:
        try:
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if (now_utc() - fetched_at) < timedelta(minutes=60):
                result = dict(cached["data"])
                result["fetched_fresh"] = False
                return result
        except Exception:
            pass

    try:
        url  = "https://api.massive.com/economy/v1/treasury-yields"
        resp = requests.get(url, params={"apiKey": POLYGON_KEY}, timeout=5)
        if resp.status_code != 200:
            _ep_fail("yields")
            return {}
        results = resp.json().get("results") or []
        if not results:
            _ep_fail("yields")
            return {}

        # Most recent entry (last item)
        entry = results[-1]

        def _get_yield(d, *keys):
            for k in keys:
                v = d.get(k)
                if v is not None:
                    return float(v)
            return None

        y2  = _get_yield(entry, "2Y", "2y", "2_year")
        y10 = _get_yield(entry, "10Y", "10y", "10_year")
        y30 = _get_yield(entry, "30Y", "30y", "30_year")

        spread_2_10 = None
        curve_label = ""
        equity_signal = ""
        gold_signal   = ""

        if y2 is not None and y10 is not None:
            spread_2_10 = round(y10 - y2, 2)
            if spread_2_10 < -0.5:
                curve_label = "Deeply inverted (2y > 10y) — strong recession signal"
            elif spread_2_10 < -0.1:
                curve_label = "Mildly inverted — caution"
            elif spread_2_10 < 0.3:
                curve_label = "Flat — neutral"
            else:
                curve_label = "Normal/steep — growth expectations"

            if spread_2_10 < -0.3 and y2 > 4.5:
                equity_signal = "bearish_equities"
            elif spread_2_10 > 0.5:
                equity_signal = "bullish_equities"

        if y10 is not None:
            if y10 < 3.5 or (spread_2_10 is not None and spread_2_10 < -0.3):
                gold_signal = "bullish_gold"
            elif y10 > 5.0:
                gold_signal = "bearish_gold"

        data = {
            "y2":           y2,
            "y10":          y10,
            "y30":          y30,
            "spread_2_10":  spread_2_10,
            "curve_label":  curve_label,
            "equity_signal": equity_signal,
            "gold_signal":  gold_signal,
            "date":         entry.get("date", ""),
        }
        yields_cache["yields"] = {
            "fetched_at": now_utc().isoformat(),
            "data":       data,
        }
        result = dict(data)
        result["fetched_fresh"] = True
        _ep_reset("yields")
        print(f"  Treasury yields fetched: 2Y={y2}, 10Y={y10}, spread={spread_2_10}")
        return result

    except Exception as e:
        _ep_fail("yields")
        print(f"  Treasury yields error: {e}")
        return {}


# ─────────────────────────────
# SHORT DATA
# ─────────────────────────────
def fetch_short_data(symbol: str, memory: dict) -> dict:
    """Fetch short interest + daily short volume for SPY/QQQ from Polygon.

    Two calls per refresh (same 6-hour cache):
      /stocks/v1/short-interest → bi-weekly DTC + short float %
      /stocks/v1/short-volume   → today's intraday short sale volume ratio
        short_vol_ratio > 55% = heavy shorting (bearish pressure)
        short_vol_ratio < 40% = light shorting (low supply side pressure)
    Returns days-to-cover, short float %, short volume ratio, squeeze flag.
    """
    if not POLYGON_KEY:
        return {}
    poly_sym = POLYGON_SYMBOLS.get(symbol, "")
    if poly_sym not in ("SPY", "QQQ"):
        return {}
    if not _ep_ok("short"):
        return {}

    short_cache = memory.setdefault("short_cache", {})
    cache_key = f"{poly_sym}_short"
    cached = short_cache.get(cache_key)
    if cached:
        try:
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if (now_utc() - fetched_at) < timedelta(hours=6):
                result = dict(cached["data"])
                result["fetched_fresh"] = False
                return result
        except Exception:
            pass

    try:
        url  = f"https://api.massive.com/stocks/v1/short-interest"
        resp = requests.get(url, params={"ticker": poly_sym, "limit": 2,
                                          "apiKey": POLYGON_KEY}, timeout=5)
        if resp.status_code != 200:
            _ep_fail("short")
            return {}
        results = resp.json().get("results") or []
        if not results:
            _ep_fail("short")
            return {}

        entry = results[0]

        # Flexible field name handling
        shares_short = (entry.get("short_interest") or
                        entry.get("shares_short") or 0)
        avg_vol      = (entry.get("avg_daily_share_volume") or
                        entry.get("avg_daily_volume") or 0)

        days_to_cover = None
        if shares_short and avg_vol:
            days_to_cover = round(shares_short / avg_vol, 1)

        short_float_pct = entry.get("percent_float")
        if short_float_pct is None:
            outstanding = entry.get("outstanding_shares")
            if outstanding and shares_short:
                short_float_pct = round(shares_short / outstanding * 100, 2)

        # ── Short volume (today's intraday short sale ratio) ─────────────────
        # Distinct from short interest: this is the % of TODAY's volume that
        # is short selling. Resets daily. High ratio = active bear pressure NOW.
        short_vol_ratio    = None
        short_vol_label    = ""
        try:
            from datetime import date as _sv_date
            today_str = _sv_date.today().isoformat()
            sv_resp = requests.get(
                "https://api.massive.com/stocks/v1/short-volume",
                params={"ticker": poly_sym, "date": today_str,
                        "limit": 1, "apiKey": POLYGON_KEY},
                timeout=5,
            )
            if sv_resp.status_code == 200:
                sv_results = sv_resp.json().get("results") or []
                if sv_results:
                    sv = sv_results[0]
                    sv_short = (sv.get("short_volume") or
                                sv.get("shortVolume") or 0)
                    sv_total = (sv.get("total_volume") or
                                sv.get("totalVolume") or 0)
                    if sv_total > 0 and sv_short > 0:
                        short_vol_ratio = round(sv_short / sv_total * 100, 1)
                        if short_vol_ratio > 55:
                            short_vol_label = (f"Heavy shorting today "
                                               f"({short_vol_ratio}% of volume) — bearish supply pressure")
                        elif short_vol_ratio < 40:
                            short_vol_label = (f"Light shorting today "
                                               f"({short_vol_ratio}% of volume) — bears stepping back")
                        else:
                            short_vol_label = f"Neutral short volume ({short_vol_ratio}%)"
        except Exception as sv_err:
            print(f"  Short volume fetch failed (non-fatal): {sv_err}")

        # Squeeze score: 0-4
        squeeze_score = 0
        if days_to_cover and days_to_cover > 3:
            squeeze_score += 1
        if short_float_pct and short_float_pct > 5:
            squeeze_score += 1
        # High short volume on a rising day = classic squeeze pressure
        if short_vol_ratio and short_vol_ratio > 55:
            squeeze_score += 1

        # Check if price is above 5-day average using cached price data
        price_data = _price_cache.get(symbol) or {}
        price = price_data.get("price")
        ema5  = price_data.get("ema20")  # proxy: if price > ema20, rising
        if price and ema5 and price > ema5:
            squeeze_score += 1

        squeeze_setup = squeeze_score >= 2

        short_label = ""
        if days_to_cover is not None:
            if days_to_cover > 5:
                short_label = f"High short interest ({days_to_cover:.1f} DTC) — squeeze risk"
            elif days_to_cover > 2:
                short_label = f"Moderate short interest ({days_to_cover:.1f} DTC)"
            else:
                short_label = f"Low short interest ({days_to_cover:.1f} DTC)"

        data = {
            "shares_short":     shares_short,
            "days_to_cover":    days_to_cover,
            "short_float_pct":  short_float_pct,
            "short_vol_ratio":  short_vol_ratio,
            "short_vol_label":  short_vol_label,
            "squeeze_score":    squeeze_score,
            "squeeze_setup":    squeeze_setup,
            "short_label":      short_label,
        }
        short_cache[cache_key] = {
            "fetched_at": now_utc().isoformat(),
            "data":       data,
        }
        result = dict(data)
        result["fetched_fresh"] = True
        _ep_reset("short")
        print(f"  Short data fetched for {poly_sym}: DTC={days_to_cover}, "
              f"short_vol={short_vol_ratio}%, squeeze={squeeze_setup}")
        return result

    except Exception as e:
        _ep_fail("short")
        print(f"  Short data error ({symbol}): {e}")
        return {}


# ─────────────────────────────
# PRE-MARKET / AFTER-HOURS
# ─────────────────────────────
def fetch_premarket_data(symbol: str, memory: dict) -> dict:
    """Fetch previous day's open/close and after-hours/pre-market data.

    Returns gap analysis vs previous close.
    """
    if not POLYGON_KEY:
        return {}
    poly_sym = POLYGON_SYMBOLS.get(symbol, "")
    if not poly_sym:
        return {}
    if not _ep_ok("premarket"):
        return {}

    premarket_cache = memory.setdefault("premarket_cache", {})
    cache_key = f"{poly_sym}_premarket"
    cached = premarket_cache.get(cache_key)
    if cached:
        try:
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if (now_utc() - fetched_at) < timedelta(minutes=30):
                result = dict(cached["data"])
                result["fetched_fresh"] = False
                return result
        except Exception:
            pass

    try:
        from datetime import date as _date_cls
        yesterday = _date_cls.today() - timedelta(days=1)
        # Skip Sunday → use Friday
        if yesterday.weekday() == 6:
            yesterday = yesterday - timedelta(days=2)
        # Skip Saturday → use Friday
        elif yesterday.weekday() == 5:
            yesterday = yesterday - timedelta(days=1)
        yesterday_str = yesterday.isoformat()

        url  = f"https://api.massive.com/v1/open-close/{poly_sym}/{yesterday_str}"
        resp = requests.get(url, params={"adjusted": "true", "apiKey": POLYGON_KEY},
                            timeout=5)
        if resp.status_code != 200:
            _ep_fail("premarket")
            return {}
        body = resp.json()

        prev_close   = body.get("close")
        after_hours  = body.get("afterHours")
        pre_market   = body.get("preMarket")

        ah_change_pct = None
        pm_change_pct = None
        if prev_close and prev_close > 0:
            if after_hours is not None:
                ah_change_pct = round((after_hours - prev_close) / prev_close * 100, 2)
            if pre_market is not None:
                pm_change_pct = round((pre_market - prev_close) / prev_close * 100, 2)

        gap_note_parts = []
        if ah_change_pct is not None and abs(ah_change_pct) > 0.5:
            gap_note_parts.append(f"After-hours: {ah_change_pct:+.2f}% vs prev close")
        if pm_change_pct is not None and abs(pm_change_pct) > 0.3:
            gap_note_parts.append(f"Pre-market: {pm_change_pct:+.2f}%")
        gap_note = " | ".join(gap_note_parts)

        # Determine gap bias from the most recent extended-hours price
        gap_value = pm_change_pct if pm_change_pct is not None else ah_change_pct
        if gap_value is not None and gap_value > 0.5:
            gap_bias = "bullish"
        elif gap_value is not None and gap_value < -0.5:
            gap_bias = "bearish"
        else:
            gap_bias = "neutral"

        data = {
            "prev_close":     prev_close,
            "after_hours":    after_hours,
            "pre_market":     pre_market,
            "ah_change_pct":  ah_change_pct,
            "pm_change_pct":  pm_change_pct,
            "gap_note":       gap_note,
            "gap_bias":       gap_bias,
            "date":           yesterday_str,
        }
        premarket_cache[cache_key] = {
            "fetched_at": now_utc().isoformat(),
            "data":       data,
        }
        result = dict(data)
        result["fetched_fresh"] = True
        _ep_reset("premarket")
        print(f"  Pre-market data fetched for {poly_sym}: gap_bias={gap_bias}")
        return result

    except Exception as e:
        _ep_fail("premarket")
        print(f"  Pre-market data error ({symbol}): {e}")
        return {}


# ─────────────────────────────
# SIGNAL COOLDOWN
# ─────────────────────────────
def cooldown_active(symbol: str, cooldowns: dict) -> bool:
    last_str = cooldowns.get(symbol)
    if not last_str:
        return False
    try:
        last = datetime.fromisoformat(last_str)
        return (now_utc() - last) < timedelta(minutes=SYMBOL_COOLDOWN_MIN)
    except Exception:
        return False

# ─────────────────────────────
# PRIMARY SYMBOL
# ─────────────────────────────
def get_primary_symbol(title: str, moves: dict) -> str:
    for sym, keywords in ASSET_MAP.items():
        if any(k in title.lower() for k in keywords) and sym in moves:
            return sym
    if moves:
        return max(moves, key=lambda s: abs(moves[s]["move"]))
    return "^GSPC"

# ─────────────────────────────
# SCORING
# ─────────────────────────────
ENERGY_KEYWORDS = [
    "oil", "crude", "wti", "brent", "opec", "petroleum", "energy",
    "gold", "xau", "bullion", "precious metal",
    "silver", "xag", "comex silver",
    "copper", "comex copper", "freeport", "southern copper",
    "aluminium", "aluminum", "alcoa", "bauxite",
]

def score_signal(title: str, moves: dict, signal_type: str, src_count: int = 1):
    avg_move   = sum(abs(d["move"]) for d in moves.values()) / len(moves) if moves else 0
    api_failed = len(moves) == 0

    # ── Freshness thresholds ─────────────────────────────────────────────────
    if avg_move < 1.0:   freshness, reaction = 50, "FRESH"
    elif avg_move < 2.5: freshness, reaction = 40, "WARMING"
    elif avg_move < 5.0: freshness, reaction = 25, "MOVING"
    elif avg_move < 8.0: freshness, reaction = 10, "RUNNING"
    else:                freshness, reaction = 0,  "PRICED IN"

    # ── Bonuses ─────────────────────────────────────────────────────────────
    macro_bonus    = 20 if is_macro(title) else 0
    energy_bonus   = 15 if any(k in title.lower() for k in ENERGY_KEYWORDS) else 0
    analysis_bonus = 10 if signal_type == "📊 ANALYSIS" else 0
    breadth_bonus  = 10 if api_failed else min(20, len(moves) * 4)
    source_bonus   = min(20, (src_count - 1) * 7)

    vol_bonus = 0
    for d in moves.values():
        if (d.get("vol_ratio") or 0) >= 2.0:
            vol_bonus = 10
            break

    # Delta flip bonus — confirmed order flow reversal adds conviction
    delta_bonus = 0
    for d in moves.values():
        of = d.get("order_flow") or {}
        if of.get("delta_flip"):
            delta_bonus = 8
            break

    # Polygon order flow quality bonus — scales with how strong the real
    # data signal is, not just whether Polygon responded at all.
    polygon_bonus = 0
    for d in moves.values():
        pf = d.get("polygon_flow") or {}
        if pf.get("data_source") == "polygon_real":
            polygon_bonus = 5                                          # base: real data present
            if pf.get("of_bias") in ("bullish", "bearish"):
                polygon_bonus += 3                                     # clear directional delta
            bp = pf.get("buy_pct") or 50
            if bp >= 65 or bp <= 35:
                polygon_bonus += 3                                     # strong buy/sell pressure
            if pf.get("imbalance"):
                polygon_bonus += 3                                     # bars dominated one side
            if pf.get("block_trade"):
                polygon_bonus += 4                                     # institutional footprint
            if (pf.get("vol_spike") or 1.0) >= 2.0:
                polygon_bonus += 3                                     # volume confirmation
            polygon_bonus = min(polygon_bonus, 20)                    # cap at 20
            break

    # Options flow bonus — strong PCR extreme + unusual activity
    options_bonus = 0
    for d in moves.values():
        opts = d.get("options_flow") or {}
        if opts.get("pcr") is not None:
            pcr = opts["pcr"]
            if pcr < 0.6 or pcr > 1.5:        # extreme positioning
                options_bonus += 4
            if opts.get("unusual_summary"):    # unusual activity detected
                options_bonus += 3
            options_bonus = min(options_bonus, 8)
            break

    # Short squeeze bonus — high short interest aligned with BUY
    # Squeeze / short-volume bonus
    # +5 for squeeze setup (high DTC + rising price = short squeeze risk)
    # +3 for heavy intraday short volume on a bearish signal (bears piling in)
    squeeze_bonus = 0
    for d in moves.values():
        short = d.get("short_data") or {}
        if short.get("squeeze_setup"):
            squeeze_bonus = max(squeeze_bonus, 5)
        svr = short.get("short_vol_ratio")
        if svr is not None and svr > 55:    # heavy shorting today = bearish conviction
            squeeze_bonus = max(squeeze_bonus, 3)
        if squeeze_bonus >= 5:
            break

    # Treasury yield macro bonus
    yields_bonus = 0
    for d in moves.values():
        yields = d.get("treasury_yields") or {}
        eq_sig = yields.get("equity_signal", "")
        if eq_sig:
            yields_bonus = 4
            break

    total = min(100, freshness + macro_bonus + energy_bonus
                     + analysis_bonus + breadth_bonus + source_bonus
                     + vol_bonus + delta_bonus + polygon_bonus
                     + options_bonus + squeeze_bonus + yields_bonus)
    return total, reaction

def label(s: int) -> str:
    if s < 40:  return "🟥 NO TRADE"
    if s < 55:  return "🟠 WEAK"
    if s < 70:  return "🟡 WATCH"
    if s < 85:  return "🟢 GOOD SETUP"
    return "🔥 HIGH CONVICTION"

# ─────────────────────────────
# AI ANALYSIS
# ─────────────────────────────
def _build_tech_block(d: dict, label: str) -> str:
    """Format a full technical data block for one timeframe."""
    lines = [f"{label}:"]
    if d.get("rsi") is not None:
        lines.append(f"  RSI(14):    {rsi_label(d['rsi'])}")
    if d.get("macd") is not None and d.get("macd_signal") is not None:
        lines.append(f"  MACD:       {macd_label(d['macd'], d['macd_signal'])}")
    if d.get("trend"):
        lines.append(f"  Trend:      {d['trend']}")
    if d.get("ema20"):
        lines.append(f"  EMA20:      {d['ema20']}")
    if d.get("ema50"):
        lines.append(f"  EMA50:      {d['ema50']}")
    if d.get("vol_ratio") is not None:
        lines.append(f"  Volume:     {volume_label(d['vol_ratio'])}")
    if d.get("support") is not None:
        lines.append(f"  Support:    {d['support']}")
    if d.get("resistance") is not None:
        lines.append(f"  Resistance: {d['resistance']}")
    if d.get("atr") is not None:
        lines.append(f"  ATR(14):    {d['atr']}")
    if d.get("divergence"):
        lines.append(f"  Divergence: {d['divergence'].upper()} RSI divergence detected")
    fvg = d.get("fvg") or {}
    if fvg:
        lines.append(f"  FVG:        {fvg_label(fvg)}")
    fib = d.get("fib") or {}
    if fib:
        prox = fib.get("proximity_pct", 99)
        note = f" ← price within {prox}% (AT KEY LEVEL)" if prox < 0.5 else ""
        lines.append(
            f"  Fib ({fib['direction']}): EQ={fib['equilibrium']} | "
            f"0.382={fib['fib_0382']} | 0.618={fib['fib_0618']}{note}"
        )
        lines.append(
            f"  Nearest fib:  {fib['nearest_level']} @ {fib['nearest_price']} "
            f"({prox}% away)"
        )
        lines.append(
            f"  Fib targets:  1.272={fib['fib_1272']} | 1.618={fib['fib_1618']}"
        )
    ob = d.get("ob") or {}
    if ob.get("bullish"):
        b = ob["bullish"]
        lines.append(f"  Bull OB:    {b['low']} – {b['high']} ({b['status']})")
    if ob.get("bearish"):
        br = ob["bearish"]
        lines.append(f"  Bear OB:    {br['low']} – {br['high']} ({br['status']})")
    vwap_d = d.get("vwap") or {}
    if vwap_d:
        lines.append(f"  VWAP:       {vwap_d.get('vwap')} ({vwap_d.get('position', '')})")
        if vwap_d.get("vwap_u1"):
            lines.append(
                f"  VWAP bands: +1σ={vwap_d['vwap_u1']} / -1σ={vwap_d['vwap_l1']} | "
                f"+2σ={vwap_d['vwap_u2']} / -2σ={vwap_d['vwap_l2']}"
            )
    mkt_str = d.get("market_structure") or {}
    if mkt_str:
        lines.append(f"  Mkt Struct: {mkt_str.get('structure', '')}")
        if mkt_str.get("event"):
            lines.append(f"  Struct Evt: {mkt_str['event']}")
    price = d.get("price")
    fib_d = d.get("fib") or {}
    if price and fib_d:
        pd_lbl = premium_discount_label(price, fib_d)
        if pd_lbl:
            lines.append(f"  P/D Zone:   {pd_lbl}")
    if d.get("candle_pattern"):
        lines.append(f"  Candle:     {d['candle_pattern']}")
    if d.get("liquidity_sweep"):
        lines.append(f"  Liq Sweep:  {d['liquidity_sweep']}")
    of = d.get("order_flow") or {}
    if of:
        _last = f" | last bar={of['delta']:+,}" if "delta" in of else ""
        lines.append(f"  Delta:      {of['of_bias'].upper()} (cum={of['cum_delta']:+,}{_last})")
        if of.get("delta_flip"):
            lines.append(f"  ⚡ {of['delta_flip']}")
    return "\n".join(lines)

def analyze(title: str, reaction: str, moves: dict, signal_type: str,
            calendar_events: list, primary_symbol: str = "",
            memory: dict = None) -> str:
    """
    Full AI analysis of a signal.  Now accepts optional `memory` dict to:
      1. Fetch + cache 2-year historical data for the primary symbol
      2. Inject a lessons-learned digest from past signal outcomes
    Both additions are injected into the Claude prompt for improved reasoning.
    """
    memory = memory or {}

    # ── Build rich technical context for primary instrument ──────────────────
    primary_data = moves.get(primary_symbol, {})

    tech_15m = _build_tech_block(primary_data, "15-minute chart") if primary_data else ""

    _price = primary_data.get("price")  # shared across timeframe sub-dicts for P/D zone

    data_5m = {
        "rsi":              primary_data.get("5m_rsi"),
        "trend":            primary_data.get("5m_trend"),
        "support":          primary_data.get("5m_support"),
        "resistance":       primary_data.get("5m_resistance"),
        "fvg":              primary_data.get("5m_fvg"),
        "fib":              primary_data.get("5m_fib"),
        "ob":               primary_data.get("5m_ob"),
        "market_structure": primary_data.get("5m_market_structure"),
        "candle_pattern":   primary_data.get("5m_candle_pattern"),
        "liquidity_sweep":  primary_data.get("5m_liquidity_sweep"),
        "order_flow":       primary_data.get("5m_order_flow"),
        "price":            _price,
    }
    tech_5m = _build_tech_block(data_5m, "5-minute chart") if any(v for v in data_5m.values() if v) else ""

    hourly_data = {
        "rsi":   primary_data.get("hourly_rsi"),
        "trend": primary_data.get("hourly_trend"),
    }
    tech_1h = _build_tech_block(hourly_data, "1-hour chart") if any(hourly_data.values()) else ""

    data_4h = {
        "rsi":              primary_data.get("4h_rsi"),
        "trend":            primary_data.get("4h_trend"),
        "support":          primary_data.get("4h_support"),
        "resistance":       primary_data.get("4h_resistance"),
        "fvg":              primary_data.get("4h_fvg"),
        "fib":              primary_data.get("4h_fib"),
        "ob":               primary_data.get("4h_ob"),
        "market_structure": primary_data.get("4h_market_structure"),
        "liquidity_sweep":  primary_data.get("4h_liquidity_sweep"),
        "order_flow":       primary_data.get("4h_order_flow"),
        "price":            _price,
    }
    tech_4h = _build_tech_block(data_4h, "4-hour chart") if any(v for v in data_4h.values() if v) else ""

    daily_data = {
        "rsi":              primary_data.get("daily_rsi"),
        "trend":            primary_data.get("daily_trend"),
        "support":          primary_data.get("daily_support"),
        "resistance":       primary_data.get("daily_resistance"),
        "fib":              primary_data.get("daily_fib"),
        "market_structure": primary_data.get("daily_market_structure"),
        "order_flow":       primary_data.get("daily_order_flow"),
        "price":            _price,
    }
    tech_daily = _build_tech_block(daily_data, "Daily chart") if any(daily_data.values()) else ""

    tech_block = "\n\n".join(filter(None, [tech_5m, tech_15m, tech_1h, tech_4h, tech_daily]))
    if not tech_block:
        tech_block = "  Technical data unavailable"

    # ── Other instrument moves (context) ────────────────────────────────────
    other_rows = []
    for s, d in moves.items():
        if s == primary_symbol or s not in ASSET_MAP:
            continue
        other_rows.append(f"  {friendly(s)}: {d['move']:+.2f}% | price: {d['price']}")
    other_str = "\n".join(other_rows) if other_rows else "  n/a"

    # ── Calendar ─────────────────────────────────────────────────────────────
    cal_str = ", ".join([f"{e['title']} ({e['currency']})" for e in calendar_events[:4]]) if calendar_events else "none"

    # ── Market fear context ───────────────────────────────────────────────────
    vix_d = _price_cache.get("^VIX")
    fear_str = (
        f"VIX: {vix_d.get('price', 'n/a')} — {vix_label(vix_d.get('price', 0)) if vix_d and vix_d.get('price') else 'n/a'}"
        if vix_d else "n/a"
    )

    # ── DXY context ──────────────────────────────────────────────────────────
    dxy_d = _price_cache.get("DX-Y.NYB")
    dxy_str = f"DXY: {dxy_d['price']} ({dxy_d['move']:+.2f}% today) | {dxy_d.get('trend','n/a')}" if dxy_d else "n/a"

    macro_note = "This is a macro event affecting all instruments." if is_macro(title) else ""

    # ── Polygon real order flow context (fed into AI prompt) ─────────────────
    poly_flow = primary_data.get("polygon_flow") or {}
    polygon_context = ""
    if poly_flow.get("data_source") == "polygon_real":
        pn    = poly_flow.get("proxy_note", "")
        parts = [f"Source: Polygon 1-min bars{' (' + pn + ')' if pn else ''}"]

        delta = poly_flow.get("cum_delta")
        if delta is not None:
            parts.append(f"Cumulative delta: {delta:+,.0f} ({'bullish' if delta > 0 else 'bearish'} net flow)")
        buy_pct = poly_flow.get("buy_pct")
        if buy_pct is not None:
            lbl = "buyers dominant" if buy_pct >= 55 else "sellers dominant" if buy_pct <= 45 else "balanced"
            parts.append(f"Buy pressure: {buy_pct}% ({lbl})")
        vwap_v = poly_flow.get("vwap")
        pvwap  = poly_flow.get("price_vs_vwap", "")
        if vwap_v is not None:
            parts.append(f"VWAP: {vwap_v} — price is {pvwap}")
        vpoc = poly_flow.get("vpoc")
        if vpoc is not None:
            parts.append(f"VPOC (most-traded price): {vpoc}")
        vah = poly_flow.get("vah"); val = poly_flow.get("val")
        if vah and val:
            parts.append(f"Value Area: {val} – {vah} (70% of session volume)")
        vs = poly_flow.get("vol_spike")
        if vs and vs >= 1.5:
            parts.append(f"Volume spike: {vs}× average — elevated participation")
        mkt = poly_flow.get("mkt_structure", "")
        if mkt:
            parts.append(f"Market structure: {mkt}")
        kr = poly_flow.get("key_resistance"); ks = poly_flow.get("key_support")
        if kr and ks:
            parts.append(f"Key S/R from swings: support {ks} | resistance {kr}")
        s_hi2 = poly_flow.get("session_high"); s_lo2 = poly_flow.get("session_low")
        if s_hi2 and s_lo2:
            parts.append(f"Session high/low: {s_lo2} – {s_hi2}")
        or_st2 = poly_flow.get("or_status", "")
        if or_st2:
            parts.append(f"Opening range (15-min): {or_st2}")
        for ev_key in ("block_trade", "delta_flip", "exhaustion", "absorption", "imbalance", "bar_streak"):
            ev = poly_flow.get(ev_key, "")
            if ev:
                parts.append(f"{ev_key.replace('_',' ').title()}: {ev}")
        polygon_context = "\n\n━━━ REAL ORDER FLOW (Polygon — live exchange data) ━━━\n" + "\n".join(parts)

    # ── Historical context (2-year daily bars, Polygon) ───────────────────────
    hist_block = ""
    if POLYGON_KEY and primary_symbol in POLYGON_SYMBOLS:
        hist_data = fetch_polygon_historical(primary_symbol, memory)
        # Store on primary_data so compute_trade_levels() can use it too
        if hist_data and primary_data:
            primary_data["poly_hist"] = hist_data
            # Also store on the live cache so format_msg sees it
            if primary_symbol in _price_cache:
                _price_cache[primary_symbol]["poly_hist"] = hist_data
        # Rate-limit: only sleep when a fresh API call was actually made
        if hist_data.get("fetched_fresh"):
            time.sleep(13)

        if hist_data:
            h_lines = ["━━━ HISTORICAL CONTEXT (2 years of daily bars) ━━━"]
            if hist_data.get("high_52w") and hist_data.get("low_52w"):
                h_lines.append(
                    f"52-week range: {hist_data['low_52w']} – {hist_data['high_52w']} | "
                    f"Price vs 52w high: {hist_data['pct_from_52h']:+.1f}% | "
                    f"vs 52w low: +{hist_data['pct_from_52l']:.1f}%"
                )
            if hist_data.get("ema_200_trend"):
                h_lines.append(f"Long-term trend: {hist_data['ema_200_trend']}")
            if hist_data.get("hist_vol_pct"):
                h_lines.append(
                    f"Historical daily volatility (ATR%): {hist_data['hist_vol_pct']:.2f}% "
                    f"— use this to calibrate stop sizing"
                )
            if hist_data.get("at_hist_level"):
                h_lines.append(f"⚠️ KEY LEVEL: {hist_data['at_hist_level']}")
            key_lvls = hist_data.get("key_levels") or []
            if key_lvls:
                kl_str = " | ".join(
                    f"{kl['type']} {kl['price']} ({kl['count']} touches)"
                    for kl in key_lvls[:4]
                )
                h_lines.append(f"Historical S/R zones: {kl_str}")
            hist_block = "\n" + "\n".join(h_lines)

    # ── Fetch new supplementary data ─────────────────────────────────────────
    # Fetched here so they're available for the AI prompt AND signal_confidence().
    # Each result is stored on primary_data so downstream callers can see them.
    options_data   = {}
    yields_data    = {}
    short_data     = {}
    premarket_data = {}

    if POLYGON_KEY and primary_symbol and primary_data:
        # Options flow (SPY/QQQ only)
        options_data = fetch_options_flow(primary_symbol, memory)
        if options_data.get("fetched_fresh"):
            time.sleep(2)
        if options_data:
            primary_data["options_flow"] = options_data
            if primary_symbol in _price_cache:
                _price_cache[primary_symbol]["options_flow"] = options_data

        # Treasury yields (global, any symbol)
        yields_data = fetch_treasury_yields(memory)
        if yields_data.get("fetched_fresh"):
            time.sleep(2)
        if yields_data:
            primary_data["treasury_yields"] = yields_data
            if primary_symbol in _price_cache:
                _price_cache[primary_symbol]["treasury_yields"] = yields_data

        # Short data (SPY/QQQ only)
        short_data = fetch_short_data(primary_symbol, memory)
        if short_data.get("fetched_fresh"):
            time.sleep(2)
        if short_data:
            primary_data["short_data"] = short_data
            if primary_symbol in _price_cache:
                _price_cache[primary_symbol]["short_data"] = short_data

        # Pre-market / after-hours
        premarket_data = fetch_premarket_data(primary_symbol, memory)
        if premarket_data.get("fetched_fresh"):
            time.sleep(2)
        if premarket_data:
            primary_data["premarket_data"] = premarket_data
            if primary_symbol in _price_cache:
                _price_cache[primary_symbol]["premarket_data"] = premarket_data

    # ── Options context for AI ────────────────────────────────────────────────
    options_block = ""
    if options_data and options_data.get("pcr") is not None:
        o_lines = ["━━━ OPTIONS MARKET (Polygon live chain) ━━━"]
        o_lines.append(f"Put/Call Ratio: {options_data['pcr']:.2f} — {options_data.get('pcr_label', '')}")
        if options_data.get("iv_avg"):
            o_lines.append(
                f"Implied Volatility (avg): {options_data['iv_avg']:.1%} — "
                f"market pricing in {options_data['iv_avg'] * 100:.1f}% annual move"
            )
        if options_data.get("unusual_summary"):
            o_lines.append(f"Unusual activity: {options_data['unusual_summary']}")
        if options_data.get("unusual_calls"):
            o_lines.append(f"Notable calls: {', '.join(options_data['unusual_calls'][:3])}")
        if options_data.get("unusual_puts"):
            o_lines.append(f"Notable puts: {', '.join(options_data['unusual_puts'][:3])}")
        if options_data.get("gamma_bias"):
            o_lines.append(f"Gamma exposure: {options_data['gamma_bias']}")
        options_block = "\n" + "\n".join(o_lines)

    # ── Treasury yields context for AI ───────────────────────────────────────
    yields_block = ""
    if yields_data and yields_data.get("y10") is not None:
        y_lines = ["━━━ TREASURY YIELDS (macro context) ━━━"]
        y2  = yields_data.get("y2")
        y10 = yields_data.get("y10")
        y30 = yields_data.get("y30")
        if y2 and y10:
            y30_part = f" | 30Y: {y30:.2f}%" if y30 else ""
            y_lines.append(f"2Y: {y2:.2f}% | 10Y: {y10:.2f}%{y30_part}")
            y_lines.append(
                f"2Y-10Y spread: {yields_data['spread_2_10']:+.2f}% — "
                f"{yields_data.get('curve_label', '')}"
            )
        if yields_data.get("equity_signal"):
            y_lines.append(f"Equity signal: {yields_data['equity_signal'].replace('_', ' ')}")
        if yields_data.get("gold_signal"):
            y_lines.append(f"Gold signal: {yields_data['gold_signal'].replace('_', ' ')}")
        yields_block = "\n" + "\n".join(y_lines)

    # ── Short data context for AI ─────────────────────────────────────────────
    short_block = ""
    if short_data and (short_data.get("days_to_cover") is not None
                       or short_data.get("short_vol_ratio") is not None):
        s_lines = ["━━━ SHORT DATA ━━━"]
        if short_data.get("short_label"):
            s_lines.append(short_data["short_label"])
        if short_data.get("short_vol_label"):
            s_lines.append(short_data["short_vol_label"])
        if short_data.get("squeeze_setup"):
            s_lines.append(
                "⚠️ SQUEEZE SETUP: High short interest + rising price — potential short squeeze"
            )
        short_block = "\n" + "\n".join(l for l in s_lines if l)

    # ── Pre-market context for AI ─────────────────────────────────────────────
    pm_block = ""
    if premarket_data and premarket_data.get("gap_note"):
        pm_block = f"\n━━━ PRE-MARKET / AFTER-HOURS ━━━\n{premarket_data['gap_note']}"

    # ── Memory lessons digest ─────────────────────────────────────────────────
    lessons_str  = build_lessons_digest(memory, primary_symbol) if memory else ""
    memory_block = f"\n\n━━━ BOT LEARNING (past signal outcomes) ━━━\n{lessons_str}" if lessons_str else ""

    prim_name = friendly(primary_symbol) if primary_symbol else "the most affected asset"
    prim_price = primary_data.get("price", "n/a")
    prim_move  = f"{primary_data['move']:+.2f}%" if primary_data.get("move") is not None else "n/a"

    prompt = f"""You are a senior institutional trading analyst with deep expertise in macro, technical analysis, order flow, and risk management.

━━━ HEADLINE ━━━
{title}
Signal type: {signal_type} | Market reaction so far: {reaction}

━━━ PRIMARY INSTRUMENT ━━━
{prim_name} | Price: {prim_price} | Today: {prim_move}

{tech_block}

━━━ MACRO CONTEXT ━━━
Market fear: {fear_str}
Dollar Index: {dxy_str}
High-impact events today: {cal_str}
{macro_note}
{polygon_context}
{hist_block}
{options_block}
{yields_block}
{short_block}
{pm_block}
{memory_block}

━━━ OTHER INSTRUMENTS (context) ━━━
{other_str}

━━━ YOUR TASK ━━━
Analyze this headline for {prim_name}. Synthesize ALL data above — especially the real Polygon order flow and historical context — into one unified view:

1. FUNDAMENTAL: What does this news mean for this asset? Is the move justified or overreacted?
2. TECHNICAL: Do RSI, trend (all timeframes), VWAP, market structure CONFIRM or CONTRADICT the bias?
3. ORDER FLOW (most important): Does the real Polygon data agree with the direction?
   - Cumulative delta, buy%, imbalance, bar streak = directional conviction
   - Delta flip = momentum turning point (high-conviction entry signal)
   - Absorption = a level is being defended (can be FOR or AGAINST the trade)
   - Exhaustion = one side running out of fuel (context-dependent)
   - Block trades = institutional footprint (smart money direction)
   - Opening range breakout = structural confirmation for equities
   Conflicting order flow MUST lower IMPACT. Aligned order flow MUST raise IMPACT.
4. LOCATION: Is price at a favourable entry (discount for BUY, premium for SELL)? VPOC, VAH/VAL, S/R levels? Historical key zones?
5. HISTORICAL: Does the 52-week context support this trade? Is price near a major historical level?
5b. OPTIONS: If options data is present, use PCR and unusual activity to confirm or contradict the bias. Extreme PCR = strong sentiment signal. Unusual activity = smart money positioning.
5c. SHORT SQUEEZE: If squeeze setup is present and bias is BUY, flag as high conviction. If squeeze setup but bias is SELL, warn it could reverse sharply.
6. PAST PERFORMANCE: If signal memory is provided, use the win-rate patterns to calibrate your IMPACT rating. High-confidence signals with historically predictive indicators should be rated higher.
7. RISK: The single biggest reason this trade fails.

IMPACT RATING GUIDE (be strict):
  1-3 = minor news, order flow neutral or contradicting, technicals mixed
  4-5 = moderate news, partial alignment
  6-7 = strong news catalyst + order flow and technicals confirming
  8-9 = major catalyst + full alignment (delta, imbalance, structure, VWAP all agree)
  10  = rare — systemic event + every indicator aligned

RULES:
- Always use full plain-English names. NEVER use tickers (QQQ, SPX, GC, CL, DXY, NQ etc.)
- ENTRY, STOP, TARGET are for {prim_name} ONLY — numbers only, no words
- REASON must cover: (1) news catalyst, (2) order flow read (reference specific metrics), (3) exact trade logic with key level

Return EXACTLY this format, no extra text:
BIAS: BUY / SELL / NEUTRAL
IMPACT: [1-10]
AFFECTS: [full plain-English names of affected instruments]
REASON: [3 sentences max: catalyst + order flow + trade logic]
ENTRY: [price or range for {prim_name} only — numbers only]
STOP: [stop loss — single number]
TARGET: [profit target — single number or range]
WATCH: [the one thing that would invalidate this trade]"""

    _ai_last_err = None
    for _ai_attempt in range(3):
        try:
            res = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            return res.content[0].text
        except Exception as e:
            _ai_last_err = e
            print(f"  AI error (attempt {_ai_attempt + 1}/3): {type(e).__name__}: {e}")
            if _ai_attempt < 2:
                time.sleep(4 * (_ai_attempt + 1))  # 4s then 8s before retrying
    print(f"  All AI retries exhausted.")
    return "BIAS: NEUTRAL\nIMPACT: 5\nAFFECTS: unknown\nREASON: AI unavailable\nENTRY: n/a\nSTOP: n/a\nTARGET: n/a\nWATCH: n/a"

def parse_ai(ai_text: str) -> dict:
    """Extract AI response fields with line-anchored regex.

    Old implementation used `text.split("KEY:")` which collided when a key
    name appeared inside another field's value (e.g. AI writes "TARGET: 100"
    inside REASON, then TARGET parses the wrong text). The line-anchored
    regex below only matches a key at the start of a line, so casual
    mentions inside values are ignored.
    """
    fields  = {"BIAS": "", "IMPACT": "5", "AFFECTS": "",
               "REASON": "", "ENTRY": "", "STOP": "", "TARGET": "", "WATCH": ""}
    key_alt = "|".join(fields.keys())
    pattern = rf'^\s*({key_alt}):\s*(.*?)(?=\n\s*(?:{key_alt}):|\Z)'
    for match in re.finditer(pattern, ai_text, re.MULTILINE | re.DOTALL | re.IGNORECASE):
        key = match.group(1).upper()
        if key in fields:
            fields[key] = match.group(2).strip()
    return fields

def parse_impact(raw: str) -> int:
    try:
        m = re.search(r'\d+', raw)
        return max(1, min(10, int(m.group()))) if m else 5
    except Exception:
        return 5

def compute_rr(entry_str: str, stop_str: str, target_str: str, bias: str) -> str:
    """Return a formatted R:R string like '1:2.4', or '' if inputs are invalid."""
    try:
        # Entry — handle range like "1990.0-2003.0"
        parts = entry_str.split("-")
        if len(parts) == 2:
            entry = (float(parts[0]) + float(parts[1])) / 2
        else:
            entry = float(entry_str)

        stop = float(stop_str)

        # Target — use the conservative end (closest to entry)
        tparts = target_str.split("-")
        if len(tparts) == 2:
            t1, t2 = float(tparts[0]), float(tparts[1])
            target = min(t1, t2) if bias == "BUY" else max(t1, t2)
        else:
            target = float(target_str)

        risk   = abs(entry - stop)
        reward = abs(target - entry)
        if risk <= 0:
            return ""
        rr = reward / risk
        return f"1:{rr:.1f}"
    except Exception:
        return ""

# ─────────────────────────────
# TICKER REPLACER
# ─────────────────────────────
TICKER_REPLACEMENTS = {
    "QQQ":    "US Tech 100 (Nasdaq)",
    "SPX":    "S&P 500",
    "SPY":    "S&P 500",
    "ES":     "S&P 500",
    "NQ":     "US Tech 100 (Nasdaq)",
    "ALI":    "Aluminium (ALI/USD)",
    "CL":     "Crude Oil (WTI)",
    "WTI":    "Crude Oil (WTI)",
    "GC":     "Gold (XAU/USD)",
    "XAU":    "Gold (XAU/USD)",
    "DXY":    "US Dollar Index",
    "GBP":    "GBP/USD (Pound)",
    "EUR":    "EUR/USD (Euro)",
    "EURUSD": "EUR/USD (Euro)",
    "GBPUSD": "GBP/USD (Pound)",
    "BTCUSD": "Bitcoin (BTC/USD)",
    "ETHUSD": "Ethereum (ETH/USD)",
}

def replace_tickers(text: str) -> str:
    for ticker, name in TICKER_REPLACEMENTS.items():
        if ticker in ("GBP", "EUR"):
            # Negative lookahead: don't replace when followed by "/" so
            # "GBP/USD" stays intact but standalone "GBP" gets replaced.
            text = re.sub(rf'\b{ticker}\b(?!/)', name, text, flags=re.IGNORECASE)
        else:
            # Negative lookbehind: don't replace when preceded by "(" so
            # "Crude Oil (WTI)" or "Gold (XAU/USD)" are left untouched —
            # the ticker is already embedded inside its own friendly name.
            text = re.sub(rf'(?<!\()\b{ticker}\b', name, text, flags=re.IGNORECASE)
    return text

def sanitize(text: str) -> str:
    text = replace_tickers(text)
    return (
        text
        .replace("*", "")
        .replace("_", " ")
        .replace("`", "'")
        .replace("[", "(")
        .replace("]", ")")
    )

# ─────────────────────────────
# KEY LEVEL TRAP CHECK
# ─────────────────────────────
def is_at_trap_level(bias: str, primary_data: dict, threshold_pct: float = 0.4) -> tuple:
    """Detect signals that would buy at resistance or sell at support.

    Returns (is_trap, reason). Buying right at resistance (or selling right
    at support) is the classic retail trap — odds favour rejection. We block
    these even if the headline and AI both say BUY/SELL.

    threshold_pct: how close (in %) we consider "at" the level. 0.4% by default.
    """
    if bias not in ("BUY", "SELL"):
        return False, ""
    price = primary_data.get("price")
    if not price or price <= 0:
        return False, ""

    if bias == "BUY":
        r15 = primary_data.get("resistance")
        if r15 and r15 > price:
            distance_pct = (r15 - price) / price * 100
            if distance_pct <= threshold_pct:
                return True, f"BUY blocked — price ({price}) is {distance_pct:.2f}% below 15m resistance ({r15}). Risk of immediate rejection."
    else:  # SELL
        s15 = primary_data.get("support")
        if s15 and s15 < price:
            distance_pct = (price - s15) / price * 100
            if distance_pct <= threshold_pct:
                return True, f"SELL blocked — price ({price}) is {distance_pct:.2f}% above 15m support ({s15}). Risk of immediate bounce."
    return False, ""

# ─────────────────────────────
# TRADE LEVELS (programmatic — anchored to support/resistance)
# ─────────────────────────────
def compute_trade_levels(bias: str, primary_data: dict,
                         hist_data: dict = None) -> dict:
    """Compute Entry/SL/TP using full Polygon + yfinance + historical confluence.

    Level sources (all used simultaneously):
      Polygon real data  : VPOC, VAH, VAL, VWAP, swing S/R, session H/L, OR H/L
      yfinance 15m       : support, resistance
      yfinance daily     : daily_support, daily_resistance
      Historical (2yr)   : 52-week high/low, historical key S/R cluster levels
      volatility         : ATR for buffer sizing

    BUY logic:
      Entry  : tight band around current price (±0.15%)
      SL     : highest support meaningfully below price — prioritises real
               Polygon levels (VAL, swing support, VWAP breakdown) over yfinance;
               closest valid level = tightest stop = best R/R
      TP1    : nearest upside confluence ≥1.5R away, priority order:
               VPOC (price magnet) → VWAP (mean-reversion target) → VAH
               → OR high → 15m resistance → Polygon swing resistance
               → daily resistance → historical S/R → 52-week high → session high
      TP2    : next confluence level above TP1, or 2.5R extension

    SELL logic: mirror of BUY.

    Always enforces SL on loss side, TP on profit side, R:R ≥ 1.5:1.
    Returns {} if bias is non-directional or price is missing.
    """
    price = primary_data.get("price")
    if not price or price <= 0 or bias not in ("BUY", "SELL"):
        return {}

    # ── yfinance levels ──────────────────────────────────────────────────────
    s15 = primary_data.get("support")
    r15 = primary_data.get("resistance")
    sd  = primary_data.get("daily_support")
    rd  = primary_data.get("daily_resistance")
    atr = primary_data.get("atr")

    # ── Polygon real levels ──────────────────────────────────────────────────
    poly  = primary_data.get("polygon_flow") or {}
    vpoc  = poly.get("vpoc")           # most-traded price — strongest magnet
    vah   = poly.get("vah")            # value area high — institutional resistance
    val   = poly.get("val")            # value area low  — institutional support
    vwap  = poly.get("vwap")           # session VWAP — dynamic S/R and mean target
    p_res = poly.get("key_resistance") # Polygon swing high resistance
    p_sup = poly.get("key_support")    # Polygon swing low support
    s_hi  = poly.get("session_high")   # session ceiling
    s_lo  = poly.get("session_low")    # session floor
    or_hi = poly.get("or_high")        # opening range high (15-min)
    or_lo = poly.get("or_low")         # opening range low  (15-min)

    # ── Historical levels (2-year daily bars) ────────────────────────────────
    hist  = hist_data or (primary_data.get("poly_hist") or {})
    h52_hi = hist.get("high_52w")      # 52-week high — major resistance zone
    h52_lo = hist.get("low_52w")       # 52-week low  — major support zone
    # Historical key levels: sorted cluster of swing S/R over 2 years
    hist_key_levels = hist.get("key_levels") or []
    # Extract the top historical R and S levels separately
    h_key_res = next((kl["price"] for kl in hist_key_levels if kl["type"] == "R"), None)
    h_key_sup = next((kl["price"] for kl in hist_key_levels if kl["type"] == "S"), None)

    # ── IV-aware stop sizing ─────────────────────────────────────────────────
    # If options data shows high implied volatility, widen stops so they're
    # not clipped by normal volatility noise.
    iv_avg       = (primary_data.get("options_flow") or {}).get("iv_avg")
    iv_multiplier = 1.0
    if iv_avg is not None:
        if iv_avg > 0.40:   iv_multiplier = 1.35   # very high IV → much wider stops
        elif iv_avg > 0.25: iv_multiplier = 1.15   # elevated IV → slightly wider stops

    # ── ATR-aware buffer ─────────────────────────────────────────────────────
    # At least 0.15% of price, scales with volatility so stops breathe in
    # fast-moving markets (futures, volatile ETFs) without being reckless.
    buf = max(price * 0.0015, (atr * 0.5) if atr else 0) * iv_multiplier

    # ── Decimal precision ─────────────────────────────────────────────────────
    if   price >= 1000: dec = 1
    elif price >= 10:   dec = 2
    elif price >= 1:    dec = 4
    else:               dec = 5
    fmt = lambda n: f"{n:.{dec}f}"

    # ── Helper: filter levels to a valid band ────────────────────────────────
    def _below(lvl, min_gap=0.002):
        """Level is meaningfully below price (at least min_gap %)."""
        return lvl is not None and lvl < price * (1 - min_gap)

    def _above(lvl, min_gap=0.002):
        """Level is meaningfully above price (at least min_gap %)."""
        return lvl is not None and lvl > price * (1 + min_gap)

    # ════════════════════════════════════════════════════════════════════════
    if bias == "BUY":
        entry_lo = price * 0.999
        entry_hi = price * 1.0015

        # ── SL: best support below price (take HIGHEST = tightest valid stop)
        # Priority: VAL > Polygon swing > VWAP breakdown > 15m S/R > daily S/R
        #           > historical key support > 52-week low
        sl_pool = [
            lvl for lvl in [val, p_sup, vwap, or_lo, s15, sd, h_key_sup, h52_lo]
            if _below(lvl)
        ]
        sl = (max(sl_pool) - buf) if sl_pool else (price * 0.99)

        risk = price - sl
        if risk <= 0:
            sl   = price * 0.99
            risk = price - sl

        min_tp = price + risk * 1.5    # floor: 1.5R minimum — ensures worthwhile R/R
        max_tp = price + risk * 4.0    # ceiling: 4R cap

        # ── Two-tier TP selection ─────────────────────────────────────────────
        # Tier 1 — Polygon magnets (VPOC, VWAP, VAH/VAL): highest probability
        #   targets because they represent where real volume/interest sits.
        #   Picked first regardless of distance vs structural levels.
        # Tier 2 — Structural S/R (OR high, swing, daily, session, historical):
        #   TP1 fallback and TP2 extension. Historical 52w high and key resistance
        #   add long-term confluence for extended moves.
        magnet_tp = sorted([
            lvl - buf for lvl in [vpoc, vwap, vah]
            if lvl is not None and (lvl - buf) >= min_tp and (lvl - buf) <= max_tp
        ])
        struct_tp = sorted([
            lvl - buf for lvl in [or_hi, r15, p_res, rd, h_key_res, h52_hi, s_hi]
            if lvl is not None and (lvl - buf) >= min_tp and (lvl - buf) <= max_tp
        ])
        all_tp = magnet_tp + [t for t in struct_tp if t not in magnet_tp]

        # TP1: best magnet first, else nearest structural, else 1.5R hard fallback
        tp1 = magnet_tp[0] if magnet_tp else (struct_tp[0] if struct_tp else price + risk * 1.5)
        # TP2: next level beyond TP1 from either pool, else 2.5R
        tp2_pool = sorted([t for t in all_tp if t > tp1 * 1.003])
        tp2      = tp2_pool[0] if tp2_pool else min(price + risk * 2.5, max_tp)

        if tp2 > tp1 * 1.003:
            target = f"{fmt(tp1)}-{fmt(tp2)}"
        else:
            target = fmt(tp1)

        return {"entry": f"{fmt(entry_lo)}-{fmt(entry_hi)}", "stop": fmt(sl), "target": target}

    # ════════════════════════════════════════════════════════════════════════
    # SELL
    entry_lo = price * 0.9985
    entry_hi = price * 1.001

    # ── SL: best resistance above price (take LOWEST = tightest valid stop)
    # Priority: VAH > Polygon swing > VWAP reclaim > 15m S/R > daily S/R
    #           > historical key resistance > 52-week high
    sl_pool = [
        lvl for lvl in [vah, p_res, vwap, or_hi, r15, rd, h_key_res, h52_hi]
        if _above(lvl)
    ]
    sl = (min(sl_pool) + buf) if sl_pool else (price * 1.01)

    risk = sl - price
    if risk <= 0:
        sl   = price * 1.01
        risk = sl - price

    min_tp = price - risk * 1.5    # floor: 1.5R minimum — ensures worthwhile R/R
    max_tp = price - risk * 4.0    # ceiling: 4R cap (going down)

    # ── Two-tier TP selection (mirror of BUY) ─────────────────────────────────
    magnet_tp = sorted([
        lvl + buf for lvl in [vpoc, vwap, val]
        if lvl is not None and (lvl + buf) <= min_tp and (lvl + buf) >= max_tp
    ], reverse=True)   # highest = nearest to price for SELL
    struct_tp = sorted([
        lvl + buf for lvl in [or_lo, s15, p_sup, sd, h_key_sup, h52_lo, s_lo]
        if lvl is not None and (lvl + buf) <= min_tp and (lvl + buf) >= max_tp
    ], reverse=True)
    all_tp = magnet_tp + [t for t in struct_tp if t not in magnet_tp]

    tp1 = magnet_tp[0] if magnet_tp else (struct_tp[0] if struct_tp else price - risk * 1.5)
    tp2_pool = sorted([t for t in all_tp if t < tp1 * 0.997], reverse=True)
    tp2      = tp2_pool[0] if tp2_pool else max(price - risk * 2.5, max_tp)

    if tp1 > tp2 * 1.003:
        target = f"{fmt(tp2)}-{fmt(tp1)}"   # show lower-higher (natural reading)
    else:
        target = fmt(tp1)

    return {"entry": f"{fmt(entry_lo)}-{fmt(entry_hi)}", "stop": fmt(sl), "target": target}

# ─────────────────────────────
# FORMAT MESSAGE
# ─────────────────────────────
def format_msg(title, reaction, base_score, moves, primary_symbol,
               ai_text, signal_type, calendar_events,
               state, src_count=1, memory=None) -> str:

    ai     = parse_ai(ai_text)
    bias   = sanitize(ai["BIAS"]).strip().upper()
    impact = parse_impact(ai["IMPACT"])

    bonus       = max(0, (impact - 5) * 2)
    final_score = min(100, base_score + bonus)

    if bias == "BUY":
        action = f"✅ BUY — {friendly(primary_symbol)}"
    elif bias == "SELL":
        action = f"🔴 SELL — {friendly(primary_symbol)}"
    else:
        action = f"⚠️ WATCH — {friendly(primary_symbol)}"

    # Live moves (primary instruments only)
    if moves:
        moves_lines = "\n".join([
            f"  {friendly(s)}: {d['move']:+.2f}% | "
            f"{'$' if s not in FOREX_SYMBOLS else ''}{d['price']}"
            for s, d in moves.items()
            if s in ASSET_MAP
        ])
    else:
        moves_lines = "  prices unavailable"

    # Trade levels — computed programmatically from price + S/R so they
    # always respect the bot's own technicals (AI levels often violated them).
    # AI levels are used only as a last-resort fallback if computation fails
    # (e.g. missing price data on a directional signal).
    primary_data = dict(moves.get(primary_symbol, {}))  # copy so we can annotate safely

    # Pull historical data from cache (set by analyze()) — pass to trade levels
    hist_data      = primary_data.get("poly_hist") or {}
    computed_levels = compute_trade_levels(bias, primary_data, hist_data=hist_data)
    if computed_levels:
        entry  = computed_levels["entry"]
        stop   = computed_levels["stop"]
        target = computed_levels["target"]
        has_levels = True
    else:
        entry  = sanitize(ai.get("ENTRY", "n/a"))
        stop   = sanitize(ai.get("STOP",  "n/a"))
        target = sanitize(ai.get("TARGET","n/a"))
        placeholders = {"n/a", "", "n.a.", "tbd", "tba", "unknown", "none", "?", "??"}
        def _has_real_level(v: str) -> bool:
            return v.strip().lower() not in placeholders and bool(re.search(r'\d', v))
        has_levels = all(_has_real_level(v) for v in [entry, stop, target])
    rr_str = compute_rr(entry, stop, target, bias) if has_levels and bias in ("BUY", "SELL") else ""
    rr_display = f"  R:R Ratio: {rr_str}\n" if rr_str else ""
    levels_section = (
        f"📌 *Trade levels ({friendly(primary_symbol)}):*\n"
        f"  Entry:     {entry}\n"
        f"  Stop Loss: {stop}\n"
        f"  Target:    {target}\n"
        f"{rr_display}\n"
    ) if has_levels else ""

    # Calendar
    cal_section = ""
    if calendar_events:
        cal_items   = " | ".join([f"{e['title']} ({e['currency']})" for e in calendar_events[:3]])
        cal_section = f"📅 *High-impact events today:* {cal_items}\n\n"

    # ── 5m chart ─────────────────────────────────────────────────────────────
    fivem_lines = []
    if primary_data.get("5m_rsi") is not None:
        fivem_lines.append(f"  RSI(14): {rsi_label(primary_data['5m_rsi'])}")
    if primary_data.get("5m_trend"):
        fivem_lines.append(f"  Trend:   {primary_data['5m_trend']}")
    if primary_data.get("5m_support") is not None:
        fivem_lines.append(f"  Support: {primary_data['5m_support']}")
    if primary_data.get("5m_resistance") is not None:
        fivem_lines.append(f"  Resist:  {primary_data['5m_resistance']}")
    fvg_5m = primary_data.get("5m_fvg") or {}
    if fvg_5m:
        fvg_emoji = "🟢" if fvg_5m["type"] == "bullish" else "🔴"
        fivem_lines.append(f"  FVG:     {fvg_emoji} {fvg_label(fvg_5m)}")
    fib_5m = primary_data.get("5m_fib") or {}
    if fib_5m:
        fivem_lines.append(
            f"  EQ(0.5): {fib_5m['equilibrium']} | "
            f"0.618: {fib_5m['fib_0618']} | nearest: {fib_5m['nearest_level']} "
            f"({fib_5m['proximity_pct']}% away)"
        )
    ob_5m = primary_data.get("5m_ob") or {}
    if ob_5m.get("bullish"):
        b = ob_5m["bullish"]
        fivem_lines.append(f"  Bull OB: {b['low']} – {b['high']} ({b['status']})")
    if ob_5m.get("bearish"):
        br = ob_5m["bearish"]
        fivem_lines.append(f"  Bear OB: {br['low']} – {br['high']} ({br['status']})")
    ms_5m = primary_data.get("5m_market_structure") or {}
    if ms_5m:
        fivem_lines.append(f"  Struct:  {ms_5m.get('structure', '')}")
        if ms_5m.get("event"):
            fivem_lines.append(f"  ⚡ {ms_5m['event']}")
    if primary_data.get("5m_candle_pattern"):
        fivem_lines.append(f"  Candle:  {primary_data['5m_candle_pattern']}")
    if primary_data.get("5m_liquidity_sweep"):
        fivem_lines.append(f"  ⚡ {primary_data['5m_liquidity_sweep']}")
    of_5m = primary_data.get("5m_order_flow") or {}
    if of_5m:
        _last5 = f" | last={of_5m['delta']:+,}" if "delta" in of_5m else ""
        fivem_lines.append(f"  Delta:   {of_5m['of_bias'].upper()} (cum={of_5m['cum_delta']:+,}{_last5})")
        if of_5m.get("delta_flip"):
            fivem_lines.append(f"  ⚡ {of_5m['delta_flip']}")
    fivem_section = "🕐 *5-min chart:*\n" + "\n".join(fivem_lines) + "\n\n" if fivem_lines else ""

    # ── 15m chart ────────────────────────────────────────────────────────────
    ta_lines = []
    if primary_data.get("rsi") is not None:
        ta_lines.append(f"  RSI(14):    {rsi_label(primary_data['rsi'])}")
    ta_lines.append(f"  MACD:       {macd_label(primary_data.get('macd'), primary_data.get('macd_signal'))}")
    ta_lines.append(f"  Trend:      {primary_data.get('trend', 'n/a')}")
    if primary_data.get("vol_ratio") is not None:
        ta_lines.append(f"  Volume:     {volume_label(primary_data['vol_ratio'])}")
    if primary_data.get("support") is not None:
        ta_lines.append(f"  Support:    {primary_data['support']}")
    if primary_data.get("resistance") is not None:
        ta_lines.append(f"  Resistance: {primary_data['resistance']}")
    divergence = primary_data.get("divergence", "")
    if divergence:
        emoji = "🟢" if divergence == "bullish" else "🔴"
        ta_lines.append(f"  Divergence: {emoji} {divergence.upper()} (RSI signaling possible reversal)")
    fib_15 = primary_data.get("fib") or {}
    if fib_15:
        ta_lines.append(
            f"  EQ(0.5):    {fib_15['equilibrium']} | "
            f"0.382: {fib_15['fib_0382']} | 0.618: {fib_15['fib_0618']}"
        )
        ta_lines.append(
            f"  Nearest fib: {fib_15['nearest_level']} @ {fib_15['nearest_price']} "
            f"({fib_15['proximity_pct']}% away)"
        )
    ob_15 = primary_data.get("ob") or {}
    if ob_15.get("bullish"):
        b = ob_15["bullish"]
        ta_lines.append(f"  Bull OB:    {b['low']} – {b['high']} ({b['status']})")
    if ob_15.get("bearish"):
        br = ob_15["bearish"]
        ta_lines.append(f"  Bear OB:    {br['low']} – {br['high']} ({br['status']})")
    vwap_d = primary_data.get("vwap") or {}
    if vwap_d:
        ta_lines.append(f"  VWAP:       {vwap_d.get('vwap')} ({vwap_d.get('position', '')})")
        if vwap_d.get("vwap_u1"):
            ta_lines.append(
                f"  VWAP σ:     +1σ={vwap_d['vwap_u1']} / -1σ={vwap_d['vwap_l1']}"
            )
    ms_15 = primary_data.get("market_structure") or {}
    if ms_15:
        ta_lines.append(f"  Struct:     {ms_15.get('structure', '')}")
        if ms_15.get("event"):
            ta_lines.append(f"  ⚡ {ms_15['event']}")
    price_15 = primary_data.get("price")
    if price_15 and fib_15:
        pd_15 = premium_discount_label(price_15, fib_15)
        if pd_15:
            ta_lines.append(f"  P/D Zone:   {pd_15}")
    if primary_data.get("candle_pattern"):
        ta_lines.append(f"  Candle:     {primary_data['candle_pattern']}")
    if primary_data.get("liquidity_sweep"):
        ta_lines.append(f"  ⚡ {primary_data['liquidity_sweep']}")
    of_15 = primary_data.get("order_flow") or {}
    if of_15:
        _last15 = f" | last={of_15['delta']:+,}" if "delta" in of_15 else ""
        ta_lines.append(f"  Delta:      {of_15['of_bias'].upper()} (cum={of_15['cum_delta']:+,}{_last15})")
        if of_15.get("delta_flip"):
            ta_lines.append(f"  ⚡ {of_15['delta_flip']}")
    ta_section = "📈 *Technical (15m):*\n" + "\n".join(ta_lines) + "\n\n" if ta_lines else ""

    # ── 1h chart ─────────────────────────────────────────────────────────────
    hourly_section = ""
    h_trend = primary_data.get("hourly_trend", "")
    h_rsi   = primary_data.get("hourly_rsi")
    if h_trend or h_rsi is not None:
        h_lines = []
        if h_trend:
            t15 = primary_data.get("trend", "")
            agree_marker = ""
            if "Uptrend" in t15 and "Uptrend" in h_trend:
                agree_marker = " ✅ aligned with 15m"
            elif "Downtrend" in t15 and "Downtrend" in h_trend:
                agree_marker = " ✅ aligned with 15m"
            elif (("Uptrend" in t15 and "Downtrend" in h_trend) or
                  ("Downtrend" in t15 and "Uptrend" in h_trend)):
                agree_marker = " ⚠️ disagrees with 15m"
            h_lines.append(f"  Trend:   {h_trend}{agree_marker}")
        if h_rsi is not None:
            h_lines.append(f"  RSI(14): {rsi_label(h_rsi)}")
        if h_lines:
            hourly_section = "⏱ *1h chart:*\n" + "\n".join(h_lines) + "\n\n"

    # ── 4h chart ─────────────────────────────────────────────────────────────
    fourh_section = ""
    fh_trend = primary_data.get("4h_trend", "")
    fh_rsi   = primary_data.get("4h_rsi")
    if fh_trend or fh_rsi is not None:
        fh_lines = []
        if fh_trend:
            t15 = primary_data.get("trend", "")
            agree_marker = ""
            if "Uptrend" in t15 and "Uptrend" in fh_trend:
                agree_marker = " ✅ aligned with 15m"
            elif "Downtrend" in t15 and "Downtrend" in fh_trend:
                agree_marker = " ✅ aligned with 15m"
            elif (("Uptrend" in t15 and "Downtrend" in fh_trend) or
                  ("Downtrend" in t15 and "Uptrend" in fh_trend)):
                agree_marker = " ⚠️ disagrees with 15m"
            fh_lines.append(f"  Trend:   {fh_trend}{agree_marker}")
        if fh_rsi is not None:
            fh_lines.append(f"  RSI(14): {rsi_label(fh_rsi)}")
        if primary_data.get("4h_support") is not None:
            fh_lines.append(f"  Support: {primary_data['4h_support']}")
        if primary_data.get("4h_resistance") is not None:
            fh_lines.append(f"  Resist:  {primary_data['4h_resistance']}")
        fvg_4h = primary_data.get("4h_fvg") or {}
        if fvg_4h:
            fvg_emoji = "🟢" if fvg_4h["type"] == "bullish" else "🔴"
            fh_lines.append(f"  FVG:     {fvg_emoji} {fvg_label(fvg_4h)}")
        fib_4h = primary_data.get("4h_fib") or {}
        if fib_4h:
            fh_lines.append(
                f"  EQ(0.5): {fib_4h['equilibrium']} | "
                f"0.382: {fib_4h['fib_0382']} | 0.618: {fib_4h['fib_0618']}"
            )
            fh_lines.append(
                f"  Nearest: {fib_4h['nearest_level']} @ {fib_4h['nearest_price']} "
                f"({fib_4h['proximity_pct']}% away)"
            )
        ob_4h = primary_data.get("4h_ob") or {}
        if ob_4h.get("bullish"):
            b = ob_4h["bullish"]
            fh_lines.append(f"  Bull OB: {b['low']} – {b['high']} ({b['status']})")
        if ob_4h.get("bearish"):
            br = ob_4h["bearish"]
            fh_lines.append(f"  Bear OB: {br['low']} – {br['high']} ({br['status']})")
        ms_4h = primary_data.get("4h_market_structure") or {}
        if ms_4h:
            fh_lines.append(f"  Struct:  {ms_4h.get('structure', '')}")
            if ms_4h.get("event"):
                fh_lines.append(f"  ⚡ {ms_4h['event']}")
        if primary_data.get("4h_liquidity_sweep"):
            fh_lines.append(f"  ⚡ {primary_data['4h_liquidity_sweep']}")
        of_4h = primary_data.get("4h_order_flow") or {}
        if of_4h:
            _last4h = f" | last={of_4h['delta']:+,}" if "delta" in of_4h else ""
            fh_lines.append(f"  Delta:   {of_4h['of_bias'].upper()} (cum={of_4h['cum_delta']:+,}{_last4h})")
            if of_4h.get("delta_flip"):
                fh_lines.append(f"  ⚡ {of_4h['delta_flip']}")
        if fh_lines:
            fourh_section = "📊 *4h chart:*\n" + "\n".join(fh_lines) + "\n\n"

    # ── Daily chart ───────────────────────────────────────────────────────────
    daily_lines = []
    if primary_data.get("daily_trend"):
        daily_lines.append(f"  Trend:   {primary_data['daily_trend']}")
    if primary_data.get("daily_rsi") is not None:
        daily_lines.append(f"  RSI(14): {rsi_label(primary_data['daily_rsi'])}")
    if primary_data.get("daily_support") is not None:
        daily_lines.append(f"  Support: {primary_data['daily_support']}")
    if primary_data.get("daily_resistance") is not None:
        daily_lines.append(f"  Resist:  {primary_data['daily_resistance']}")
    fib_d = primary_data.get("daily_fib") or {}
    if fib_d:
        daily_lines.append(
            f"  EQ(0.5): {fib_d['equilibrium']} | "
            f"0.382: {fib_d['fib_0382']} | 0.618: {fib_d['fib_0618']}"
        )
        daily_lines.append(
            f"  TP targets: 1.272={fib_d['fib_1272']} | 1.618={fib_d['fib_1618']}"
        )
    ms_d = primary_data.get("daily_market_structure") or {}
    if ms_d:
        daily_lines.append(f"  Struct:  {ms_d.get('structure', '')}")
        if ms_d.get("event"):
            daily_lines.append(f"  ⚡ {ms_d['event']}")
    daily_section = "🗓 *Daily chart:*\n" + "\n".join(daily_lines) + "\n\n" if daily_lines else ""

    # ── Historical context section ─────────────────────────────────────────────
    hist_section = ""
    if hist_data and bias in ("BUY", "SELL"):
        h_lines = []
        h52_hi = hist_data.get("high_52w")
        h52_lo = hist_data.get("low_52w")
        if h52_hi and h52_lo:
            h_lines.append(
                f"  52w range:  {h52_lo} – {h52_hi} | "
                f"vs high: {hist_data.get('pct_from_52h', 0):+.1f}% | "
                f"vs low: +{hist_data.get('pct_from_52l', 0):.1f}%"
            )
        if hist_data.get("ema_200_trend"):
            h_lines.append(f"  Long-term:  {hist_data['ema_200_trend']}")
        if hist_data.get("hist_vol_pct"):
            h_lines.append(f"  Daily vol:  {hist_data['hist_vol_pct']:.2f}% ATR/price")
        if hist_data.get("at_hist_level"):
            h_lines.append(f"  ⚡ {hist_data['at_hist_level']}")
        if h_lines:
            hist_section = "📅 *Historical context (2yr):*\n" + "\n".join(h_lines) + "\n\n"

    # ── Options flow section ───────────────────────────────────────────────────
    options_section = ""
    opts = primary_data.get("options_flow") or {}
    if opts.get("pcr") is not None:
        opts_lines = ["📊 *Options Market:*"]
        opts_lines.append(f"  P/C Ratio: {opts['pcr']:.2f} — {opts.get('pcr_label', '')}")
        if opts.get("iv_avg"):
            opts_lines.append(f"  Impl. Vol:  {opts['iv_avg']:.1%}")
        if opts.get("unusual_summary"):
            opts_lines.append(f"  ⚡ {opts['unusual_summary']}")
        if opts.get("gamma_bias"):
            opts_lines.append(f"  Gamma:     {opts['gamma_bias']}")
        options_section = "\n".join(opts_lines) + "\n\n"

    # ── Short data section (interest + intraday volume) ───────────────────────
    short_section = ""
    short = primary_data.get("short_data") or {}
    if short.get("short_label") or short.get("short_vol_label"):
        short_lines = ["📉 *Short Data:*"]
        if short.get("short_label"):
            short_lines.append(f"  {short['short_label']}")
        if short.get("short_vol_label"):
            short_lines.append(f"  {short['short_vol_label']}")
        if short.get("squeeze_setup"):
            short_lines.append("  🚨 SQUEEZE SETUP — high short float, price rising")
        short_section = "\n".join(short_lines) + "\n\n"

    # ── Pre-market / after-hours section ─────────────────────────────────────
    premarket_section = ""
    pm = primary_data.get("premarket_data") or {}
    if pm.get("gap_note"):
        premarket_section = f"🌅 *Extended Hours:*\n  {pm['gap_note']}\n\n"

    # ── Treasury yields section (for equities) ────────────────────────────────
    yields_section = ""
    yields = primary_data.get("treasury_yields") or {}
    if yields.get("y10") is not None and primary_symbol in ("^GSPC", "QQQ"):
        y_lines = ["📈 *Treasury Yields:*"]
        y2  = yields.get("y2")
        y10 = yields.get("y10")
        if y2 and y10:
            y_lines.append(
                f"  2Y: {y2:.2f}% | 10Y: {y10:.2f}% | "
                f"Spread: {yields.get('spread_2_10', 0):+.2f}%"
            )
            y_lines.append(f"  Curve: {yields.get('curve_label', '')}")
        yields_section = "\n".join(y_lines) + "\n\n"

    # ── Memory / past performance section ─────────────────────────────────────
    memory_section = ""
    if memory and bias in ("BUY", "SELL"):
        digest = build_lessons_digest(memory, primary_symbol)
        if digest:
            # Condense for Telegram: strip the header line, show win rate + top insight
            dlines  = digest.split("\n")
            compact = [ln for ln in dlines if ln.strip()][:4]  # max 4 lines in message
            memory_section = "🧠 *Signal memory:*\n" + "\n".join(compact) + "\n\n"

    # Signal confidence
    conf_score, conf_total = signal_confidence(primary_data, bias)
    conf_bar     = confidence_bar(conf_score, conf_total)
    # Only show confidence for directional signals — WATCH/NEUTRAL always score 0 which is misleading
    conf_section = f"💪 *Confidence: {conf_score}/{conf_total} indicators aligned* |{conf_bar}|\n\n" if (conf_total > 0 and bias in ("BUY", "SELL")) else ""

    # ── Confidence bonus → final_score ───────────────────────────────────────
    # The 15-indicator system now directly affects signal strength, not just
    # display. Higher alignment = stronger signal = higher label.
    #   ≥70% aligned → +10 (HIGH CONVICTION boost)
    #   ≥55% aligned → +6
    #   ≥40% aligned → +3
    #   <40% aligned → +0 (signal still passed the gate at 35%, just no boost)
    if conf_total >= 5 and bias in ("BUY", "SELL"):
        conf_ratio = conf_score / conf_total
        conf_boost = 10 if conf_ratio >= 0.70 else 6 if conf_ratio >= 0.55 else 3 if conf_ratio >= 0.40 else 0
        final_score = min(100, final_score + conf_boost)

    # Streak — only displayed when current bias extends an existing streak
    streak_section = get_streak_label(state, primary_symbol, bias)
    if streak_section:
        streak_section += "\n"

    # VIX — market fear gauge
    vix_data   = _price_cache.get("^VIX")
    fg_section = ""
    if vix_data:
        vix_price = vix_data.get("price")
        if vix_price:
            note = vix_signal_note(vix_price, bias)
            fg_section = f"📊 *Market Fear (VIX):* {vix_label(vix_price)}{note}\n\n"

    # DXY context — shown for macro events, gold, and oil signals
    dxy_section = ""
    show_dxy = is_macro(title) or primary_symbol in {"GC=F", "CL=F"}
    if show_dxy and "DX-Y.NYB" in _price_cache:
        dxy = _price_cache["DX-Y.NYB"]
        dxy_section = (
            f"💵 *US Dollar Index (DXY):* {dxy['price']} ({dxy['move']:+.2f}% today)"
            f" | {dxy.get('trend', 'n/a')}\n\n"
        )

    # Polygon real order flow section (only when real data available — SPY/QQQ)
    poly_section = ""
    poly_flow = primary_data.get("polygon_flow") or {}
    if poly_flow.get("data_source") == "polygon_real":
        pn = poly_flow.get("proxy_note", "")
        header = f"📡 *Real Order Flow (Polygon){' — ' + pn if pn else ''}:*"
        poly_lines = [header]

        # ── Delta & pressure ─────────────────────────────────────────────────
        delta = poly_flow.get("cum_delta")
        if delta is not None:
            d_sign     = "+" if delta >= 0 else ""
            bias_emoji = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
            poly_lines.append(f"  {bias_emoji} Cum Delta:  {d_sign}{delta:,.0f}")
        buy_pct = poly_flow.get("buy_pct")
        if buy_pct is not None:
            bar = "█" * (buy_pct // 10) + "░" * (10 - buy_pct // 10)
            lbl = "buyers dominant" if buy_pct >= 55 else "sellers dominant" if buy_pct <= 45 else "balanced"
            poly_lines.append(f"  Pressure:   {buy_pct}% buy |{bar}| {lbl}")

        # ── VWAP ─────────────────────────────────────────────────────────────
        vwap_v = poly_flow.get("vwap")
        pvwap  = poly_flow.get("price_vs_vwap", "")
        if vwap_v is not None:
            poly_lines.append(f"  VWAP:       {vwap_v} → price {pvwap}")

        # ── Volume Profile ────────────────────────────────────────────────────
        vpoc = poly_flow.get("vpoc")
        vah  = poly_flow.get("vah")
        val  = poly_flow.get("val")
        if vpoc is not None:
            poly_lines.append(f"  VPOC:       {vpoc}  ← most traded price")
        if vah is not None and val is not None:
            poly_lines.append(f"  Value Area: {val} – {vah}  (70% of vol)")

        # ── Volume spike ─────────────────────────────────────────────────────
        vol_spike = poly_flow.get("vol_spike")
        if vol_spike and vol_spike >= 1.5:
            poly_lines.append(f"  Vol Spike:  {vol_spike}× average ⚡")

        # ── Session levels ────────────────────────────────────────────────────
        s_hi = poly_flow.get("session_high")
        s_lo = poly_flow.get("session_low")
        if s_hi is not None and s_lo is not None:
            poly_lines.append(f"  Session:    {s_lo} – {s_hi}  (H/L)")

        or_st = poly_flow.get("or_status", "")
        if or_st:
            poly_lines.append(f"  Open Rng:   {or_st}")

        # ── Structure ────────────────────────────────────────────────────────
        mkt = poly_flow.get("mkt_structure", "")
        if mkt:
            poly_lines.append(f"  Structure:  {mkt}")
        kr = poly_flow.get("key_resistance")
        ks = poly_flow.get("key_support")
        if kr is not None and ks is not None:
            poly_lines.append(f"  S/R:        support {ks} | resist {kr}")
        elif kr is not None:
            poly_lines.append(f"  Resistance: {kr}")
        elif ks is not None:
            poly_lines.append(f"  Support:    {ks}")

        # ── Events ───────────────────────────────────────────────────────────
        events = [
            poly_flow.get("block_trade"),
            poly_flow.get("delta_flip"),
            poly_flow.get("exhaustion"),
            poly_flow.get("absorption"),
            poly_flow.get("imbalance"),
            poly_flow.get("bar_streak"),
        ]
        for ev in events:
            if ev:
                poly_lines.append(f"  ⚡ {ev}")

        poly_section = "\n".join(poly_lines) + "\n\n"

    # Multi-source
    source_section = ""
    if src_count >= 2:
        source_section = f"🗞 *{src_count} outlets reporting this story*\n\n"

    # TradingView chart
    cv_url = chart_url(primary_symbol)
    chart_section = f"🔗 [View chart on TradingView]({cv_url})\n\n" if cv_url else ""

    return (
        f"{'─' * 28}\n"
        f"{action}\n"
        f"{'─' * 28}\n\n"
        f"⏰ {now_utc().strftime('%H:%M UTC')} | {get_session_label()}\n"
        f"📊 Signal strength: {label(final_score)} ({final_score}/100)\n"
        f"⚡ AI impact rating: {impact}/10\n"
        f"⚖️ Market reaction: {reaction}\n\n"
        f"{streak_section}"
        f"{conf_section}"
        f"{source_section}"
        f"{cal_section}"
        f"📰 *What happened:*\n{sanitize(title)}\n\n"
        f"💡 *Why trade this:*\n{sanitize(ai['REASON'])[:600]}\n\n"
        f"{levels_section}"
        f"{poly_section}"
        f"{fg_section}"
        f"{dxy_section}"
        f"{fivem_section}"
        f"{ta_section}"
        f"{hourly_section}"
        f"{fourh_section}"
        f"{daily_section}"
        f"{hist_section}"
        f"{options_section}"
        f"{short_section}"
        f"{premarket_section}"
        f"{yields_section}"
        f"{memory_section}"
        f"🎯 *Assets affected:*\n{sanitize(ai['AFFECTS'])}\n\n"
        f"👀 *Watch for:*\n{sanitize(ai['WATCH'])}\n\n"
        f"💹 *Live moves:*\n{moves_lines}\n\n"
        f"{chart_section}"
        f"{'─' * 28}\n"
        f"{signal_type}"
    )

# ─────────────────────────────
# TELEGRAM
# ─────────────────────────────
def send_telegram(msg: str, retries: int = 3) -> bool:
    # Telegram hard limit is 4096 chars — truncate gracefully if over
    if len(msg) > 4000:
        msg = msg[:3950] + "\n\n_(signal truncated — message too long)_"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    chat_ids = [cid for cid in [CHAT_ID, CHAT_ID_2] if cid]
    primary_ok = False  # return value tracks primary CHAT_ID only

    for i, chat_id in enumerate(chat_ids):
        sent = False
        for attempt in range(retries):
            try:
                r = requests.post(
                    url,
                    data={
                        "chat_id":                  chat_id,
                        "text":                     msg,
                        "parse_mode":               "Markdown",
                        "disable_web_page_preview": "true",
                    },
                    timeout=10,
                )
                data = r.json()
                if r.status_code == 200 and data.get("ok"):
                    sent = True
                    break
                print(f"  Telegram error (attempt {attempt + 1}) [{chat_id}]: {data}")
            except Exception as e:
                print(f"  Telegram attempt {attempt + 1} exception [{chat_id}]: {e}")
            time.sleep(3)
        if not sent:
            print(f"  All Telegram retries exhausted for chat_id={chat_id}.")
        if i == 0:
            # Only the primary CHAT_ID determines success — CHAT_ID_2 failures
            # are logged but must not block state updates (cooldowns, recap flags, etc.)
            primary_ok = sent

    return primary_ok

# ─────────────────────────────
# HEARTBEAT — once per day at 7 AM UTC
# ─────────────────────────────
def maybe_send_heartbeat(state: dict, calendar_events: list):
    # Replaced by send_morning_recap() — the 06:30 Zürich brief is now the
    # daily "bot alive" signal. Keeping this function as a no-op so any
    # existing call sites don't crash.
    pass

# ─────────────────────────────
# WEEKLY SUMMARY — Monday 7 AM UTC
# ─────────────────────────────
def maybe_send_weekly_summary(state: dict):
    now  = now_utc()
    if now.weekday() != 0 or now.hour != 7:
        return

    current_week = now.isocalendar()[1]
    if state.get("__weekly_sent_week__") == current_week:
        return

    signals = state.get("__weekly_signals__", [])
    if not signals:
        state["__weekly_sent_week__"] = current_week
        return

    week_ago = now - timedelta(days=7)
    recent   = [
        s for s in signals
        if datetime.fromisoformat(s["ts"]) > week_ago
    ]

    if not recent:
        state["__weekly_sent_week__"] = current_week
        return

    buys    = sum(1 for s in recent if "BUY" in s.get("bias", "").upper())
    sells   = sum(1 for s in recent if "SELL" in s.get("bias", "").upper())
    neutral = len(recent) - buys - sells
    top5    = sorted(recent, key=lambda x: x.get("score", 0), reverse=True)[:5]

    lines = [
        f"📊 *Weekly Signal Summary*",
        f"Week of {(now - timedelta(days=7)).strftime('%b %d')} — {now.strftime('%b %d, %Y')}",
        f"",
        f"Total signals: {len(recent)}",
        f"✅ BUY: {buys}  |  🔴 SELL: {sells}  |  ⚠️ WATCH: {neutral}",
        f"",
        f"*Top signals this week:*",
    ]
    for s in top5:
        bias_upper = s.get("bias", "").upper()
        emoji = "✅" if "BUY" in bias_upper else "🔴" if "SELL" in bias_upper else "⚠️"
        lines.append(
            f"{emoji} {s.get('symbol', '?')} ({s.get('score', 0)}/100) — "
            f"{s.get('headline', '')[:55]}"
        )

    if send_telegram("\n".join(lines)):
        state["__weekly_sent_week__"] = current_week
        state["__weekly_signals__"]   = []

# ─────────────────────────────
# PRICE MOVEMENT ALERTS
# ─────────────────────────────
# Thresholds: how much an asset must move since the LAST bot run to fire an alert.
PRICE_ALERT_THRESHOLDS = {
    "GC=F":  1.0,   # Gold         — alert on 1%+ move
    "SI=F":  1.5,   # Silver       — more volatile than gold
    "HG=F":  1.0,   # Copper       — industrial bellwether
    "ALI=F": 1.5,   # Aluminium
    "CL=F":  1.5,   # Crude Oil    — alert on 1.5%+ move
    "^GSPC": 1.0,   # S&P 500
    "QQQ":   1.0,   # Nasdaq
}

def maybe_send_breakout_alerts(state: dict):
    """Fire alerts when price breaks above resistance or below support.

    Catches moves that don't have an associated news headline — pure technical
    breakouts. Uses 15-min S/R levels as the breakout boundary. Each level can
    only fire once per direction per cooldown window so we don't spam if price
    chops back and forth across the level.
    """
    last_breakouts = state.get("__last_breakouts__", {})
    alerts         = []
    now            = now_utc().isoformat()
    cooldown_min   = 60  # don't re-fire the same break within 60 min

    for sym, data in _price_cache.items():
        if sym not in ASSET_MAP:
            continue

        price = data.get("price")
        s15   = data.get("support")
        r15   = data.get("resistance")
        if not price or price <= 0:
            continue

        # Vol confirmation — a breakout without volume is almost always a fakeout
        vol = data.get("vol_ratio") or 0
        if vol < 1.3:
            continue

        last = last_breakouts.get(sym, {})

        if r15 and price > r15:
            last_up_str = last.get("up")
            recent = False
            if last_up_str:
                try:
                    recent = (now_utc() - datetime.fromisoformat(last_up_str)) < timedelta(minutes=cooldown_min)
                except Exception:
                    recent = False
            if not recent:
                alerts.append((sym, "up", price, r15, vol))
                last["up"] = now

        if s15 and price < s15:
            last_dn_str = last.get("down")
            recent = False
            if last_dn_str:
                try:
                    recent = (now_utc() - datetime.fromisoformat(last_dn_str)) < timedelta(minutes=cooldown_min)
                except Exception:
                    recent = False
            if not recent:
                alerts.append((sym, "down", price, s15, vol))
                last["down"] = now

        if last:
            last_breakouts[sym] = last

    if alerts:
        lines = ["🚨 *Technical Breakout Alert*\n"]
        for sym, direction, price, level, vol in alerts:
            arrow  = "🟢 ↑" if direction == "up" else "🔴 ↓"
            label_ = "ABOVE resistance" if direction == "up" else "BELOW support"
            prefix = "$" if sym not in FOREX_SYMBOLS else ""
            lines.append(
                f"{arrow} *{friendly(sym)}* broke {label_} ({prefix}{level})\n"
                f"   Now: {prefix}{price} | Volume: {vol:.1f}x avg"
            )
        lines.append("\n_Pure technical break — no news catalyst required. Volume confirms._")
        send_telegram("\n".join(lines))

    state["__last_breakouts__"] = last_breakouts

def maybe_send_event_warnings(state: dict, calendar_events: list):
    """Send a heads-up 30 min before each high-impact economic event.

    Lets the user de-risk positions before scheduled volatility (CPI, NFP, FOMC).
    Each event is only warned about once per day — tracked by a date+title key.
    """
    if not calendar_events:
        return

    now            = now_utc()
    today_key      = now.strftime("%Y-%m-%d")
    sent           = state.get("__event_warnings_sent__", {})
    sent_today     = sent.get(today_key, [])
    new_warnings   = []

    for event in calendar_events:
        time_str = event.get("time", "").replace(" UTC", "").strip()
        title    = event.get("title", "")
        currency = event.get("currency", "")
        if not time_str or not title:
            continue
        try:
            hh, mm    = time_str.split(":")
            event_dt  = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        except Exception:
            continue

        minutes_until = (event_dt - now).total_seconds() / 60
        # Fire once when event is 20-35 min out
        if 20 <= minutes_until <= 35:
            event_key = f"{title}|{currency}|{time_str}"
            if event_key in sent_today:
                continue
            sent_today.append(event_key)
            new_warnings.append((title, currency, time_str, int(minutes_until)))

    if new_warnings:
        lines = ["⏰ *High-Impact Event Warning*\n"]
        for title, currency, time_str, mins in new_warnings:
            lines.append(f"⚡ *{title}* ({currency}) at {time_str} UTC — in ~{mins} min")
        lines.append("\n_Expect volatility and possible whipsaws. Tighten stops or stay flat._")
        send_telegram("\n".join(lines))

    sent[today_key] = sent_today
    # Garbage-collect old date keys (keep only today + yesterday)
    yesterday_key = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    state["__event_warnings_sent__"] = {
        k: v for k, v in sent.items() if k in (today_key, yesterday_key)
    }

def maybe_send_price_alerts(state: dict):
    """Fire a Telegram alert if any asset moved significantly since the last run."""
    last_prices = state.get("__last_prices__", {})
    alerts      = []

    for sym, data in _price_cache.items():
        if sym not in ASSET_MAP:
            continue

        current = data.get("price")
        if not current:
            continue

        threshold = PRICE_ALERT_THRESHOLDS.get(sym, 1.5)
        last      = last_prices.get(sym)

        if last and last > 0:
            pct_change = abs((current - last) / last * 100)
            if pct_change >= threshold:
                direction = "📈" if current > last else "📉"
                signed    = (current - last) / last * 100
                alerts.append((sym, signed, pct_change, current, direction))

    if alerts:
        # Sort by magnitude — biggest move first
        alerts.sort(key=lambda x: x[2], reverse=True)
        lines = ["⚡ *Price Movement Alert*\n"]
        for sym, signed, pct, price, direction in alerts:
            prefix = "$" if sym not in FOREX_SYMBOLS else ""
            lines.append(
                f"{direction} *{friendly(sym)}*: {signed:+.2f}% "
                f"| {prefix}{price}"
            )
        lines.append(
            f"\n_Prices moved since last bot run — check the chart for context._"
        )
        send_telegram("\n".join(lines))

    # Always save current prices for next run comparison
    state["__last_prices__"] = {
        sym: data["price"]
        for sym, data in _price_cache.items()
        if sym in ASSET_MAP and data.get("price")
    }

# ─────────────────────────────
# MORNING RECAP  (06:30 Zürich)
# ─────────────────────────────
def send_morning_recap(state: dict, memory: dict = None) -> bool:
    """Morning brief — fires once at 06:30 Zürich.

    Two modes:
      Weekends (Sat/Sun 06:30) → Gold & oil overnight recap using WEEKEND_FEEDS
      Weekdays (Mon–Fri 06:30) → Full all-market overnight recap using ALL_FEEDS
    """
    print("  Generating morning recap...")
    zh         = zurich_now()
    weekday    = zh.weekday()   # 0=Mon … 6=Sun
    is_weekend_day = weekday >= 5   # Sat=5, Sun=6

    # ── 1. Decide scope ──────────────────────────────────────────────────────
    if is_weekend_day:
        news_hours   = 10
        feeds_to_use = WEEKEND_FEEDS
        recap_title  = "🌅 *Good morning — Gold & Oil Weekend Recap*"
        scope_label  = "Last 10 hours (gold, oil, global news)"
        asset_scope  = "gold, silver, copper, crude oil, aluminium futures, S&P 500, and Nasdaq"
    else:
        news_hours   = 8
        feeds_to_use = ALL_FEEDS
        recap_title  = "🌅 *Good morning — Overnight Recap*"
        scope_label  = "Last 8 hours (all markets)"
        asset_scope  = "gold, silver, copper, crude oil, aluminium futures, S&P 500, and Nasdaq"

    # ── 2. Read prices from the shared cache populated by main() ─────────────
    # Do NOT call refresh_price_cache() here — main() already called it once
    # before invoking this function. Calling it again would clear the cache,
    # make extra yfinance requests, hit rate limits, and leave _price_cache
    # empty for the signal loop that runs immediately after the recap.
    top_movers = []
    for sym in ASSET_MAP.keys():
        d = _price_cache.get(sym)
        if d and d.get("move") is not None:
            top_movers.append((sym, d["move"], d.get("price", "n/a")))
    top_movers.sort(key=lambda x: abs(x[1]), reverse=True)

    mover_lines = []
    for sym, mv, price in top_movers[:10]:
        arrow = "📈" if mv > 0 else "📉"
        mover_lines.append(f"  {arrow} {friendly(sym)}: {mv:+.2f}% | ${price}")
    movers_str = "\n".join(mover_lines) if mover_lines else "  No price data available"

    # ── 3. Collect headlines ─────────────────────────────────────────────────
    cutoff    = now_utc() - timedelta(hours=news_hours)
    headlines = []
    for url, _stype in feeds_to_use:
        try:
            raw  = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            feed = feedparser.parse(raw.content)
            for entry in feed.entries[:8]:
                title = entry.get("title", "").strip()
                if not title or not is_important(title):
                    continue
                try:
                    pub = parsedate_to_datetime(entry.get("published", ""))
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=timezone.utc)
                    if pub < cutoff:
                        continue
                except Exception:
                    pass   # no timestamp → include anyway
                if title not in headlines:
                    headlines.append(title)
                if len(headlines) >= 20:
                    break
        except Exception as e:
            print(f"  Recap feed error ({url[:55]}): {e}")
        if len(headlines) >= 20:
            break

    headlines_str = "\n".join(f"  • {h}" for h in headlines[:20]) if headlines \
                    else f"  No major headlines in the last {news_hours} hours"

    # ── 4. Sentiment context ─────────────────────────────────────────────────
    vix = _price_cache.get("^VIX")
    fear_str = f"VIX: {vix['price']} ({vix_label(vix['price'])})" if vix and vix.get("price") else "n/a"

    # ── 4b. Treasury yields context ──────────────────────────────────────────
    yields_ctx = ""
    if POLYGON_KEY:
        try:
            yields_data = fetch_treasury_yields(memory or {})
            if yields_data and yields_data.get("y10") is not None:
                y2  = yields_data.get("y2")
                y10 = yields_data.get("y10")
                yields_ctx = (
                    f"\n\nTreasury Yields: "
                    + (f"2Y={y2:.2f}% | " if y2 else "")
                    + f"10Y={y10:.2f}% | "
                    + f"Spread: {yields_data.get('spread_2_10', 0):+.2f}% "
                    + f"({yields_data.get('curve_label', '')})"
                )
        except Exception as e:
            print(f"  Yields fetch in recap failed (non-fatal): {e}")

    # ── 5. AI brief ──────────────────────────────────────────────────────────
    if is_weekend_day:
        task_prompt = """Write a sharp weekend brief for a trader watching futures and indices. Cover:
1. OVERNIGHT SUMMARY: What moved the most in gold, silver, copper, or oil overnight and why (1-2 sentences)
2. TOP 2 SETUPS: The 2 clearest setups right now (gold, silver, copper or crude oil preferred since they trade on weekends)
3. KEY LEVELS: One key price level for each setup
4. MAIN RISK: The one macro or geopolitical factor most likely to drive price today"""
    else:
        task_prompt = """Write a sharp morning brief the trader can read in 60 seconds. Cover:
1. OVERNIGHT SUMMARY: What moved the most and why (1-2 sentences)
2. TOP 3 ASSETS TO WATCH TODAY: Which 3 of the 5 assets have the clearest setup and why
3. KEY LEVELS: For each of those 3 assets, one price level to watch (support or resistance)
4. MAIN RISK: The one macro factor that could surprise markets today"""

    prompt = f"""You are a senior trading analyst giving a concise morning briefing.
The trader watches: {asset_scope}.

Current Zürich time: {zh.strftime('%H:%M on %A %d %b')} | Scope: {scope_label}

━━━ PRICE MOVES ━━━
{movers_str}

━━━ NEWS HEADLINES ━━━
{headlines_str}

━━━ MARKET SENTIMENT ━━━
{fear_str}{yields_ctx}

━━━ YOUR TASK ━━━
{task_prompt}

Rules:
- Use full plain-English names, NEVER tickers
- Be direct — no fluff, no intro sentence, go straight to the content
- Numbers only for price levels"""

    brief = "AI unavailable — check markets manually."
    for _ai_attempt in range(3):
        try:
            res = client.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 650,
                messages   = [{"role": "user", "content": prompt}],
            )
            brief = res.content[0].text.strip()
            break
        except Exception as e:
            print(f"  Morning recap AI error (attempt {_ai_attempt + 1}/3): {type(e).__name__}: {e}")
            if _ai_attempt < 2:
                time.sleep(4 * (_ai_attempt + 1))

    # ── 6. Format & send ─────────────────────────────────────────────────────
    cal_line = ""
    if not is_weekend_day:
        try:
            events = fetch_calendar()
            if events:
                ev_str   = " | ".join(f"{e['title']} ({e['currency']})" for e in events[:3])
                cal_line = f"\n📅 *High-impact events today:* {ev_str}\n"
        except Exception:
            pass

    msg = (
        f"{'─' * 28}\n"
        f"{recap_title}\n"
        f"{'─' * 28}\n\n"
        f"⏰ Zürich {zh.strftime('%H:%M')} | {scope_label}\n"
        f"{cal_line}\n"
        f"📊 *Top movers:*\n{movers_str}\n\n"
        f"🧠 *AI brief:*\n{sanitize(brief)}\n\n"
        f"{'─' * 28}\n"
        f"Bot watching {len(ASSET_MAP)} assets across {len(ALL_FEEDS)} feeds 🎯"
    )

    if send_telegram(msg):
        print("  Morning recap sent.")
        return True
    return False


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    zh   = zurich_now()
    print(f"[{now_utc().strftime('%H:%M:%S UTC')}] Bot starting... "
          f"(Zürich {zh.strftime('%H:%M')})")

    # ── QUIET HOURS GATE (23:00 – 06:30 Zürich) ─────────────────────────────
    # During this window the bot wakes every 10 min but sends nothing.
    # At 06:30 the gate opens and the morning recap fires first.
    if is_zurich_quiet_hours():
        print(f"Quiet hours ({zh.strftime('%H:%M')} Zürich) — no signals until 06:30. Sleeping.")
        return

    state  = load_state()
    memory = load_memory()

    weekend = is_weekend()

    # Fetch prices first — needed for both morning recap and signals.
    print("Fetching intraday prices + daily context...")
    refresh_price_cache()

    # Check outcomes of previously sent signals against current prices.
    # Must run AFTER refresh_price_cache() so _price_cache has current prices.
    check_and_update_outcomes(memory)

    # ── MORNING RECAP (fires once at 06:30 Zürich) ───────────────────────────
    # Must run BEFORE the market-open gate — US markets are closed at 06:30
    # Zürich (04:30 UTC), so checking is_market_open() first would skip the
    # recap entirely every weekday morning.
    if should_send_morning_recap(state):
        print("Morning recap window — sending overnight brief...")
        # Mark the date and persist to disk BEFORE sending — this prevents
        # concurrent GitHub Actions runs from both passing the check and
        # sending duplicate recaps.
        state["__last_recap_date__"] = zurich_now().date().isoformat()
        save_state(state)
        try:
            send_morning_recap(state, memory=memory)
        except Exception as e:
            print(f"  Morning recap error (non-fatal): {e}")

    # ── MARKET OPEN GATE ─────────────────────────────────────────────────────
    if weekend:
        print("Weekend mode — monitoring gold, oil, and global news.")
        active_feeds = WEEKEND_FEEDS
    elif not is_market_open():
        print("Market closed — nothing to do.")
        save_memory(memory)   # save any outcome updates from check_and_update_outcomes
        return
    else:
        # ── Holiday check ─────────────────────────────────────────────────────
        if POLYGON_KEY:
            try:
                holiday_url = f"https://api.massive.com/v1/marketstatus/now"
                holiday_r   = requests.get(holiday_url,
                                           params={"apiKey": POLYGON_KEY}, timeout=8)
                if holiday_r.status_code == 200:
                    holiday_data = holiday_r.json()
                    exchanges    = holiday_data.get("exchanges") or {}
                    nyse         = exchanges.get("nyse", "open")
                    if nyse == "closed":
                        print("Market holiday (NYSE closed) — no signals today.")
                        save_memory(memory)
                        return
                    elif nyse == "early-close":
                        print("Market closing early today — reducing signal activity.")
            except Exception as e:
                print(f"  Holiday check failed (non-fatal): {e}")
        active_feeds = ALL_FEEDS

    maybe_send_weekly_summary(state)

    calendar_events = fetch_calendar() if not weekend else []

    maybe_send_heartbeat(state, calendar_events)

    # ── PRE-EVENT WARNINGS ───────────────────────────────────────────────
    maybe_send_event_warnings(state, calendar_events)

    # ── PRICE MOVEMENT ALERTS ────────────────────────────────────────────
    maybe_send_price_alerts(state)

    # ── TECHNICAL BREAKOUT ALERTS (no news required) ─────────────────────
    maybe_send_breakout_alerts(state)

    # ── PRE-FETCH SUPPLEMENTARY POLYGON DATA (once per run) ─────────────────
    # Calling these here populates memory cache so every analyze() call in the
    # signal loop gets instant cache hits instead of live HTTP timeouts.
    # On failure _ep_fail() is set; after ≥2 fails the endpoint is skipped
    # for the rest of this job — no repeated 5-second timeouts per signal.
    if POLYGON_KEY and not weekend:
        print("Pre-fetching supplementary Polygon data (once per run)...")
        try:
            # Treasury yields — no symbol restriction
            fetch_treasury_yields(memory)
            # Pre-market gap data — all symbols
            for _pm_sym in list(POLYGON_SYMBOLS.keys()):
                fetch_premarket_data(_pm_sym, memory)
                time.sleep(0.3)
            # Options flow + short data — SPY / QQQ only
            for _eq_sym in ["^GSPC", "QQQ"]:
                if _eq_sym in POLYGON_SYMBOLS:
                    fetch_options_flow(_eq_sym, memory)
                    time.sleep(0.5)
                    fetch_short_data(_eq_sym, memory)
                    time.sleep(0.5)
        except Exception as _pf_err:
            print(f"  Pre-fetch error (non-fatal): {_pf_err}")
        print("  Supplementary pre-fetch done.")

    # ── PRE-SCAN: count how many outlets report each story ──────────────
    print("Pre-scanning feeds for multi-source stories...")
    all_entries: list  = []
    story_counts: dict = {}

    for url, signal_type in active_feeds:
        try:
            raw  = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            feed = feedparser.parse(raw.content)
            for entry in feed.entries[:5]:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                all_entries.append((title, entry, signal_type))
                key     = title.lower()[:150]
                matched = False
                for existing_key in list(story_counts.keys()):
                    if SequenceMatcher(None, key, existing_key).ratio() >= 0.70:
                        story_counts[existing_key] += 1
                        matched = True
                        break
                if not matched:
                    story_counts[key] = 1
        except Exception as e:
            print(f"  Pre-scan error ({url[:55]}): {e}")

    # ── PROCESS ENTRIES ──────────────────────────────────────────────────
    seen_headlines = state["seen_headlines"]
    cooldowns      = state["__cooldowns__"]
    weekly_signals = state["__weekly_signals__"]

    signals_sent = 0

    for title, entry, signal_type in all_entries:
        try:
            if title in seen_headlines:
                continue

            if is_duplicate(title, seen_headlines):
                print(f"  Duplicate skipped: {title[:60]}")
                seen_headlines.append(title)
                continue

            seen_headlines.append(title)

            if not is_fresh(entry):
                continue

            if not is_important(title):
                continue

            moves        = get_cached_moves(title)
            src_count    = count_sources(title, story_counts)
            sc, reaction = score_signal(title, moves, signal_type, src_count)

            # Sentiment pre-filter — adds a small score nudge for clearly
            # directional language. Doesn't reject anything by itself, just
            # helps directional headlines clear the threshold faster.
            sent_score, sent_label = score_headline_sentiment(title)
            if sent_label == "bullish" or sent_label == "bearish":
                sc = min(100, sc + 5)

            print(f"  Score {sc:>3} | src:{src_count} | sent:{sent_label} | {signal_type} | {title[:55]}")

            if sc < SCORE_THRESHOLD:
                continue

            primary_symbol = get_primary_symbol(title, moves)

            if cooldown_active(primary_symbol, cooldowns):
                print(f"  Cooldown active for {primary_symbol}, skipping.")
                continue

            ai_text    = analyze(title, reaction, moves, signal_type, calendar_events, primary_symbol, memory=memory)
            ai_parsed  = parse_ai(ai_text)
            ai_impact  = parse_impact(ai_parsed.get("IMPACT", "5"))
            if ai_impact <= 2:
                print(f"  AI impact too low ({ai_impact}/10), skipping signal.")
                continue

            # Key-level trap check — block BUYs right at resistance and SELLs
            # right at support. These are the classic retail trap setups where
            # odds favour rejection rather than continuation.
            ai_bias      = sanitize(ai_parsed.get("BIAS", "")).strip().upper()
            primary_data = moves.get(primary_symbol, {})

            # ── Confidence gate — only directional signals with meaningful
            # indicator alignment get through. Requires ≥35% of available
            # indicators to agree with the bias direction. When Polygon data
            # is present this can be up to 15 votes; without it ~5 yfinance
            # votes. Guards against noise signals where order flow contradicts
            # the AI direction (e.g. bearish delta on a BUY signal).
            if ai_bias in ("BUY", "SELL"):
                conf_score, conf_total = signal_confidence(primary_data, ai_bias)
                if conf_total >= 5 and (conf_score / conf_total) < 0.35:
                    print(f"  Low confidence ({conf_score}/{conf_total} = {conf_score/conf_total:.0%}), skipping signal.")
                    continue

            is_trap, trap_reason = is_at_trap_level(ai_bias, primary_data)
            if is_trap:
                print(f"  {trap_reason}")
                continue

            msg     = format_msg(
                title, reaction, sc, moves,
                primary_symbol, ai_text, signal_type, calendar_events,
                state, src_count, memory=memory,
            )

            if send_telegram(msg):
                signals_sent += 1
                cooldowns[primary_symbol] = now_utc().isoformat()

                bias_clean = sanitize(ai_parsed.get("BIAS", "NEUTRAL")).strip()

                # ── Record signal in memory for outcome tracking ──────────────
                if bias_clean in ("BUY", "SELL"):
                    p_data    = moves.get(primary_symbol, {})
                    levels    = compute_trade_levels(bias_clean, p_data,
                                                     hist_data=p_data.get("poly_hist"))
                    cs, ct    = signal_confidence(p_data, bias_clean)
                    record_signal(
                        memory      = memory,
                        symbol      = primary_symbol,
                        bias        = bias_clean,
                        price       = p_data.get("price", 0),
                        stop_str    = levels.get("stop",   ""),
                        target_str  = levels.get("target", ""),
                        conf_score  = cs,
                        conf_total  = ct,
                        poly_flow   = p_data.get("polygon_flow") or {},
                        headline    = title,
                        base_score  = sc,
                    )

                update_streak(state, primary_symbol, bias_clean)

                weekly_signals.append({
                    "ts":       now_utc().isoformat(),
                    "headline": title[:80],
                    "bias":     bias_clean,
                    "score":    sc,
                    "symbol":   friendly(primary_symbol),
                })

                if signals_sent >= MAX_SIGNALS_PER_RUN:
                    print(f"  Signal cap ({MAX_SIGNALS_PER_RUN}) reached — stopping this run.")
                    break

            time.sleep(2)

        except Exception as e:
            print(f"  Entry error ({title[:50]}): {e}")

    state["seen_headlines"]     = seen_headlines
    state["__cooldowns__"]      = cooldowns
    state["__weekly_signals__"] = weekly_signals
    save_state(state)
    save_memory(memory)

    print(f"Done. Signals sent: {signals_sent}")


if __name__ == "__main__":
    main()
