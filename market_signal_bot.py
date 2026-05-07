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
  11. Crypto Fear & Greed Index — from alternative.me (free, no key)
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

SCORE_THRESHOLD      = 55
MAX_HEADLINE_AGE_MIN = 45
SYMBOL_COOLDOWN_MIN  = 15
SIMILARITY_THRESHOLD = 0.78
SEEN_FILE            = "seen_headlines.json"
MAX_SIGNALS_PER_RUN  = 5   # Cap signals per 5-min run — prevents Telegram spam during news floods

if not all([TELEGRAM_TOKEN, CHAT_ID, ANTHROPIC_KEY]):
    raise SystemExit("ERROR: TELEGRAM_TOKEN, CHAT_ID, and ANTHROPIC_KEY must be set.")

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

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
    "https://www.investing.com/rss/news_25.rss",   # Forex
    "https://www.investing.com/rss/news_8.rss",    # Stock markets
    "https://www.investing.com/rss/news_14.rss",   # Commodities
    "https://www.fxstreet.com/rss/news",
    "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
    "https://www.aljazeera.com/xml/rss/all.xml",   # Live geopolitical/war updates
    "https://www.forexlive.com/feed/news",          # Best real-time macro/forex feed
    "https://www.kitco.com/rss/kitco-news.xml",     # Gold & precious metals focus
    "https://oilprice.com/rss/main",                # Oil & energy focus
]

ANALYSIS_FEEDS = [
    "https://www.fxstreet.com/rss/analysis",
    "https://www.dailyfx.com/feeds/all",            # Professional forex analysis (IG Group)
]

CRYPTO_FEEDS = [
    "https://www.coindesk.com/arc/outbound/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.investing.com/rss/news_301.rss",
    "https://www.theblock.co/rss.xml",              # Institutional crypto journalism
    "https://blockworks.co/feed",                   # Macro x crypto intersection
]

ALL_FEEDS = (
    [(url, "📰 NEWS") for url in NEWS_FEEDS] +
    [(url, "📊 ANALYSIS") for url in ANALYSIS_FEEDS] +
    [(url, "🪙 CRYPTO") for url in CRYPTO_FEEDS]
)

WEEKEND_FEEDS = [(url, "🪙 CRYPTO") for url in CRYPTO_FEEDS]

# ─────────────────────────────
# ASSET MAP & FRIENDLY NAMES
# ─────────────────────────────
ASSET_MAP = {
    "GC=F":     ["gold", "xau", "bullion", "precious metal"],
    "ALI=F":    ["aluminium", "aluminum", "alcoa", "bauxite", "rusal", "aluminium smelter",
                 "aluminum smelter", "norsk hydro", "hindalco", "aluminium tariff",
                 "aluminum tariff", "aluminium supply", "aluminum supply"],
    "CL=F":     ["oil", "crude", "wti", "brent", "opec", "petroleum", "energy"],
    "^GSPC":    ["s&p", "spx", "spy", "sp500", "s&p 500", "equities", "wall street"],
    "QQQ":      ["nasdaq", "qqq", "nq", "us100", "tech 100"],
    "GBPUSD=X": ["gbp", "pound", "sterling", "bank of england"],
    "EURUSD=X": ["eur", "euro", "ecb", "eurozone", "european central bank"],
    "BTC-USD":  ["bitcoin", "btc", "crypto", "cryptocurrency", "digital asset", "satoshi", "halving",
                 "stablecoin", "stablecoins", "usdt", "usdc", "tether",
                 "blockchain", "tge", "token generation", "web3", "altcoin", "coinbase", "binance"],
    "ETH-USD":  ["ethereum", "defi", "smart contract", "layer 2", "l2", "eip"],
}

ASSET_NAMES = {
    "GC=F":     "Gold (XAU/USD)",
    "ALI=F":    "Aluminium (ALI/USD)",
    "CL=F":     "Crude Oil (WTI)",
    "^GSPC":    "S&P 500",
    "QQQ":      "US Tech 100 (Nasdaq)",
    "GBPUSD=X": "GBP/USD (Pound)",
    "EURUSD=X": "EUR/USD (Euro)",
    "BTC-USD":  "Bitcoin (BTC/USD)",
    "ETH-USD":  "Ethereum (ETH/USD)",
}

# Context-only symbols — fetched for data, never trigger signals
CONTEXT_SYMBOLS = {
    "DX-Y.NYB": "US Dollar Index (DXY)",
    "^VIX":     "CBOE Volatility Index (VIX)",
}

# Crypto symbols — trade 24/7 including weekends
CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD"}

# Forex pairs are exchange rates — displayed without $ prefix
FOREX_SYMBOLS = {"GBPUSD=X", "EURUSD=X"}

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
    "aluminium", "aluminum", "alcoa", "bauxite", "rusal", "norsk hydro", "hindalco",
    "oil", "crude", "wti", "brent", "opec", "petroleum", "energy",
    "s&p", "spx", "spy", "sp500", "s&p 500", "wall street", "equities",
    "nasdaq", "qqq", "nq", "us100", "tech 100",
    "gbp", "pound", "sterling", "bank of england",
    "eur", "euro", "ecb", "european central bank", "eurozone",
    "bitcoin", "btc", "ethereum", "crypto", "cryptocurrency",
    "digital asset", "blockchain", "defi", "halving", "altcoin", "stablecoin",
    "sec crypto", "etf crypto", "bitcoin etf", "coinbase", "binance",
    "smart contract", "layer 2", "l2", "on-chain",
    # Note: "eth" and "ether" intentionally excluded — substrings of "netherlands",
    # "together", "whether" etc. "ethereum" is specific enough.
]

CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "crypto", "cryptocurrency",
    "digital asset", "blockchain", "defi", "halving", "altcoin", "stablecoin",
    "stablecoins", "usdt", "usdc", "tether", "coinbase", "binance",
    "bitcoin etf", "sec crypto",
]
# Note: "eth" intentionally excluded — it is a substring of "netherlands",
# "method", "elizabeth" etc. ASSET_MAP still routes ETH headlines correctly.
# Crypto scoring is handled via signal_type "🪙 CRYPTO" from crypto feeds.

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
    """True on Sat/Sun and Fri after 22:00 UTC.

    Friday after 22:00 UTC is when forex closes for the week, so all
    traditional markets are shut — only crypto trades. Treating this
    window as 'weekend' makes the bot stay alive on crypto-only mode
    instead of returning early via is_market_open and missing 2 hours
    of crypto signals every week.
    """
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:                          # Saturday, Sunday
        return True
    if now.weekday() == 4 and now.hour >= 22:       # Friday after forex close
        return True
    return False

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
    """Returns (score, total) where score = indicators agreeing with bias."""
    score, total = 0, 0

    rsi = data.get("rsi")
    if rsi is not None:
        total += 1
        if bias == "BUY" and rsi < 65:
            score += 1
        elif bias == "SELL" and rsi > 35:
            score += 1

    macd_val = data.get("macd")
    macd_sig = data.get("macd_signal")
    if macd_val is not None and macd_sig is not None:
        total += 1
        if bias == "BUY" and macd_val > macd_sig:
            score += 1
        elif bias == "SELL" and macd_val < macd_sig:
            score += 1

    trend = data.get("trend", "")
    if trend and trend != "n/a":
        total += 1
        if bias == "BUY" and "Uptrend" in trend:
            score += 1
        elif bias == "SELL" and "Downtrend" in trend:
            score += 1

    # 1h trend — multi-timeframe confirmation
    hourly = data.get("hourly_trend", "")
    if hourly and hourly != "n/a":
        total += 1
        if bias == "BUY" and "Uptrend" in hourly:
            score += 1
        elif bias == "SELL" and "Downtrend" in hourly:
            score += 1

    fourh = data.get("4h_trend", "")
    if fourh and fourh != "n/a" and bias in ("BUY", "SELL"):
        total += 1
        if bias == "BUY" and "Uptrend" in fourh:
            score += 1
        elif bias == "SELL" and "Downtrend" in fourh:
            score += 1

    daily = data.get("daily_trend", "")
    if daily and daily != "n/a" and bias in ("BUY", "SELL"):
        total += 1
        if bias == "BUY" and "Uptrend" in daily:
            score += 1
        elif bias == "SELL" and "Downtrend" in daily:
            score += 1

    vol = data.get("vol_ratio")
    if vol is not None and bias in ("BUY", "SELL"):
        total += 1
        if vol >= 1.5:
            score += 1

    # RSI divergence — bullish divergence supports BUY, bearish supports SELL.
    # Only count when present (don't penalise signals that have no divergence).
    divergence = data.get("divergence", "")
    if divergence:
        total += 1
        if bias == "BUY" and divergence == "bullish":
            score += 1
        elif bias == "SELL" and divergence == "bearish":
            score += 1

    # VIX sentiment — extreme readings are contrarian signals.
    # Only counted for non-crypto (VIX measures equity/macro fear, not crypto).
    vix_price = data.get("vix_price")
    is_crypto  = data.get("_is_crypto", False)
    if vix_price is not None and not is_crypto:
        total += 1
        # Extreme fear (VIX >= 28) supports BUY — fear marks bottoms
        # Extreme complacency (VIX <= 14) supports SELL — greed marks tops
        if bias == "BUY" and vix_price >= 28:
            score += 1
        elif bias == "SELL" and vix_price <= 14:
            score += 1

    # Order flow delta — does buy/sell pressure confirm the bias?
    # Counts both 15m (primary) and 4h (macro confirmation) if present.
    of_15 = data.get("order_flow") or {}
    of_4h = data.get("4h_order_flow") or {}
    for of in [of_15, of_4h]:
        if of and bias in ("BUY", "SELL"):
            total += 1
            if bias == "BUY" and of.get("of_bias") == "bullish":
                score += 1
            elif bias == "SELL" and of.get("of_bias") == "bearish":
                score += 1

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
    "GC=F":     "COMEX:GC1!",
    "ALI=F":    "COMEX:ALI1!",
    "CL=F":     "NYMEX:CL1!",
    "^GSPC":    "SP:SPX",
    "QQQ":      "NASDAQ:QQQ",
    "GBPUSD=X": "FX:GBPUSD",
    "EURUSD=X": "FX:EURUSD",
    "BTC-USD":  "BITSTAMP:BTCUSD",
    "ETH-USD":  "BITSTAMP:ETHUSD",
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

    # On weekends only fetch crypto + DXY — all other markets are closed
    if is_weekend():
        symbols = list(CRYPTO_SYMBOLS) + list(CONTEXT_SYMBOLS.keys())
    else:
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
            _price_cache[symbol] = data
        time.sleep(0.3)

    summary = {s: f"{d['move']:+.2f}% @ {d['price']}" for s, d in _price_cache.items()}
    print(f"  Prices: {summary or 'all unavailable'}")

def get_cached_moves(title: str) -> dict:
    title_lower = title.lower()
    weekend     = is_weekend()

    # Macro events affect everything — but only tradeable assets for the current session
    if is_macro(title_lower) and not weekend:
        return {s: d for s, d in _price_cache.items() if s in ASSET_MAP}

    # Try to match a specific asset from the headline
    for sym, keywords in ASSET_MAP.items():
        if weekend and sym not in CRYPTO_SYMBOLS:
            continue  # Never return a closed market on weekends
        if any(k in title_lower for k in keywords):
            return {sym: _price_cache[sym]} if sym in _price_cache else {}

    # Crypto headline with no specific asset match → default to BTC, never S&P
    if is_crypto_headline(title_lower):
        return {"BTC-USD": _price_cache["BTC-USD"]} if "BTC-USD" in _price_cache else {}
    # Fallback — use BTC on weekends, S&P 500 on weekdays
    if weekend:
        return {"BTC-USD": _price_cache["BTC-USD"]} if "BTC-USD" in _price_cache else {}
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
    # Crypto headline with no specific match → BTC, never S&P 500
    if is_crypto_headline(title):
        return "BTC-USD"
    if moves:
        return max(moves, key=lambda s: abs(moves[s]["move"]))
    # Safe fallback — never return a closed market on weekends
    return "BTC-USD" if is_weekend() else "^GSPC"

# ─────────────────────────────
# SCORING
# ─────────────────────────────
ENERGY_KEYWORDS = [
    "oil", "crude", "wti", "brent", "opec", "petroleum", "energy",
    "gold", "xau", "bullion", "precious metal",
    "aluminium", "aluminum", "alcoa", "bauxite",
]

def score_signal(title: str, moves: dict, signal_type: str, src_count: int = 1):
    avg_move   = sum(abs(d["move"]) for d in moves.values()) / len(moves) if moves else 0
    api_failed = len(moves) == 0
    crypto     = signal_type == "🪙 CRYPTO" or is_crypto_headline(title)

    # ── Unified freshness thresholds ────────────────────────────────────────
    # Previously crypto had 3-4x looser thresholds than non-crypto, causing
    # crude/forex to score near-zero whenever they were already moving — exactly
    # when signals are most needed. Now every asset class uses the same scale.
    if avg_move < 1.0:   freshness, reaction = 50, "FRESH"
    elif avg_move < 2.5: freshness, reaction = 40, "WARMING"
    elif avg_move < 5.0: freshness, reaction = 25, "MOVING"
    elif avg_move < 8.0: freshness, reaction = 10, "RUNNING"
    else:                freshness, reaction = 0,  "PRICED IN"

    # ── Bonuses ─────────────────────────────────────────────────────────────
    macro_bonus  = 20 if is_macro(title) else 0
    # Energy/commodity bonus — matches macro_bonus so oil/gold aren't ignored
    energy_bonus = 15 if any(k in title.lower() for k in ENERGY_KEYWORDS) else 0
    # Crypto bonus halved — crypto feeds (6 feeds) already generate far more
    # volume than commodity/forex feeds; +20 was creating a ~30pt unfair advantage
    crypto_bonus   = 10 if crypto else 0
    analysis_bonus = 10 if signal_type == "📊 ANALYSIS" else 0
    breadth_bonus  = 10 if api_failed else min(20, len(moves) * 4)
    source_bonus   = min(20, (src_count - 1) * 7)

    vol_bonus = 0
    for d in moves.values():
        if (d.get("vol_ratio") or 0) >= 2.0:
            vol_bonus = 10
            break

    # Delta flip bonus — a confirmed order flow reversal adds conviction
    delta_bonus = 0
    for d in moves.values():
        of = d.get("order_flow") or {}
        if of.get("delta_flip"):
            delta_bonus = 8
            break

    total = min(100, freshness + macro_bonus + energy_bonus + crypto_bonus
                     + analysis_bonus + breadth_bonus + source_bonus
                     + vol_bonus + delta_bonus)
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
        lines.append(f"  Delta:      {of['of_bias'].upper()} (cum={of['cum_delta']:+,} | last bar={of['delta']:+,})")
        if of.get("delta_flip"):
            lines.append(f"  ⚡ {of['delta_flip']}")
    return "\n".join(lines)

def analyze(title: str, reaction: str, moves: dict, signal_type: str,
            calendar_events: list, primary_symbol: str = "") -> str:

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
    if vix_d and primary_symbol not in CRYPTO_SYMBOLS:
        fear_str = f"VIX: {vix_d.get('price', 'n/a')} — {vix_label(vix_d.get('price', 0)) if vix_d.get('price') else 'n/a'}"
    elif primary_symbol in CRYPTO_SYMBOLS:
        fg = fetch_fear_greed()
        fear_str = f"Crypto Fear & Greed: {fg['value']} — {fg['label']}" if fg else "n/a"
    else:
        fear_str = "n/a"

    # ── DXY context ──────────────────────────────────────────────────────────
    dxy_d = _price_cache.get("DX-Y.NYB")
    dxy_str = f"DXY: {dxy_d['price']} ({dxy_d['move']:+.2f}% today) | {dxy_d.get('trend','n/a')}" if dxy_d else "n/a"

    macro_note  = "This is a macro event affecting all instruments." if is_macro(title) else ""
    crypto_note = "This is a crypto headline — focus on Bitcoin (BTC/USD) and Ethereum (ETH/USD)." if is_crypto_headline(title) else ""

    prim_name = friendly(primary_symbol) if primary_symbol else "the most affected asset"
    prim_price = primary_data.get("price", "n/a")
    prim_move  = f"{primary_data['move']:+.2f}%" if primary_data.get("move") is not None else "n/a"

    prompt = f"""You are a senior institutional trading analyst with deep expertise in macro, technical analysis, and risk management.

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
{macro_note}{crypto_note}

━━━ OTHER INSTRUMENTS (context) ━━━
{other_str}

━━━ YOUR TASK ━━━
Analyze this headline for {prim_name}. Synthesize ALL data above into one unified view:
1. What does this news fundamentally mean for this asset?
2. Do the technicals CONFIRM or CONTRADICT? Check: RSI, MACD, trend (all timeframes), divergence, market structure (BOS/CHoCH), order flow delta, liquidity sweeps, VWAP position, candle patterns, premium/discount zone.
3. Does ORDER FLOW agree with the directional bias? A delta flip or strong cumulative delta in the same direction as the trade is high-conviction confirmation. Disagreement = lower conviction.
4. Is price at a favourable location (discount for BUY, premium for SELL)? Are there nearby S/R or order blocks that invalidate the setup?
5. What is the single biggest risk to this trade?

RULES:
- Always use full plain-English names. NEVER use tickers (QQQ, SPX, GC, CL, BTC, ETH, DXY, NQ etc.)
- ENTRY, STOP, TARGET are for {prim_name} ONLY — numbers only, no words
- REASON must cover: (1) news catalyst, (2) multi-timeframe technical + order flow read, (3) exact trade logic with key level

Return EXACTLY this format, no extra text:
BIAS: BUY / SELL / NEUTRAL
IMPACT: [1-10 — how market-moving is this news]
AFFECTS: [full plain-English names of affected instruments]
REASON: [3 sentences max: catalyst + technicals/order flow read + trade logic]
ENTRY: [price or range for {prim_name} only — numbers only]
STOP: [stop loss — single number]
TARGET: [profit target — single number or range]
WATCH: [the one thing that would invalidate this trade]"""

    try:
        res = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.content[0].text
    except Exception as e:
        print(f"  AI error: {e}")
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
def compute_trade_levels(bias: str, primary_data: dict) -> dict:
    """Compute Entry/SL/TP from price + 15m and daily support/resistance.

    AI-generated levels frequently violated the bot's own technical levels
    (e.g. SELL stop placed below resistance, BUY target placed beyond
    resistance with no breakout logic, target inside the support zone we'd
    expect to hold). This function replaces those with rule-based levels
    that always respect S/R:

      - Entry: tight ±0.15% range around current price.
      - BUY  SL: just below 15m support (0.15% buffer); falls back to daily
        support, else a 1% safety stop.
      - SELL SL: just above 15m resistance (0.15% buffer); falls back to
        daily resistance, else a 1% safety stop.
      - TP1 (conservative): the nearest S/R giving at least 1R reward,
        otherwise a 1.5R extension.
      - TP2 (aggressive): a 2.5R extension capped at 3.5R; or daily S/R
        if it sits between TP1 and the cap.
      - Always enforces SL on the loss side, TP on the profit side, and
        formats decimals based on price magnitude.

    Returns {} if bias is non-directional or price is missing.
    """
    price = primary_data.get("price")
    if not price or price <= 0 or bias not in ("BUY", "SELL"):
        return {}

    s15 = primary_data.get("support")
    r15 = primary_data.get("resistance")
    sd  = primary_data.get("daily_support")
    rd  = primary_data.get("daily_resistance")
    atr = primary_data.get("atr")

    # ATR-aware buffer: at least 0.15% of price, but scales up with volatility
    # (half an ATR) so stops aren't too tight in fast-moving markets.
    buf = max(price * 0.0015, (atr * 0.5) if atr else 0)

    # Decimal precision based on price magnitude
    if   price >= 1000: dec = 1
    elif price >= 10:   dec = 2
    elif price >= 1:    dec = 4
    else:               dec = 5
    fmt = lambda n: f"{n:.{dec}f}"

    if bias == "BUY":
        entry_lo = price * 0.999
        entry_hi = price * 1.0015

        # SL: below support (must be below current price)
        if s15 and s15 < price * 0.998:
            sl = s15 - buf
        elif sd and sd < price * 0.99:
            sl = sd - buf
        else:
            sl = price * 0.99  # 1% safety stop

        risk = price - sl
        if risk <= 0:
            sl = price * 0.99
            risk = price - sl

        min_tp = price + risk * 1.0   # any TP must be at least 1R
        max_tp = price + risk * 3.5   # cap on far target

        # TP1: nearest meaningful resistance giving 1R+, else 1.5R extension
        tp1_candidates = []
        if r15 and (r15 - buf) >= min_tp:
            tp1_candidates.append(r15 - buf)
        if rd and (rd - buf) >= min_tp and (rd - buf) <= max_tp:
            tp1_candidates.append(rd - buf)
        tp1 = min(tp1_candidates) if tp1_candidates else (price + risk * 1.5)

        # TP2: 2.5R extension, or daily resistance if between TP1 and cap
        tp2 = price + risk * 2.5
        if rd and (rd - buf) > tp1 and (rd - buf) <= max_tp:
            tp2 = max(tp2, rd - buf)
        tp2 = min(tp2, max_tp)

        if tp2 > tp1 * 1.003:  # ranges only when meaningfully apart
            target = f"{fmt(tp1)}-{fmt(tp2)}"
        else:
            target = fmt(tp1)

        return {
            "entry":  f"{fmt(entry_lo)}-{fmt(entry_hi)}",
            "stop":   fmt(sl),
            "target": target,
        }

    # SELL
    entry_lo = price * 0.9985
    entry_hi = price * 1.001

    if r15 and r15 > price * 1.002:
        sl = r15 + buf
    elif rd and rd > price * 1.01:
        sl = rd + buf
    else:
        sl = price * 1.01

    risk = sl - price
    if risk <= 0:
        sl = price * 1.01
        risk = sl - price

    min_tp = price - risk * 1.0   # any TP must be at least 1R below
    max_tp = price - risk * 3.5   # furthest TP allowed

    tp1_candidates = []
    if s15 and (s15 + buf) <= min_tp:
        tp1_candidates.append(s15 + buf)
    if sd and (sd + buf) <= min_tp and (sd + buf) >= max_tp:
        tp1_candidates.append(sd + buf)
    tp1 = max(tp1_candidates) if tp1_candidates else (price - risk * 1.5)

    tp2 = price - risk * 2.5
    if sd and (sd + buf) < tp1 and (sd + buf) >= max_tp:
        tp2 = min(tp2, sd + buf)
    tp2 = max(tp2, max_tp)

    if tp1 > tp2 * 1.003:
        target = f"{fmt(tp2)}-{fmt(tp1)}"  # smaller-larger reads naturally
    else:
        target = fmt(tp1)

    return {
        "entry":  f"{fmt(entry_lo)}-{fmt(entry_hi)}",
        "stop":   fmt(sl),
        "target": target,
    }

# ─────────────────────────────
# FORMAT MESSAGE
# ─────────────────────────────
def format_msg(title, reaction, base_score, moves, primary_symbol,
               ai_text, signal_type, calendar_events,
               state, src_count=1) -> str:

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

    # Inject VIX and crypto flag into primary_data so signal_confidence can use them
    vix_data = _price_cache.get("^VIX")
    if vix_data:
        primary_data["vix_price"] = vix_data.get("price")
    primary_data["_is_crypto"] = primary_symbol in CRYPTO_SYMBOLS

    computed_levels = compute_trade_levels(bias, primary_data)
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
        fivem_lines.append(f"  Delta:   {of_5m['of_bias'].upper()} (cum={of_5m['cum_delta']:+,} | last={of_5m['delta']:+,})")
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
        ta_lines.append(f"  Delta:      {of_15['of_bias'].upper()} (cum={of_15['cum_delta']:+,} | last={of_15['delta']:+,})")
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
            fh_lines.append(f"  Delta:   {of_4h['of_bias'].upper()} (cum={of_4h['cum_delta']:+,} | last={of_4h['delta']:+,})")
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

    # Signal confidence
    conf_score, conf_total = signal_confidence(primary_data, bias)
    conf_bar     = confidence_bar(conf_score, conf_total)
    # Only show confidence for directional signals — WATCH/NEUTRAL always score 0 which is misleading
    conf_section = f"💪 *Confidence: {conf_score}/{conf_total} indicators aligned* |{conf_bar}|\n\n" if (conf_total > 0 and bias in ("BUY", "SELL")) else ""

    # Streak — only displayed when current bias extends an existing streak
    streak_section = get_streak_label(state, primary_symbol, bias)
    if streak_section:
        streak_section += "\n"

    # Fear & Greed — crypto uses the Crypto F&G index; all other assets use VIX
    fg_section = ""
    if primary_symbol in CRYPTO_SYMBOLS:
        fg = fetch_fear_greed()
        if fg:
            emoji = fg_emoji(fg["value"])
            fg_section = f"{emoji} *Crypto Fear & Greed: {fg['value']} — {fg['label']}*\n\n"
    elif vix_data:
        vix_price = vix_data.get("price")
        if vix_price:
            note = vix_signal_note(vix_price, bias)
            fg_section = f"📊 *Market Fear (VIX):* {vix_label(vix_price)}{note}\n\n"

    # DXY context (macro/forex signals)
    dxy_section = ""
    is_forex_or_macro = is_macro(title) or primary_symbol in {"GBPUSD=X", "EURUSD=X"}
    if is_forex_or_macro and "DX-Y.NYB" in _price_cache:
        dxy = _price_cache["DX-Y.NYB"]
        dxy_section = (
            f"💵 *US Dollar Index (DXY):* {dxy['price']} ({dxy['move']:+.2f}% today)"
            f" | {dxy.get('trend', 'n/a')}\n\n"
        )

    # Multi-source
    source_section = ""
    if src_count >= 2:
        source_section = f"📡 *{src_count} outlets reporting this story*\n\n"

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
        f"{fg_section}"
        f"{dxy_section}"
        f"{fivem_section}"
        f"{ta_section}"
        f"{hourly_section}"
        f"{fourh_section}"
        f"{daily_section}"
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
    "GC=F":     1.0,   # Gold     — alert on 1%+ move
    "ALI=F":    1.5,   # Aluminium
    "CL=F":     1.5,   # Crude Oil — alert on 1.5%+ move
    "^GSPC":    1.0,   # S&P 500
    "QQQ":      1.0,   # Nasdaq
    "GBPUSD=X": 0.4,   # GBP/USD  — forex moves less
    "EURUSD=X": 0.4,   # EUR/USD
    "BTC-USD":  3.0,   # Bitcoin  — volatile, needs bigger threshold
    "ETH-USD":  3.0,   # Ethereum
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
    weekend        = is_weekend()
    now            = now_utc().isoformat()
    cooldown_min   = 60  # don't re-fire the same break within 60 min

    for sym, data in _price_cache.items():
        if sym not in ASSET_MAP:
            continue
        if weekend and sym not in CRYPTO_SYMBOLS:
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
    weekend     = is_weekend()

    for sym, data in _price_cache.items():
        if sym not in ASSET_MAP:
            continue
        if weekend and sym not in CRYPTO_SYMBOLS:
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
def send_morning_recap(state: dict) -> bool:
    """Morning brief — replaces the old heartbeat. Fires once at 06:30 Zürich.

    Three modes depending on the day:
      Mon 06:30  → Full weekend crypto recap (Fri night → now, ~60 h window)
      Sat/Sun    → Crypto overnight recap (last 8 h, crypto-only feeds)
      Tue–Fri    → All-market overnight recap (last 8 h, all feeds)
    """
    print("  Generating morning recap...")
    zh       = zurich_now()
    weekday  = zh.weekday()   # 0=Mon … 6=Sun
    is_monday  = weekday == 0
    is_weekend = weekday >= 5  # Sat=5, Sun=6

    # ── 1. Decide scope ──────────────────────────────────────────────────────
    _crypto_feeds = [(url, "🪙 CRYPTO") for url in [
        "https://www.coindesk.com/arc/outbound/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://www.investing.com/rss/news_301.rss",
        "https://www.theblock.co/rss.xml",
        "https://blockworks.co/feed",
    ]]

    if is_monday:
        # Monday is a double session:
        #   • Crypto ran all weekend  → 62 h news window, crypto feeds
        #   • Regular markets opened at 00:00 Zürich → also scan all feeds + all assets
        # We use ALL_FEEDS with 62 h so both crypto weekend news AND the
        # early Monday morning market moves are captured in one combined recap.
        news_hours   = 62
        feeds_to_use = ALL_FEEDS          # all feeds: catches crypto weekend + early market open
        crypto_only  = False              # show ALL asset movers (markets opened at midnight)
        recap_title  = "🌅 *Good morning — Monday Recap (Weekend Crypto + Market Open)*"
        scope_label  = "Full weekend crypto + markets since 00:00 Zürich"
    elif is_weekend:
        news_hours   = 8
        feeds_to_use = _crypto_feeds
        crypto_only  = True
        recap_title  = "🌅 *Good morning — Crypto Overnight Recap*"
        scope_label  = "Last 8 hours (crypto)"
    else:
        news_hours   = 8
        feeds_to_use = ALL_FEEDS
        crypto_only  = False
        recap_title  = "🌅 *Good morning — Overnight Recap*"
        scope_label  = "Last 8 hours (all markets)"

    # ── 2. Read prices from the shared cache populated by main() ─────────────
    # Do NOT call refresh_price_cache() here — main() already called it once
    # before invoking this function. Calling it again would clear the cache,
    # make ~75 Yahoo Finance requests, hit rate limits, and leave _price_cache
    # empty for the signal loop that runs immediately after the recap.
    symbols_pool = list(CRYPTO_SYMBOLS) if crypto_only else list(ASSET_MAP.keys())
    top_movers   = []
    for sym in symbols_pool:
        d = _price_cache.get(sym)
        if d and d.get("move") is not None:
            top_movers.append((sym, d["move"], d.get("price", "n/a")))
    top_movers.sort(key=lambda x: abs(x[1]), reverse=True)

    mover_lines = []
    for sym, mv, price in top_movers[:10]:
        arrow  = "📈" if mv > 0 else "📉"
        prefix = "$" if sym not in FOREX_SYMBOLS else ""
        mover_lines.append(f"  {arrow} {friendly(sym)}: {mv:+.2f}% | {prefix}{price}")
    movers_str = "\n".join(mover_lines) if mover_lines else "  No price data available"

    # ── 3. Collect headlines ─────────────────────────────────────────────────
    cutoff    = now_utc() - timedelta(hours=news_hours)
    headlines = []
    for url, _stype in feeds_to_use:
        try:
            feed = feedparser.parse(url)
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
        except Exception:
            pass
        if len(headlines) >= 20:
            break

    headlines_str = "\n".join(f"  • {h}" for h in headlines[:20]) if headlines \
                    else f"  No major headlines in the last {news_hours} hours"

    # ── 4. Sentiment context ─────────────────────────────────────────────────
    fg  = fetch_fear_greed()
    vix = _price_cache.get("^VIX")
    fear_parts = []
    if fg:
        fear_parts.append(f"Crypto Fear & Greed: {fg['value']} — {fg['label']}")
    if vix and vix.get("price") and not crypto_only:
        fear_parts.append(f"VIX: {vix['price']} ({vix_label(vix['price'])})")
    fear_str = " | ".join(fear_parts) if fear_parts else "n/a"

    # ── 5. AI brief ──────────────────────────────────────────────────────────
    if is_monday:
        task_prompt = """Monday is a double session — crypto ran all weekend AND traditional markets just opened at midnight. Write a Monday morning brief that covers BOTH. Include:
1. WEEKEND CRYPTO RECAP: What Bitcoin and Ethereum did over the weekend and why (1-2 sentences)
2. MARKET OPEN OVERNIGHT: What traditional markets (indices, forex, gold, oil) did since they opened at midnight — any gaps, big moves, or surprises?
3. TOP 3 SETUPS FOR TODAY: The 3 clearest trade setups right now (can be any asset — crypto or traditional)
4. KEY LEVELS: One key price level for each of those 3 assets
5. MAIN RISK: The one macro factor most likely to drive volatility today"""
        asset_scope = "forex (EUR/USD, GBP/USD), indices (S&P 500, NASDAQ, DAX), gold, oil, and crypto (Bitcoin, Ethereum)"
    elif is_weekend:
        task_prompt = """Write a sharp crypto morning brief. Cover:
1. OVERNIGHT SUMMARY: What moved in crypto and why (1-2 sentences)
2. TOP 2 CRYPTO SETUPS: Bitcoin and Ethereum — current bias and key level
3. KEY LEVELS: One support and one resistance for each
4. MAIN RISK: What could flip the overnight move today"""
        asset_scope = "Bitcoin and Ethereum"
    else:
        task_prompt = """Write a sharp morning brief the trader can read in 60 seconds. Cover:
1. OVERNIGHT SUMMARY: What moved the most and why (1-2 sentences)
2. TOP 3 ASSETS TO WATCH TODAY: Which 3 assets have the clearest setup and why
3. KEY LEVELS: For each of those 3 assets, one price level to watch (support or resistance)
4. MAIN RISK: The one macro factor that could surprise markets today"""
        asset_scope = "forex (EUR/USD, GBP/USD), indices (S&P 500, NASDAQ, DAX), gold, oil, and crypto"

    prompt = f"""You are a senior trading analyst giving a concise morning briefing.
The trader watches: {asset_scope}.

Current Zürich time: {zh.strftime('%H:%M on %A %d %b')} | Scope: {scope_label}

━━━ PRICE MOVES ━━━
{movers_str}

━━━ NEWS HEADLINES ━━━
{headlines_str}

━━━ MARKET SENTIMENT ━━━
{fear_str}

━━━ YOUR TASK ━━━
{task_prompt}

Rules:
- Use full plain-English names, NEVER tickers
- Be direct — no fluff, no intro sentence, go straight to the content
- Numbers only for price levels"""

    try:
        res = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 650,
            messages   = [{"role": "user", "content": prompt}],
        )
        brief = res.content[0].text.strip()
    except Exception as e:
        print(f"  Morning recap AI error: {e}")
        brief = "AI unavailable — check markets manually."

    # ── 6. Format & send ─────────────────────────────────────────────────────
    cal_line = ""
    if not is_weekend and not is_monday:
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

    state = load_state()

    weekend = is_weekend()

    # Fetch prices first — needed for both morning recap and signals.
    print("Fetching intraday prices + daily context...")
    refresh_price_cache()

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
            send_morning_recap(state)
        except Exception as e:
            print(f"  Morning recap error (non-fatal): {e}")

    # ── MARKET OPEN GATE ─────────────────────────────────────────────────────
    if weekend:
        print("Weekend mode — running crypto signals only.")
        active_feeds = WEEKEND_FEEDS
    elif not is_market_open():
        print("Market closed — nothing to do.")
        return
    else:
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

    # ── PRE-SCAN: count how many outlets report each story ──────────────
    print("Pre-scanning feeds for multi-source stories...")
    all_entries: list  = []
    story_counts: dict = {}

    for url, signal_type in active_feeds:
        try:
            feed = feedparser.parse(url)
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

            ai_text    = analyze(title, reaction, moves, signal_type, calendar_events, primary_symbol)
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
            is_trap, trap_reason = is_at_trap_level(ai_bias, primary_data)
            if is_trap:
                print(f"  {trap_reason}")
                continue

            msg     = format_msg(
                title, reaction, sc, moves,
                primary_symbol, ai_text, signal_type, calendar_events,
                state, src_count,
            )

            if send_telegram(msg):
                signals_sent += 1
                cooldowns[primary_symbol] = now_utc().isoformat()
                if signals_sent >= MAX_SIGNALS_PER_RUN:
                    print(f"  Signal cap ({MAX_SIGNALS_PER_RUN}) reached — stopping this run.")
                    break

                bias_clean = sanitize(ai_parsed.get("BIAS", "NEUTRAL")).strip()

                update_streak(state, primary_symbol, bias_clean)

                weekly_signals.append({
                    "ts":       now_utc().isoformat(),
                    "headline": title[:80],
                    "bias":     bias_clean,
                    "score":    sc,
                    "symbol":   friendly(primary_symbol),
                })

            time.sleep(2)

        except Exception as e:
            print(f"  Entry error ({title[:50]}): {e}")

    state["seen_headlines"]     = seen_headlines
    state["__cooldowns__"]      = cooldowns
    state["__weekly_signals__"] = weekly_signals
    save_state(state)

    print(f"Done. Signals sent: {signals_sent}")


if __name__ == "__main__":
    main()
