"""Centralized configuration for Quant Portfolio Optimizer."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Magnificent 7 tickers ──────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# ── Ticker display colors (matching reference designs) ─────────────
TICKER_COLORS = {
    "AAPL": "#34d399",   # green
    "MSFT": "#f472b6",   # pink
    "GOOGL": "#fbbf24",  # yellow
    "AMZN": "#38bdf8",   # sky blue
    "NVDA": "#a78bfa",   # purple
    "META": "#2dd4bf",   # teal
    "TSLA": "#fb923c",   # orange
}

# Extended palette for dynamically-added tickers
_EXTRA_COLORS = [
    "#60a5fa",  # blue
    "#f87171",  # red
    "#4ade80",  # lime
    "#e879f9",  # fuchsia
    "#facc15",  # yellow
    "#22d3ee",  # cyan
    "#c084fc",  # violet
    "#fb7185",  # rose
    "#a3e635",  # lime-green
    "#fdba74",  # apricot
]


_TICKER_NAME_CACHE: dict[str, str] = {}


def get_ticker_name(ticker: str) -> str:
    """Return the short company name for *ticker* via yfinance (cached).

    Returns the ticker itself if the name cannot be resolved.
    """
    if ticker in _TICKER_NAME_CACHE:
        return _TICKER_NAME_CACHE[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ticker
        _TICKER_NAME_CACHE[ticker] = name
        return name
    except Exception:
        _TICKER_NAME_CACHE[ticker] = ticker
        return ticker


def get_ticker_names(tickers: list[str]) -> dict[str, str]:
    """Return a dict of {ticker: company_name} for all tickers (cached)."""
    return {t: get_ticker_name(t) for t in tickers}


def get_ticker_color(ticker: str) -> str:
    """Return a display color for *ticker*.

    Known tickers use TICKER_COLORS; unknown ones get a deterministic
    color from _EXTRA_COLORS based on a hash of the ticker string.
    """
    if ticker in TICKER_COLORS:
        return TICKER_COLORS[ticker]
    idx = hash(ticker) % len(_EXTRA_COLORS)
    return _EXTRA_COLORS[idx]

# ── Optimization defaults ─────────────────────────────────────────
SLSQP_MAX_ITER = 1000
SLSQP_FTOL = 1e-12
RISK_PARITY_MAX_ITER = 2000
RISK_PARITY_FTOL = 1e-14
MIN_WEIGHT_BOUND = 1e-6
EF_NUM_POINTS = 80
CML_MAX_VOL_MULTIPLIER = 1.1

# ── Simulation ────────────────────────────────────────────────────
CVAR_NUM_SCENARIOS = 5000
CVAR_CONFIDENCE = 0.95
RANDOM_SEED = 42

# ── Sentiment / News ─────────────────────────────────────────────
MAX_NEWS_PER_TICKER = 3

# ── Agent ─────────────────────────────────────────────────────────
AGENT_RECURSION_LIMIT = 40

# ── Monte Carlo defaults ───────────────────────────────────────────
DEFAULT_NUM_SIMULATIONS = 10_000
DEFAULT_RISK_FREE_RATE = 0.04       # 4%
DEFAULT_VAR_CONFIDENCE = 0.95       # 95%
TRADING_DAYS_PER_YEAR = 252
DATA_PERIOD = "2y"                  # 2 years of historical data

# ── LLM / Agent settings ──────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MAX_DEBATE_ROUNDS = 2

# ── Dashboard settings ─────────────────────────────────────────────
DASH_DEBUG = os.getenv("DASH_DEBUG", "true").lower() == "true"
DASH_PORT = int(os.getenv("DASH_PORT", "8050"))

# ── Theme colors ───────────────────────────────────────────────────
THEME = {
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "text_muted": "#8b949e",
    "accent": "#00d4aa",
    "accent2": "#58a6ff",
    "sharpe_color": "#00d4aa",
    "return_color": "#f472b6",
    "vol_color": "#a78bfa",
    "var_color": "#fb923c",
}
