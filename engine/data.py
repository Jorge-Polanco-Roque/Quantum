"""Stock data fetching and return computation."""

import numpy as np
import pandas as pd
import yfinance as yf

from config import TICKERS, TRADING_DAYS_PER_YEAR, DATA_PERIOD


def fetch_stock_data(
    tickers: list[str] = TICKERS,
    period: str = DATA_PERIOD,
) -> pd.DataFrame:
    """Download close prices for *tickers* over *period* via yfinance.

    Handles both old (<=0.2.x) and new (>=1.x) yfinance column formats.
    Raises RuntimeError if download yields no data.
    """
    df = yf.download(tickers, period=period, progress=False)

    if df.empty:
        raise RuntimeError(
            "yfinance no devolvio datos. Verifica tu conexion a internet "
            "y que los tickers sean validos."
        )

    # yfinance >=1.x returns MultiIndex ('Price', 'Ticker')
    # yfinance <=0.2.x with multi-tickers returns MultiIndex too
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    elif "Close" in df.columns:
        df = df[["Close"]]

    # Drop any column-level name artifacts
    if hasattr(df.columns, "name"):
        df.columns.name = None

    # Reorder columns to match requested ticker order
    available = [t for t in tickers if t in df.columns]
    df = df[available]

    return df.dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price DataFrame."""
    return np.log(prices / prices.shift(1)).dropna()


def get_annual_stats(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return annualized (mean_returns, cov_matrix) from daily log returns."""
    mean_returns = returns.mean().values * TRADING_DAYS_PER_YEAR
    cov_matrix = returns.cov().values * TRADING_DAYS_PER_YEAR
    return mean_returns, cov_matrix
