"""Portfolio risk metrics and weight normalization."""

import numpy as np
from scipy.stats import norm

from config import (
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_VAR_CONFIDENCE,
    TRADING_DAYS_PER_YEAR,
)


def calc_portfolio_metrics(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    var_confidence: float = DEFAULT_VAR_CONFIDENCE,
) -> dict:
    """Compute key portfolio metrics.

    Returns dict with: expected_return, volatility, sharpe_ratio, var.
    VaR is parametric daily VaR expressed as a positive percentage.
    """
    weights = np.asarray(weights, dtype=float)
    mean_returns = np.asarray(mean_returns, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    # Annualized return & volatility
    port_return = float(weights @ mean_returns)
    port_vol = float(np.sqrt(weights @ cov_matrix @ weights))

    # Sharpe ratio
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    # Parametric daily VaR
    daily_return = port_return / TRADING_DAYS_PER_YEAR
    daily_vol = port_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    z = norm.ppf(var_confidence)
    var = -(daily_return - z * daily_vol)  # positive number = loss

    return {
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe_ratio": sharpe,
        "var": float(var),
    }


def calc_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """Compute each asset's percentage contribution to total portfolio risk.

    Returns an array of the same length as *weights* where values sum to 1.0.
    """
    weights = np.asarray(weights, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    if port_vol < 1e-12:
        return np.ones(len(weights)) / len(weights)

    marginal = cov_matrix @ weights
    risk_contrib = weights * marginal / port_vol
    total = risk_contrib.sum()
    if total > 0:
        return risk_contrib / total
    return np.ones(len(weights)) / len(weights)


def normalize_weights(
    weights: np.ndarray,
    locked_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize *weights* so they sum to 1.0.

    If *locked_mask* (boolean array) is provided, locked weights stay fixed
    and only the unlocked weights are rescaled to fill the remaining budget.
    """
    weights = np.asarray(weights, dtype=float)

    if locked_mask is None:
        total = weights.sum()
        if total > 0:
            return weights / total
        # Fallback: equal weights
        return np.ones_like(weights) / len(weights)

    locked_mask = np.asarray(locked_mask, dtype=bool)
    locked_sum = weights[locked_mask].sum()
    remaining = 1.0 - locked_sum
    unlocked = ~locked_mask

    if remaining <= 0 or not unlocked.any():
        # Locked weights already exhaust or exceed budget; clamp and return
        result = weights.copy()
        if locked_sum > 0:
            result[locked_mask] = result[locked_mask] / locked_sum
        result[unlocked] = 0.0
        return result

    unlocked_sum = weights[unlocked].sum()
    result = weights.copy()
    if unlocked_sum > 0:
        result[unlocked] = weights[unlocked] * (remaining / unlocked_sum)
    else:
        # Distribute remaining budget equally among unlocked
        n_unlocked = unlocked.sum()
        result[unlocked] = remaining / n_unlocked

    return result
