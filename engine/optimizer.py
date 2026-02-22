"""Analytical portfolio optimization — Max Sharpe, Efficient Frontier, CML."""

import numpy as np
from scipy.optimize import minimize


def optimize_max_sharpe(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.04,
) -> np.ndarray:
    """Find the portfolio weights that maximize the Sharpe ratio (SLSQP)."""
    n = len(mean_returns)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = w @ mean_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
        return -(port_ret - risk_free_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


def compute_efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.04,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the efficient frontier as (volatilities, returns) arrays.

    For each target return, minimize portfolio volatility subject to:
      - weights sum to 1
      - portfolio return equals the target
      - each weight in [0, 1]
    """
    n = len(mean_returns)

    # Determine feasible return range from individual asset returns
    min_ret = float(np.min(mean_returns))
    max_ret = float(np.max(mean_returns))
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_vols = np.empty(n_points)
    frontier_rets = np.empty(n_points)
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    for i, target in enumerate(target_returns):
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mean_returns - t},
        ]

        result = minimize(
            lambda w: np.sqrt(w @ cov_matrix @ w),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        frontier_vols[i] = np.sqrt(result.x @ cov_matrix @ result.x)
        frontier_rets[i] = result.x @ mean_returns
        # Warm-start next iteration
        x0 = result.x

    return frontier_vols, frontier_rets


def compute_capital_market_line(
    risk_free_rate: float,
    optimal_return: float,
    optimal_vol: float,
    max_vol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays for the Capital Market Line.

    The CML is a straight line from (0, risk_free_rate) through the tangency
    portfolio (optimal_vol, optimal_return), extended to *max_vol*.
    """
    x = np.linspace(0, max_vol, 200)
    slope = (optimal_return - risk_free_rate) / optimal_vol
    y = risk_free_rate + slope * x
    return x, y
