"""Monte Carlo portfolio simulation — fully vectorized with numpy."""

import numpy as np

from config import DEFAULT_NUM_SIMULATIONS, DEFAULT_RISK_FREE_RATE


def run_monte_carlo(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_sims: int = DEFAULT_NUM_SIMULATIONS,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict:
    """Run *num_sims* Monte Carlo portfolio simulations.

    Returns a dict with keys: weights, returns, volatilities, sharpe_ratios.
    """
    n_assets = len(mean_returns)

    # Dirichlet(1,...,1) gives uniform random weights on the simplex
    weights = np.random.dirichlet(np.ones(n_assets), size=num_sims)

    # Portfolio expected returns: w @ mu  (vectorized)
    port_returns = weights @ mean_returns

    # Portfolio volatilities via einsum: sqrt(w @ Sigma @ w^T) per row
    port_variances = np.einsum("ij,jk,ik->i", weights, cov_matrix, weights)
    port_vols = np.sqrt(port_variances)

    # Sharpe ratios
    sharpe_ratios = (port_returns - risk_free_rate) / port_vols

    return {
        "weights": weights,
        "returns": port_returns,
        "volatilities": port_vols,
        "sharpe_ratios": sharpe_ratios,
    }


def get_optimal_portfolio(mc_results: dict) -> dict:
    """Extract the portfolio with the highest Sharpe ratio from MC results."""
    idx = int(np.argmax(mc_results["sharpe_ratios"]))
    return {
        "index": idx,
        "weights": mc_results["weights"][idx],
        "return": float(mc_results["returns"][idx]),
        "volatility": float(mc_results["volatilities"][idx]),
        "sharpe": float(mc_results["sharpe_ratios"][idx]),
    }
