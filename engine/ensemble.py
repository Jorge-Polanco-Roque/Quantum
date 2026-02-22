"""Ensemble optimization — 6 methods + voting/averaging."""

import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from config import DEFAULT_RISK_FREE_RATE, DEFAULT_VAR_CONFIDENCE
from engine.risk import calc_portfolio_metrics


# ── Individual optimization methods ──────────────────────────────────

def min_variance_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
) -> np.ndarray:
    """Minimize portfolio variance (SLSQP)."""
    n = len(mean_returns)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        lambda w: w @ cov_matrix @ w,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


def risk_parity_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
) -> np.ndarray:
    """Equal risk contribution portfolio (Spinu 2013 formulation)."""
    n = len(mean_returns)

    def risk_parity_objective(w):
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < 1e-12:
            return 1e10
        marginal = cov_matrix @ w
        risk_contrib = w * marginal / port_vol
        target = port_vol / n
        return np.sum((risk_contrib - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        risk_parity_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    return result.x


def max_diversification_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
) -> np.ndarray:
    """Maximize the diversification ratio: sum(w_i * sigma_i) / sigma_p."""
    n = len(mean_returns)
    asset_vols = np.sqrt(np.diag(cov_matrix))

    def neg_div_ratio(w):
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < 1e-12:
            return 1e10
        weighted_vols = w @ asset_vols
        return -(weighted_vols / port_vol)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        neg_div_ratio,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


def hrp_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado).

    Uses scipy.cluster.hierarchy for clustering on the correlation matrix.
    """
    n = len(mean_returns)
    if n <= 1:
        return np.ones(n)

    # Correlation from covariance
    vols = np.sqrt(np.diag(cov_matrix))
    vols_safe = np.where(vols > 0, vols, 1e-10)
    corr = cov_matrix / np.outer(vols_safe, vols_safe)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)

    # Distance matrix
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # ensure symmetry

    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = list(leaves_list(link))

    # Recursive bisection
    weights = np.ones(n)

    def _bisect(items):
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]

        # Cluster variance (inverse-variance allocation)
        cov_l = cov_matrix[np.ix_(left, left)]
        cov_r = cov_matrix[np.ix_(right, right)]

        inv_var_l = 1.0 / np.diag(cov_l)
        w_l = inv_var_l / inv_var_l.sum()
        var_l = w_l @ cov_l @ w_l

        inv_var_r = 1.0 / np.diag(cov_r)
        w_r = inv_var_r / inv_var_r.sum()
        var_r = w_r @ cov_r @ w_r

        alpha = 1 - var_l / (var_l + var_r) if (var_l + var_r) > 0 else 0.5

        weights[left] *= alpha
        weights[right] *= 1 - alpha

        _bisect(left)
        _bisect(right)

    _bisect(sort_ix)

    # Normalize
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def min_cvar_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
    confidence: float = 0.95,
    n_scenarios: int = 5000,
) -> np.ndarray:
    """Minimize Conditional Value-at-Risk using parametric scenarios."""
    n = len(mean_returns)

    # Generate scenarios from multivariate normal
    rng = np.random.default_rng(42)
    daily_mean = mean_returns / 252
    daily_cov = cov_matrix / 252
    scenarios = rng.multivariate_normal(daily_mean, daily_cov, size=n_scenarios)

    def cvar_objective(w):
        port_returns = scenarios @ w
        cutoff = int(n_scenarios * (1 - confidence))
        sorted_rets = np.sort(port_returns)
        cvar = -sorted_rets[:cutoff].mean() if cutoff > 0 else -sorted_rets[0]
        return cvar

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        cvar_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


def equal_weight_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
) -> np.ndarray:
    """Baseline 1/N equal-weight portfolio."""
    n = len(mean_returns)
    return np.ones(n) / n


# ── Method registry ──────────────────────────────────────────────────

OPTIMIZATION_METHODS = {
    "max_sharpe": {
        "nombre": "Max Sharpe",
        "descripcion": "Maximiza el ratio de Sharpe (SLSQP)",
        "funcion": None,  # uses existing optimize_max_sharpe
    },
    "min_variance": {
        "nombre": "Min. Varianza",
        "descripcion": "Minimiza la varianza del portafolio",
        "funcion": min_variance_portfolio,
    },
    "risk_parity": {
        "nombre": "Paridad de Riesgo",
        "descripcion": "Iguala la contribucion al riesgo de cada activo",
        "funcion": risk_parity_portfolio,
    },
    "max_diversification": {
        "nombre": "Max. Diversificacion",
        "descripcion": "Maximiza el ratio de diversificacion",
        "funcion": max_diversification_portfolio,
    },
    "hrp": {
        "nombre": "HRP",
        "descripcion": "Paridad de Riesgo Jerarquica (Lopez de Prado)",
        "funcion": hrp_portfolio,
    },
    "min_cvar": {
        "nombre": "Min. CVaR",
        "descripcion": "Minimiza el CVaR con escenarios parametricos",
        "funcion": min_cvar_portfolio,
    },
    "equal_weight": {
        "nombre": "Igual Peso",
        "descripcion": "Portafolio 1/N de referencia",
        "funcion": equal_weight_portfolio,
    },
}


# ── Orchestration ────────────────────────────────────────────────────

def run_all_methods(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
    var_conf: float = DEFAULT_VAR_CONFIDENCE,
) -> dict:
    """Execute all 7 optimization methods and return weights + metrics.

    Returns dict keyed by method name, each with 'weights' and 'metrics'.
    """
    from engine.optimizer import optimize_max_sharpe

    results = {}

    for key, method_info in OPTIMIZATION_METHODS.items():
        try:
            if key == "max_sharpe":
                weights = optimize_max_sharpe(mean_returns, cov_matrix, rf)
            else:
                weights = method_info["funcion"](mean_returns, cov_matrix, rf)

            # Ensure valid weights
            weights = np.clip(weights, 0, 1)
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum

            metrics = calc_portfolio_metrics(
                weights, mean_returns, cov_matrix, rf, var_conf
            )

            results[key] = {
                "nombre": method_info["nombre"],
                "descripcion": method_info["descripcion"],
                "weights": weights.tolist(),
                "metrics": metrics,
            }
        except Exception as exc:
            results[key] = {
                "nombre": method_info["nombre"],
                "descripcion": method_info["descripcion"],
                "weights": [],
                "metrics": {
                    "expected_return": 0,
                    "volatility": 0,
                    "sharpe_ratio": 0,
                    "var": 0,
                },
                "error": str(exc),
            }

    return results


def ensemble_vote(
    method_results: dict,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
    var_conf: float = DEFAULT_VAR_CONFIDENCE,
) -> dict:
    """Combine method results via simple average and Sharpe-weighted average.

    Returns dict with 'promedio_simple' and 'promedio_sharpe' ensembles.
    """
    valid_weights = []
    sharpe_values = []

    for key, result in method_results.items():
        w = result.get("weights", [])
        if len(w) > 0 and "error" not in result:
            valid_weights.append(np.array(w))
            sharpe_values.append(result["metrics"]["sharpe_ratio"])

    if not valid_weights:
        return {}

    weights_matrix = np.array(valid_weights)

    # Simple average
    avg_simple = weights_matrix.mean(axis=0)
    avg_simple = avg_simple / avg_simple.sum()
    metrics_simple = calc_portfolio_metrics(
        avg_simple, mean_returns, cov_matrix, rf, var_conf
    )

    # Sharpe-weighted average
    sharpe_arr = np.array(sharpe_values)
    sharpe_pos = np.maximum(sharpe_arr, 0)
    sharpe_sum = sharpe_pos.sum()

    if sharpe_sum > 0:
        sharpe_weights_norm = sharpe_pos / sharpe_sum
        avg_sharpe = (weights_matrix.T @ sharpe_weights_norm)
    else:
        avg_sharpe = avg_simple.copy()

    avg_sharpe = avg_sharpe / avg_sharpe.sum()
    metrics_sharpe = calc_portfolio_metrics(
        avg_sharpe, mean_returns, cov_matrix, rf, var_conf
    )

    return {
        "promedio_simple": {
            "nombre": "Ensemble (Promedio Simple)",
            "weights": avg_simple.tolist(),
            "metrics": metrics_simple,
        },
        "promedio_sharpe": {
            "nombre": "Ensemble (Pond. Sharpe)",
            "weights": avg_sharpe.tolist(),
            "metrics": metrics_sharpe,
        },
    }
