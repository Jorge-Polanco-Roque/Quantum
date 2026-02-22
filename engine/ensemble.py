"""Ensemble optimization — 6 methods + voting/averaging."""

import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

import config as cfg
from config import DEFAULT_RISK_FREE_RATE, DEFAULT_VAR_CONFIDENCE
from engine.risk import calc_portfolio_metrics


# ── Constraint helpers ───────────────────────────────────────────────

def _clip_to_bounds(weights: np.ndarray, bounds) -> np.ndarray:
    """Project *weights* onto the simplex intersected with box bounds.

    Iteratively fixes bound violations by clamping violated assets and
    redistributing the excess/deficit among assets that still have room
    to move.  Returns *weights* unchanged when *bounds* is ``None``.
    """
    if bounds is None:
        return weights

    n = len(weights)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    w = weights.copy()

    # Track which assets are pinned at a bound
    pinned = np.zeros(n, dtype=bool)

    for _ in range(n * 2):
        # Clip to bounds
        w = np.clip(w, lo, hi)
        residual = 1.0 - w.sum()
        if abs(residual) < 1e-12:
            break

        # Find free (not yet pinned) assets
        free_mask = ~pinned
        if not free_mask.any():
            break

        if residual > 0:
            # Need to add weight — only assets below their upper bound can grow
            can_grow = free_mask & (w < hi - 1e-12)
            if not can_grow.any():
                break
            room = hi[can_grow] - w[can_grow]
            total_room = room.sum()
            if total_room > 0:
                add = np.minimum(room, residual * room / total_room)
                w[can_grow] += add
        else:
            # Need to remove weight — only assets above their lower bound can shrink
            can_shrink = free_mask & (w > lo + 1e-12)
            if not can_shrink.any():
                break
            room = w[can_shrink] - lo[can_shrink]
            total_room = room.sum()
            if total_room > 0:
                remove = np.minimum(room, -residual * room / total_room)
                w[can_shrink] -= remove

        # Pin assets that hit a bound this iteration
        newly_pinned = (np.abs(w - lo) < 1e-12) | (np.abs(w - hi) < 1e-12)
        pinned = pinned | newly_pinned

    return w


# ── Individual optimization methods ──────────────────────────────────

def min_variance_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Minimize portfolio variance (SLSQP)."""
    n = len(mean_returns)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if bounds is None:
        bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        lambda w: w @ cov_matrix @ w,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": cfg.SLSQP_MAX_ITER, "ftol": cfg.SLSQP_FTOL},
    )
    return result.x


def risk_parity_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
    bounds: list[tuple[float, float]] | None = None,
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
    if bounds is None:
        bounds = [(cfg.MIN_WEIGHT_BOUND, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        risk_parity_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": cfg.RISK_PARITY_MAX_ITER, "ftol": cfg.RISK_PARITY_FTOL},
    )
    return result.x


def max_diversification_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
    bounds: list[tuple[float, float]] | None = None,
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
    if bounds is None:
        bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        neg_div_ratio,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": cfg.SLSQP_MAX_ITER, "ftol": cfg.SLSQP_FTOL},
    )
    return result.x


def hrp_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
    bounds: list[tuple[float, float]] | None = None,
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

    # Apply per-asset bounds via iterative projection (HRP is analytical)
    return _clip_to_bounds(weights, bounds)


def min_cvar_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
    confidence: float = cfg.CVAR_CONFIDENCE,
    n_scenarios: int = cfg.CVAR_NUM_SCENARIOS,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Minimize Conditional Value-at-Risk using parametric scenarios."""
    n = len(mean_returns)

    # Generate scenarios from multivariate normal
    rng = np.random.default_rng(cfg.RANDOM_SEED)
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
    if bounds is None:
        bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        cvar_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": cfg.SLSQP_MAX_ITER, "ftol": cfg.SLSQP_FTOL},
    )
    return result.x


def equal_weight_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Baseline 1/N equal-weight portfolio."""
    n = len(mean_returns)
    w = np.ones(n) / n
    return _clip_to_bounds(w, bounds)


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
    bounds: list[tuple[float, float]] | None = None,
) -> dict:
    """Execute all 7 optimization methods and return weights + metrics.

    *bounds*: optional per-asset ``[(lo, hi), ...]`` forwarded to each method.
    Returns dict keyed by method name, each with 'weights' and 'metrics'.
    """
    from engine.optimizer import optimize_max_sharpe

    results = {}

    for key, method_info in OPTIMIZATION_METHODS.items():
        try:
            if key == "max_sharpe":
                weights = optimize_max_sharpe(
                    mean_returns, cov_matrix, rf, bounds=bounds,
                )
            else:
                weights = method_info["funcion"](
                    mean_returns, cov_matrix, rf, bounds=bounds,
                )

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
    bounds: list[tuple[float, float]] | None = None,
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
    avg_simple = _clip_to_bounds(avg_simple, bounds)
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
    avg_sharpe = _clip_to_bounds(avg_sharpe, bounds)
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


def ensemble_shrinkage(
    method_results: dict,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = DEFAULT_RISK_FREE_RATE,
    var_conf: float = DEFAULT_VAR_CONFIDENCE,
    delta_min: float = 0.3,
    delta_max: float = 0.7,
    bounds: list[tuple[float, float]] | None = None,
) -> dict:
    """Combine Sharpe-weighted ensemble with 1/N using adaptive shrinkage.

    The shrinkage coefficient delta is computed from the HHI (Herfindahl-
    Hirschman Index) of the Sharpe weights across the 7 methods:

        w_final = delta * w_sharpe_weighted + (1 - delta) * w_equal

    where delta is inversely proportional to HHI concentration:
      - Low HHI  (methods contribute equally) -> high delta -> trust ensemble
      - High HHI (one method dominates)       -> low delta  -> anchor to 1/N

    Returns dict with key ``"shrinkage_ensemble"`` containing weights, metrics,
    and the computed delta.
    """
    n = len(mean_returns)
    w_equal = np.ones(n) / n

    # Collect valid Sharpe ratios and weight vectors
    valid_weights = []
    sharpe_values = []

    for key, result in method_results.items():
        w = result.get("weights", [])
        if len(w) > 0 and "error" not in result:
            valid_weights.append(np.array(w))
            sharpe_values.append(result["metrics"]["sharpe_ratio"])

    if not valid_weights:
        # Fallback to equal weight
        metrics_eq = calc_portfolio_metrics(w_equal, mean_returns, cov_matrix, rf, var_conf)
        return {
            "shrinkage_ensemble": {
                "nombre": "Ensemble (Shrinkage)",
                "weights": w_equal.tolist(),
                "metrics": metrics_eq,
                "delta": 0.0,
            }
        }

    weights_matrix = np.array(valid_weights)
    sharpe_arr = np.array(sharpe_values)

    # Sharpe-weighted average (only positive Sharpe values contribute)
    sharpe_pos = np.maximum(sharpe_arr, 0.0)
    sharpe_sum = sharpe_pos.sum()

    if sharpe_sum > 0:
        sharpe_weights_norm = sharpe_pos / sharpe_sum
        w_sharpe_weighted = weights_matrix.T @ sharpe_weights_norm
    else:
        # All methods have non-positive Sharpe — fall back to simple average
        w_sharpe_weighted = weights_matrix.mean(axis=0)
        sharpe_weights_norm = np.ones(len(sharpe_values)) / len(sharpe_values)

    # Normalize Sharpe-weighted portfolio
    sw_sum = w_sharpe_weighted.sum()
    if sw_sum > 0:
        w_sharpe_weighted = w_sharpe_weighted / sw_sum

    # Compute adaptive delta from HHI of Sharpe weights
    num_methods = len(sharpe_weights_norm)
    hhi = float(np.sum(sharpe_weights_norm ** 2))
    hhi_min = 1.0 / num_methods  # perfect dispersion
    hhi_norm = (hhi - hhi_min) / (1.0 - hhi_min) if num_methods > 1 else 1.0
    hhi_norm = np.clip(hhi_norm, 0.0, 1.0)

    # Delta inversely proportional to concentration
    delta = delta_max - (delta_max - delta_min) * hhi_norm

    # Final shrinkage blend
    w_final = delta * w_sharpe_weighted + (1.0 - delta) * w_equal

    # Ensure valid weights
    w_final = np.clip(w_final, 0.0, 1.0)
    wf_sum = w_final.sum()
    if wf_sum > 0:
        w_final = w_final / wf_sum

    # Enforce per-asset bounds when user constraints are active
    w_final = _clip_to_bounds(w_final, bounds)

    metrics = calc_portfolio_metrics(w_final, mean_returns, cov_matrix, rf, var_conf)

    return {
        "shrinkage_ensemble": {
            "nombre": "Ensemble (Shrinkage)",
            "weights": w_final.tolist(),
            "metrics": metrics,
            "delta": round(float(delta), 4),
        }
    }
