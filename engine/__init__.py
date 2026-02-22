"""Quant engine — Monte Carlo simulation, optimization, risk metrics."""

from .data import fetch_stock_data, compute_returns, get_annual_stats
from .monte_carlo import run_monte_carlo, get_optimal_portfolio
from .optimizer import optimize_max_sharpe, compute_efficient_frontier, compute_capital_market_line
from .risk import calc_portfolio_metrics, normalize_weights, calc_risk_contribution
from .ensemble import run_all_methods, ensemble_vote, ensemble_shrinkage, OPTIMIZATION_METHODS
from .sentiment import fetch_all_news
