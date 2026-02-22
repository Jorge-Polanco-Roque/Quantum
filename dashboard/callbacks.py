"""Dashboard callbacks — all interactivity logic.

Uses Dash pattern-matching callbacks (MATCH/ALL) so sliders work
dynamically with any set of tickers, not just the hardcoded Magnificent 7.
"""

import uuid

import numpy as np
from dash import (
    Input,
    Output,
    State,
    callback_context,
    no_update,
    ALL,
    ctx,
    html,
    dcc,
)

import config as cfg
from config import TICKERS, get_ticker_color, get_ticker_name
from engine import (
    fetch_stock_data,
    compute_returns,
    get_annual_stats,
    run_monte_carlo,
    get_optimal_portfolio,
    optimize_max_sharpe,
    compute_efficient_frontier,
    compute_capital_market_line,
    calc_portfolio_metrics,
    normalize_weights,
    calc_risk_contribution,
    run_all_methods,
    ensemble_vote,
    ensemble_shrinkage,
    fetch_all_news,
)
from dashboard.components.frontier_chart import create_frontier_figure
from dashboard.components.weights_chart import create_weights_figure
from dashboard.components.sliders import create_sliders_panel
from dashboard.components.correlation_chart import create_correlation_figure
from dashboard.components.risk_decomposition import create_risk_decomposition_figure
from dashboard.components.ensemble_table import create_ensemble_table
from dashboard.components.drawdown_chart import create_drawdown_figure
from dashboard.components.performance_chart import create_performance_figure


def _compute_split_weights(tickers, split_data):
    """Compute deterministic weights from a split specification.

    *split_data* has the form::

        {"groups": {
            "equity": {"tickers": ["AAPL", "MSFT"], "weight": 0.7},
            "bonds":  {"tickers": ["BND", "AGG"],  "weight": 0.3},
        }}

    Each group's weight is distributed equally among its tickers.
    Tickers not in any group get the residual (if any) split equally.
    Returns a ``{ticker: weight}`` dict that sums to 1.0.
    """
    groups = split_data.get("groups", {})
    weight_map = {}
    assigned = set()
    total_group_weight = 0.0

    for _label, gdata in groups.items():
        g_tickers = [t for t in gdata.get("tickers", []) if t in tickers]
        g_weight = float(gdata.get("weight", 0))
        if g_tickers:
            per_ticker = g_weight / len(g_tickers)
            for t in g_tickers:
                weight_map[t] = weight_map.get(t, 0.0) + per_ticker
                assigned.add(t)
            total_group_weight += g_weight

    # Residual tickers (in tickers list but not in any group)
    residual = [t for t in tickers if t not in assigned]
    residual_weight = max(1.0 - total_group_weight, 0.0)
    if residual and residual_weight > 0:
        per_r = residual_weight / len(residual)
        for t in residual:
            weight_map[t] = per_r
    elif residual:
        # No residual budget — give them zero (floor will fix later)
        for t in residual:
            weight_map.setdefault(t, 0.0)

    # Ensure every requested ticker is present
    for t in tickers:
        weight_map.setdefault(t, 0.0)

    # Normalize to exactly 1.0
    w_sum = sum(weight_map.values())
    if w_sum > 0:
        weight_map = {t: w / w_sum for t, w in weight_map.items()}
    else:
        eq = 1.0 / len(tickers)
        weight_map = {t: eq for t in tickers}

    return weight_map


def _build_bounds_from_constraints(tickers, constraints):
    """Convert a constraints dict to an SLSQP-compatible bounds list.

    *constraints* may contain per-ticker entries and a special ``"_all"``
    key that applies to every ticker (individual entries take priority).

    Returns ``[(lo, hi), ...]`` for each ticker, or ``None`` when
    *constraints* is empty/``None``.
    """
    if not constraints:
        return None

    global_c = constraints.get("_all", {})
    bounds = []
    for t in tickers:
        c = constraints.get(t, {})
        lo = c.get("min", global_c.get("min", 0.0))
        hi = c.get("max", global_c.get("max", 1.0))
        bounds.append((max(float(lo), 0.0), min(float(hi), 1.0)))
    return bounds


def _apply_weight_floor(weight_map, min_floor=0.02, constraints=None):
    """Ensure every ticker respects its floor and user constraints.

    When *constraints* is provided, each ticker's effective floor is
    ``max(min_floor, constraint_min)`` and ceiling is ``constraint_max``.
    Reduces heavier positions proportionally to fund the floor.
    Returns a new dict; always sums to 1.0.  Fully deterministic.
    """
    if not constraints:
        constraints = {}
    global_c = constraints.get("_all", {})

    # Build per-ticker effective floor and ceiling
    floors = {}
    caps = {}
    for t in weight_map:
        c = constraints.get(t, {})
        lo = c.get("min", global_c.get("min", 0.0))
        hi = c.get("max", global_c.get("max", 1.0))
        floors[t] = max(min_floor, float(lo))
        caps[t] = min(1.0, float(hi))

    # Iterative projection: floor → cap → normalize (converges quickly)
    result = dict(weight_map)
    for _ in range(50):
        # Apply floors
        for t in result:
            if result[t] < floors[t]:
                result[t] = floors[t]
        # Apply caps
        for t in result:
            if result[t] > caps[t]:
                result[t] = caps[t]
        # Normalize
        r_sum = sum(result.values())
        if r_sum > 0:
            result = {t: w / r_sum for t, w in result.items()}
        # Check convergence
        ok = all(
            result[t] >= floors[t] - 1e-10 and result[t] <= caps[t] + 1e-10
            for t in result
        )
        if ok and abs(sum(result.values()) - 1.0) < 1e-10:
            break

    # Final normalization for floating-point safety
    r_sum = sum(result.values())
    if r_sum > 0:
        result = {t: w / r_sum for t, w in result.items()}
    return result


def _run_full_pipeline(tickers, rf, num_sims, var_conf, preset_weights=None,
                       method="optimize", constraints=None):
    """Run the full MC + optimization + ensemble pipeline.

    *preset_weights*: ``{ticker: weight}`` dict — when provided the pipeline
    uses those weights (re-normalised) instead of running an optimizer.

    *method*: optimization method name.  When ``preset_weights`` is ``None``
    and method is ``"risk_parity"`` or ``"min_variance"``, the corresponding
    ensemble method weights are used instead of SLSQP Max Sharpe.

    *constraints*: optional ``{ticker: {"min": lo, "max": hi}, ...}`` dict.
    Converted to per-asset bounds and forwarded to all optimizers.

    Returns a tuple with all outputs needed by CB1 and CB8.
    """
    from engine.ensemble import risk_parity_portfolio, min_variance_portfolio

    # Fetch data and compute stats
    prices = fetch_stock_data(tickers=tickers)
    available_tickers = list(prices.columns)
    returns = compute_returns(prices)
    mean_ret, cov_mat = get_annual_stats(returns)
    n = len(available_tickers)

    # Build per-asset bounds from user constraints (None if no constraints)
    bounds = _build_bounds_from_constraints(available_tickers, constraints)

    # Monte Carlo simulation
    mc = run_monte_carlo(mean_ret, cov_mat, num_sims=num_sims, risk_free_rate=rf)

    # Run all 7 methods ONCE (reused for ensemble vote and ensemble shrinkage)
    all_methods = run_all_methods(mean_ret, cov_mat, rf, var_conf, bounds=bounds)

    # Optimal weights — determined by preset_weights or method
    if preset_weights is not None:
        raw = np.array([preset_weights.get(t, 0.0) for t in available_tickers])
        w_sum = raw.sum()
        if w_sum > 0:
            opt_weights = raw / w_sum
        else:
            opt_weights = optimize_max_sharpe(
                mean_ret, cov_mat, risk_free_rate=rf, bounds=bounds,
            )
    elif method == "ensemble":
        shrink = ensemble_shrinkage(
            all_methods, mean_ret, cov_mat, rf, var_conf,
            delta_min=cfg.ENSEMBLE_DELTA_MIN,
            delta_max=cfg.ENSEMBLE_DELTA_MAX,
            bounds=bounds,
        )
        opt_weights = np.array(shrink["shrinkage_ensemble"]["weights"])
    elif method == "risk_parity":
        opt_weights = risk_parity_portfolio(mean_ret, cov_mat, rf, bounds=bounds)
    elif method == "min_variance":
        opt_weights = min_variance_portfolio(mean_ret, cov_mat, rf, bounds=bounds)
    else:
        opt_weights = optimize_max_sharpe(
            mean_ret, cov_mat, risk_free_rate=rf, bounds=bounds,
        )

    # Apply weight floor — ensure every ticker gets >= 2% AND respects constraints
    floor_map = _apply_weight_floor(
        {t: float(w) for t, w in zip(available_tickers, opt_weights)},
        constraints=constraints,
    )
    opt_weights = np.array([floor_map[t] for t in available_tickers])
    opt_metrics = calc_portfolio_metrics(opt_weights, mean_ret, cov_mat, rf, var_conf)
    optimal_info = {
        "weights": opt_weights.tolist(),
        "return": opt_metrics["expected_return"],
        "volatility": opt_metrics["volatility"],
        "sharpe": opt_metrics["sharpe_ratio"],
    }

    # Efficient frontier + CML
    ef_vols, ef_rets = compute_efficient_frontier(mean_ret, cov_mat, rf, n_points=cfg.EF_NUM_POINTS)
    max_vol = float(np.max(mc["volatilities"])) * cfg.CML_MAX_VOL_MULTIPLIER
    cml_x, cml_y = compute_capital_market_line(
        rf, optimal_info["return"], optimal_info["volatility"], max_vol
    )

    # Ensemble vote (reuses all_methods already computed above)
    ensemble_results = ensemble_vote(
        all_methods, mean_ret, cov_mat, rf, var_conf, bounds=bounds,
    )
    # Add shrinkage ensemble to the table
    shrink_result = ensemble_shrinkage(
        all_methods, mean_ret, cov_mat, rf, var_conf,
        delta_min=cfg.ENSEMBLE_DELTA_MIN,
        delta_max=cfg.ENSEMBLE_DELTA_MAX,
        bounds=bounds,
    )
    combined_ensemble = {**all_methods, **ensemble_results, **shrink_result}

    # Risk contribution
    risk_contrib = calc_risk_contribution(opt_weights, cov_mat)

    # Prices store for drawdown/performance
    prices_store = {
        "dates": [str(d) for d in prices.index],
        "prices": {t: prices[t].tolist() for t in available_tickers},
    }

    # Build figures
    frontier_fig = create_frontier_figure(
        mc_results=mc,
        ef_vols=ef_vols,
        ef_rets=ef_rets,
        cml_x=cml_x,
        cml_y=cml_y,
        optimal=optimal_info,
        risk_free_rate=rf,
    )
    weights_fig = create_weights_figure(available_tickers, opt_weights)
    corr_fig = create_correlation_figure(available_tickers, cov_mat)
    risk_decomp_fig = create_risk_decomposition_figure(
        available_tickers, opt_weights, risk_contrib
    )
    ensemble_table = create_ensemble_table(combined_ensemble)
    drawdown_fig = create_drawdown_figure(
        prices_store, available_tickers, opt_weights.tolist()
    )
    performance_fig = create_performance_figure(
        prices_store, available_tickers, opt_weights.tolist()
    )

    # Metrics strings
    sharpe_str = f"{opt_metrics['sharpe_ratio']:.3f}"
    return_str = f"{opt_metrics['expected_return']*100:.2f}%"
    vol_str = f"{opt_metrics['volatility']*100:.2f}%"
    var_str = f"{opt_metrics['var']*100:.2f}%"

    # Serializable stores
    mc_store = {
        "returns": mc["returns"].tolist(),
        "volatilities": mc["volatilities"].tolist(),
        "sharpe_ratios": mc["sharpe_ratios"].tolist(),
    }
    opt_store = {
        "tickers": available_tickers,
        "weights": opt_weights.tolist(),
    }
    stats_store = {
        "tickers": available_tickers,
        "mean_returns": mean_ret.tolist(),
        "cov_matrix": cov_mat.tolist(),
    }

    # Serializable ensemble store
    ensemble_store = {}
    for key, data in combined_ensemble.items():
        ensemble_store[key] = {
            "nombre": data.get("nombre", key),
            "weights": data.get("weights", []),
            "metrics": data.get("metrics", {}),
        }

    total = 100.0
    bar_style = {"width": "100%"}
    bar_label = "Total: 100.0%"

    slider_vals = [round(w * 100, 2) for w in opt_weights]
    new_sliders = _build_sliders_with_weights(available_tickers, slider_vals)

    return (
        frontier_fig, weights_fig,
        sharpe_str, return_str, vol_str, var_str,
        mc_store, opt_store, stats_store,
        bar_style, bar_label,
        new_sliders,
        corr_fig, risk_decomp_fig, ensemble_table,
        drawdown_fig, performance_fig,
        ensemble_store, prices_store,
    )


def _compute_position_styles(position):
    """Return (layout_style, main_style, sidebar_style) for a chat position.

    The app-layout uses ``position: fixed; inset: 0`` so it is exactly the
    viewport.  The dashboard uses ``flex: 1 1 0%`` (fills remaining space)
    and the sidebar uses ``flex: 0 0 300px`` (rigid).

    Left/Right: only change ``order`` and borders — never touch ``flex`` or
    ``width`` so the CSS rules apply cleanly.

    Top/Bottom: switch to ``flex-direction: column``.  Override sidebar to
    full-width + fixed height, and dashboard to ``flex: 1`` for height fill.
    """
    if position == "left":
        layout_style = {"flexDirection": "row"}
        main_style = {"order": "1"}
        sidebar_style = {
            "order": "-1",
            "borderLeft": "none",
            "borderRight": "1px solid #30363d",
            "borderTop": "none",
            "borderBottom": "none",
        }
    elif position == "top":
        layout_style = {"flexDirection": "column"}
        main_style = {"order": "1", "minHeight": "0"}
        sidebar_style = {
            "order": "-1",
            "flex": "0 0 25vh",
            "width": "100%",
            "maxWidth": "100%",
            "height": "25vh",
            "borderLeft": "none",
            "borderRight": "none",
            "borderTop": "none",
            "borderBottom": "1px solid #30363d",
        }
    elif position == "bottom":
        layout_style = {"flexDirection": "column"}
        main_style = {"minHeight": "0"}
        sidebar_style = {
            "flex": "0 0 25vh",
            "width": "100%",
            "maxWidth": "100%",
            "height": "25vh",
            "borderLeft": "none",
            "borderRight": "none",
            "borderTop": "1px solid #30363d",
            "borderBottom": "none",
        }
    else:  # right (default)
        layout_style = {"flexDirection": "row"}
        main_style = {}
        sidebar_style = {
            "borderLeft": "1px solid #30363d",
            "borderRight": "none",
            "borderTop": "none",
            "borderBottom": "none",
        }
    return layout_style, main_style, sidebar_style


def register_callbacks(app):
    """Register all Dash callbacks on *app*."""

    # ── Callback 0: Switch chat position ────────────────────────────
    @app.callback(
        [
            Output("app-layout", "style"),
            Output("dashboard-main", "style"),
            Output("chat-sidebar", "style"),
            Output("store-chat-position", "data"),
            Output("chat-pos-left", "className"),
            Output("chat-pos-top", "className"),
            Output("chat-pos-bottom", "className"),
            Output("chat-pos-right", "className"),
        ],
        [
            Input("chat-pos-left", "n_clicks"),
            Input("chat-pos-top", "n_clicks"),
            Input("chat-pos-right", "n_clicks"),
            Input("chat-pos-bottom", "n_clicks"),
        ],
        State("store-chat-position", "data"),
        prevent_initial_call=True,
    )
    def switch_chat_position(n_left, n_top, n_right, n_bottom, current_pos):
        trigger = ctx.triggered_id
        pos_map = {
            "chat-pos-left": "left",
            "chat-pos-top": "top",
            "chat-pos-right": "right",
            "chat-pos-bottom": "bottom",
        }
        new_pos = pos_map.get(trigger, current_pos or "right")

        layout_style, main_style, sidebar_style = _compute_position_styles(new_pos)

        base = "chat-pos-btn"
        active = "chat-pos-btn chat-pos-btn-active"
        return (
            layout_style,
            main_style,
            sidebar_style,
            new_pos,
            active if new_pos == "left" else base,
            active if new_pos == "top" else base,
            active if new_pos == "bottom" else base,
            active if new_pos == "right" else base,
        )

    # ── Callback 1: EJECUTAR — run full MC pipeline ──────────────────
    @app.callback(
        [
            Output("frontier-chart", "figure"),
            Output("weights-chart", "figure"),
            Output("metric-sharpe-value", "children"),
            Output("metric-return-value", "children"),
            Output("metric-volatility-value", "children"),
            Output("metric-var-value", "children"),
            Output("store-mc-results", "data"),
            Output("store-optimal-weights", "data"),
            Output("store-annual-stats", "data"),
            Output("total-bar-fill", "style"),
            Output("total-bar-label", "children"),
            Output("sliders-container", "children"),
            Output("correlation-chart", "figure"),
            Output("risk-decomposition-chart", "figure"),
            Output("ensemble-table-container", "children"),
            Output("drawdown-chart", "figure"),
            Output("performance-chart", "figure"),
            Output("store-ensemble-results", "data"),
            Output("store-prices-data", "data"),
        ],
        Input("btn-ejecutar", "n_clicks"),
        [
            State("input-rf", "value"),
            State("input-sims", "value"),
            State("input-var-level", "value"),
            State("store-tickers", "data"),
        ],
        prevent_initial_call=True,
    )
    def run_simulation(n_clicks, rf_pct, num_sims, var_level_pct, tickers):
        if not n_clicks:
            return [no_update] * 19

        tickers = tickers or TICKERS
        rf = (rf_pct or 4) / 100.0
        num_sims = int(num_sims or 10000)
        var_conf = (var_level_pct or 95) / 100.0

        try:
            result = _run_full_pipeline(tickers, rf, num_sims, var_conf)
        except Exception as exc:
            raise Exception(
                f"Error ejecutando simulacion: {exc}. "
                "Verifica tu conexion a internet e intenta de nuevo."
            ) from exc

        return list(result)

    # ── Callback 2: Slider changes — recalculate metrics ─────────────
    @app.callback(
        [
            Output("metric-sharpe-value", "children", allow_duplicate=True),
            Output("metric-return-value", "children", allow_duplicate=True),
            Output("metric-volatility-value", "children", allow_duplicate=True),
            Output("metric-var-value", "children", allow_duplicate=True),
            Output("total-bar-fill", "style", allow_duplicate=True),
            Output("total-bar-label", "children", allow_duplicate=True),
        ],
        Input({"type": "slider", "index": ALL}, "value"),
        [
            State({"type": "slider", "index": ALL}, "id"),
            State("store-annual-stats", "data"),
            State("input-rf", "value"),
            State("input-var-level", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_metrics_on_slider(slider_vals, slider_ids, stats_data, rf_pct, var_level_pct):
        if not slider_vals:
            return [no_update] * 6

        n = len(slider_vals)

        # Total bar always updates
        total = sum(v or 0 for v in slider_vals)
        bar_style = {"width": f"{min(total, 100):.1f}%"}
        bar_label = f"Total: {total:.1f}%"

        if not stats_data or stats_data.get("mean_returns") is None:
            return [no_update, no_update, no_update, no_update, bar_style, bar_label]

        # Ensure slider order matches stats tickers
        stats_tickers = stats_data.get("tickers", [])
        slider_ticker_order = [sid["index"] for sid in slider_ids]

        # Build weight array aligned to stats ticker order
        slider_map = dict(zip(slider_ticker_order, slider_vals))
        aligned_vals = [slider_map.get(t, 0) or 0 for t in stats_tickers]

        try:
            mean_ret = np.array(stats_data["mean_returns"], dtype=float)
            cov_mat = np.array(stats_data["cov_matrix"], dtype=float)
        except (ValueError, TypeError):
            return [no_update, no_update, no_update, no_update, bar_style, bar_label]

        rf = (rf_pct or 4) / 100.0
        var_conf = (var_level_pct or 95) / 100.0

        weights = np.array([v / 100.0 for v in aligned_vals])
        w_sum = weights.sum()
        if w_sum > 0:
            weights_normed = weights / w_sum
        else:
            weights_normed = np.ones(len(stats_tickers)) / len(stats_tickers)

        metrics = calc_portfolio_metrics(weights_normed, mean_ret, cov_mat, rf, var_conf)

        sharpe_str = f"{metrics['sharpe_ratio']:.3f}"
        return_str = f"{metrics['expected_return']*100:.2f}%"
        vol_str = f"{metrics['volatility']*100:.2f}%"
        var_str = f"{metrics['var']*100:.2f}%"

        return [sharpe_str, return_str, vol_str, var_str, bar_style, bar_label]

    # ── Callback: Update weight displays when sliders change ─────────
    @app.callback(
        Output({"type": "weight-display", "index": ALL}, "children"),
        Input({"type": "slider", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def update_weight_displays(slider_vals):
        if not slider_vals:
            return [no_update]
        return [f"{(v or 0):.1f}%" for v in slider_vals]

    # ── Callback 3: OPTIMO — reset sliders to optimal ────────────────
    @app.callback(
        Output("sliders-container", "children", allow_duplicate=True),
        Input("btn-optimo", "n_clicks"),
        [
            State("store-optimal-weights", "data"),
            State("store-tickers", "data"),
        ],
        prevent_initial_call=True,
    )
    def reset_to_optimal(n_clicks, opt_data, tickers):
        if not n_clicks or opt_data is None:
            return no_update
        opt_tickers = opt_data.get("tickers", tickers or TICKERS)
        opt_weights = opt_data.get("weights", [])
        if not opt_weights:
            return no_update
        slider_vals = [round(w * 100, 2) for w in opt_weights]
        return _build_sliders_with_weights(opt_tickers, slider_vals)

    # ── Callback 4: IGUAL — equal weights ────────────────────────────
    @app.callback(
        Output("sliders-container", "children", allow_duplicate=True),
        Input("btn-equal", "n_clicks"),
        State("store-tickers", "data"),
        prevent_initial_call=True,
    )
    def set_equal_weights(n_clicks, tickers):
        if not n_clicks:
            return no_update
        tickers = tickers or TICKERS
        eq = round(100.0 / len(tickers), 2)
        return _build_sliders_with_weights(tickers, [eq] * len(tickers))

    # ── Callback 5: RANDOM — random weights ──────────────────────────
    @app.callback(
        Output("sliders-container", "children", allow_duplicate=True),
        Input("btn-random", "n_clicks"),
        State("store-tickers", "data"),
        prevent_initial_call=True,
    )
    def set_random_weights(n_clicks, tickers):
        if not n_clicks:
            return no_update
        tickers = tickers or TICKERS
        n = len(tickers)
        w = np.random.dirichlet(np.ones(n)) * 100
        return _build_sliders_with_weights(tickers, [round(float(x), 2) for x in w])

    # ── Callback 6: NORMALIZE — normalize respecting locks ───────────
    @app.callback(
        Output("sliders-container", "children", allow_duplicate=True),
        Input("btn-normalize", "n_clicks"),
        [
            State({"type": "slider", "index": ALL}, "value"),
            State({"type": "slider", "index": ALL}, "id"),
            State({"type": "lock", "index": ALL}, "value"),
            State({"type": "lock", "index": ALL}, "id"),
            State("store-tickers", "data"),
        ],
        prevent_initial_call=True,
    )
    def normalize_slider_weights(n_clicks, slider_vals, slider_ids, lock_vals, lock_ids, tickers):
        if not n_clicks:
            return no_update

        tickers = tickers or TICKERS

        # Build maps from ticker to value/lock
        slider_map = {sid["index"]: v for sid, v in zip(slider_ids, slider_vals)}
        lock_map = {lid["index"]: bool(v) for lid, v in zip(lock_ids, lock_vals)}

        ordered_vals = [slider_map.get(t, 0) or 0 for t in tickers]
        ordered_locks = [lock_map.get(t, False) for t in tickers]

        weights = np.array([v / 100.0 for v in ordered_vals])
        locked_mask = np.array(ordered_locks)

        normed = normalize_weights(weights, locked_mask)
        return _build_sliders_with_weights(
            tickers,
            [round(float(w) * 100, 2) for w in normed],
            lock_states=dict(zip(tickers, ordered_locks)),
        )

    # ── Callback 7: Run AI Agents (manual re-run) ──────────────────
    @app.callback(
        Output("agent-output", "children"),
        Input("btn-run-agents", "n_clicks"),
        [
            State({"type": "slider", "index": ALL}, "value"),
            State({"type": "slider", "index": ALL}, "id"),
            State("store-annual-stats", "data"),
            State("input-rf", "value"),
            State("input-var-level", "value"),
            State("store-tickers", "data"),
            State("store-ensemble-results", "data"),
        ],
        prevent_initial_call=True,
    )
    def run_agents_manual(n_clicks, slider_vals, slider_ids, stats_data, rf_pct, var_level_pct, tickers, ensemble_data):
        if not n_clicks:
            return no_update

        tickers = tickers or TICKERS

        if stats_data is None:
            return "_Primero ejecuta la simulacion Monte Carlo._"

        stats_tickers = stats_data.get("tickers", tickers)
        mean_ret = np.array(stats_data["mean_returns"])
        cov_mat = np.array(stats_data["cov_matrix"])
        rf = (rf_pct or 4) / 100.0
        var_conf = (var_level_pct or 95) / 100.0

        # Build weight array aligned to stats tickers (uses current slider values)
        slider_map = {sid["index"]: v for sid, v in zip(slider_ids, slider_vals)}
        aligned_vals = [slider_map.get(t, 0) or 0 for t in stats_tickers]

        weights = np.array([v / 100.0 for v in aligned_vals])
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum

        metrics = calc_portfolio_metrics(weights, mean_ret, cov_mat, rf, var_conf)

        portfolio_data = {
            "weights": {t: float(w) for t, w in zip(stats_tickers, weights)},
            "tickers": stats_tickers,
            "expected_return": metrics["expected_return"],
            "volatility": metrics["volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "var": metrics["var"],
        }

        return _execute_agent_analysis(portfolio_data, ensemble_data)

    # ── Callback 7b: Auto-trigger AI Agents on portfolio generation ──
    @app.callback(
        Output("agent-output", "children", allow_duplicate=True),
        Input("store-optimal-weights", "data"),
        [
            State("store-annual-stats", "data"),
            State("input-rf", "value"),
            State("input-var-level", "value"),
            State("store-tickers", "data"),
            State("store-ensemble-results", "data"),
        ],
        prevent_initial_call=True,
    )
    def auto_run_agents(opt_data, stats_data, rf_pct, var_level_pct, tickers, ensemble_data):
        if opt_data is None or stats_data is None:
            return no_update

        stats_tickers = stats_data.get("tickers", tickers or TICKERS)
        opt_weights = np.array(opt_data.get("weights", []))
        if len(opt_weights) == 0:
            return no_update

        mean_ret = np.array(stats_data["mean_returns"])
        cov_mat = np.array(stats_data["cov_matrix"])
        rf = (rf_pct or 4) / 100.0
        var_conf = (var_level_pct or 95) / 100.0

        metrics = calc_portfolio_metrics(opt_weights, mean_ret, cov_mat, rf, var_conf)

        portfolio_data = {
            "weights": {t: float(w) for t, w in zip(stats_tickers, opt_weights)},
            "tickers": stats_tickers,
            "expected_return": metrics["expected_return"],
            "volatility": metrics["volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "var": metrics["var"],
        }

        return _execute_agent_analysis(portfolio_data, ensemble_data)

    # ── Callback 8: NL Portfolio Builder ─────────────────────────────
    @app.callback(
        [
            Output("store-tickers", "data"),
            Output("sliders-container", "children", allow_duplicate=True),
            Output("nl-build-output", "children"),
            # Trigger auto-run of MC pipeline
            Output("frontier-chart", "figure", allow_duplicate=True),
            Output("weights-chart", "figure", allow_duplicate=True),
            Output("metric-sharpe-value", "children", allow_duplicate=True),
            Output("metric-return-value", "children", allow_duplicate=True),
            Output("metric-volatility-value", "children", allow_duplicate=True),
            Output("metric-var-value", "children", allow_duplicate=True),
            Output("store-mc-results", "data", allow_duplicate=True),
            Output("store-optimal-weights", "data", allow_duplicate=True),
            Output("store-annual-stats", "data", allow_duplicate=True),
            Output("total-bar-fill", "style", allow_duplicate=True),
            Output("total-bar-label", "children", allow_duplicate=True),
            Output("correlation-chart", "figure", allow_duplicate=True),
            Output("risk-decomposition-chart", "figure", allow_duplicate=True),
            Output("ensemble-table-container", "children", allow_duplicate=True),
            Output("drawdown-chart", "figure", allow_duplicate=True),
            Output("performance-chart", "figure", allow_duplicate=True),
            Output("store-ensemble-results", "data", allow_duplicate=True),
            Output("store-prices-data", "data", allow_duplicate=True),
        ],
        Input("btn-build-portfolio", "n_clicks"),
        [
            State("nl-input", "value"),
            State("input-rf", "value"),
            State("input-sims", "value"),
            State("input-var-level", "value"),
        ],
        prevent_initial_call=True,
    )
    def build_portfolio_from_nl(n_clicks, nl_text, rf_pct, num_sims, var_level_pct):
        if not n_clicks or not nl_text or not nl_text.strip():
            return [no_update] * 21

        rf = (rf_pct or 4) / 100.0
        num_sims = int(num_sims or 10000)
        var_conf = (var_level_pct or 95) / 100.0

        # Try to run the NL agent
        try:
            from agents import PortfolioBuilderAgent
            agent = PortfolioBuilderAgent()
            result = agent.run(nl_text.strip())
        except ImportError:
            result = {"error": "Dependencias LangGraph/LangChain no instaladas."}
        except Exception as exc:
            result = {"error": f"Error del agente: {exc}"}

        if "error" in result:
            error_msg = result["error"]
            reasoning = result.get("reasoning", "")
            output_msg = html.Div(
                f"Error: {error_msg}",
                style={"color": "#fb923c", "fontSize": "0.75rem"},
            )
            if reasoning:
                output_msg = html.Div([
                    html.Div(f"Error: {error_msg}", style={"color": "#fb923c", "fontSize": "0.75rem"}),
                    html.Details([
                        html.Summary("Detalle", style={"color": "#8b949e", "fontSize": "0.7rem"}),
                        html.P(reasoning[:500], style={"color": "#8b949e", "fontSize": "0.7rem"}),
                    ]),
                ])
            return [no_update] * 2 + [output_msg] + [no_update] * 18

        # Parse agent result — deterministic weight computation
        new_tickers = result.get("tickers", [])
        method = result.get("method", "ensemble")
        split_data = result.get("split", None)
        constraints = result.get("constraints", None)
        reasoning = result.get("reasoning", "Portafolio construido exitosamente.")

        # ── Normalize constraints: remove tickers with max<=0 (exclusions) ──
        if constraints:
            excluded = set()
            for tk, c in list(constraints.items()):
                if tk == "_all":
                    continue
                c_max = c.get("max")
                c_min = c.get("min")
                # max=0 → user wants to exclude this ticker
                if c_max is not None and float(c_max) <= 0:
                    excluded.add(tk)
                    del constraints[tk]
                # min=0 with no max → LLM misread "0% en X" as min=0;
                # since min=0 is already the default, treat as exclusion
                elif c_min is not None and float(c_min) <= 0 and c_max is None and len(c) == 1:
                    excluded.add(tk)
                    del constraints[tk]
            if excluded:
                new_tickers = [t for t in new_tickers if t not in excluded]
            # If constraints dict is empty (or only _all left with no entries), clear it
            remaining = {k: v for k, v in constraints.items() if k != "_all"}
            global_c = constraints.get("_all", {})
            if not remaining and not global_c:
                constraints = None

        # Compute weights deterministically based on method
        if method == "preset" and result.get("weights"):
            # User gave exact percentages — use them directly
            raw_weights = result["weights"]
            weight_map = {t: float(raw_weights.get(t, 0)) for t in new_tickers}
            # Normalize in case they don't sum to 1
            w_sum = sum(weight_map.values())
            if w_sum > 0:
                weight_map = {t: w / w_sum for t, w in weight_map.items()}
        elif method == "split" and split_data:
            weight_map = _compute_split_weights(new_tickers, split_data)
        elif method == "equal_weight":
            eq = 1.0 / len(new_tickers) if new_tickers else 0
            weight_map = {t: eq for t in new_tickers}
        elif method in ("risk_parity", "min_variance", "ensemble"):
            # Pipeline will compute via the corresponding engine method
            weight_map = None
        else:
            # "optimize" or unknown → SLSQP Max Sharpe (default)
            weight_map = None

        if not new_tickers:
            return [no_update] * 2 + [
                html.Div("No se seleccionaron tickers.", style={"color": "#fb923c", "fontSize": "0.75rem"})
            ] + [no_update] * 18

        # Now run the full pipeline with the agent-selected tickers
        try:
            pipeline_result = _run_full_pipeline(
                new_tickers, rf, num_sims, var_conf,
                preset_weights=weight_map, method=method,
                constraints=constraints,
            )
        except Exception as exc:
            return [no_update] * 2 + [
                html.Div(f"Error descargando datos: {exc}", style={"color": "#fb923c", "fontSize": "0.75rem"})
            ] + [no_update] * 18

        # Extract available tickers from the pipeline (some may have been filtered)
        available_tickers = pipeline_result[7]["tickers"]  # opt_store has tickers

        # Detect which tickers were filtered by yfinance
        filtered_out = [t for t in new_tickers if t not in available_tickers]

        # Determine weight method label
        METHOD_LABELS = {
            "preset": "Pesos del usuario",
            "ensemble": "Ensemble Shrinkage (δ adaptativo)",
            "optimize": "SLSQP Max Sharpe",
            "equal_weight": "Pesos Iguales (1/N)",
            "risk_parity": "Paridad de Riesgo",
            "min_variance": "Minima Varianza",
            "split": "Split deterministico",
        }
        peso_method = METHOD_LABELS.get(method, method)

        msg_children = [
            html.Div(
                f"Portafolio construido: {', '.join(available_tickers)}",
                style={"color": "#00d4aa", "fontSize": "0.75rem", "fontWeight": "600"},
            ),
            html.Div(
                f"Metodo de pesos: {peso_method}",
                style={"color": "#58a6ff", "fontSize": "0.7rem", "marginTop": "2px"},
            ),
        ]

        if constraints:
            parts = []
            for tk, c in constraints.items():
                if tk == "_all":
                    lo = c.get("min")
                    hi = c.get("max")
                    if lo is not None:
                        parts.append(f"todos min {lo*100:.0f}%")
                    if hi is not None:
                        parts.append(f"todos max {hi*100:.0f}%")
                else:
                    lo = c.get("min")
                    hi = c.get("max")
                    if lo is not None and hi is not None:
                        parts.append(f"{tk} [{lo*100:.0f}%-{hi*100:.0f}%]")
                    elif lo is not None:
                        parts.append(f"{tk} min {lo*100:.0f}%")
                    elif hi is not None:
                        parts.append(f"{tk} max {hi*100:.0f}%")
            if parts:
                msg_children.append(
                    html.Div(
                        f"Restricciones: {', '.join(parts)}",
                        style={"color": "#d2a8ff", "fontSize": "0.7rem", "marginTop": "2px"},
                    )
                )

        if filtered_out:
            msg_children.append(
                html.Div(
                    f"Tickers no disponibles en Yahoo Finance: {', '.join(filtered_out)}",
                    style={"color": "#fb923c", "fontSize": "0.7rem", "marginTop": "2px"},
                )
            )

        msg_children.append(
            html.Div(
                reasoning[:200],
                style={"color": "#8b949e", "fontSize": "0.7rem", "marginTop": "4px"},
            ),
        )

        output_msg = html.Div(msg_children)

        return [
            available_tickers,           # store-tickers
            pipeline_result[11],         # sliders-container
            output_msg,                  # nl-build-output
            pipeline_result[0],          # frontier-chart
            pipeline_result[1],          # weights-chart
            pipeline_result[2],          # sharpe
            pipeline_result[3],          # return
            pipeline_result[4],          # vol
            pipeline_result[5],          # var
            pipeline_result[6],          # store-mc-results
            pipeline_result[7],          # store-optimal-weights
            pipeline_result[8],          # store-annual-stats
            pipeline_result[9],          # bar style
            pipeline_result[10],         # bar label
            pipeline_result[12],         # correlation-chart
            pipeline_result[13],         # risk-decomposition-chart
            pipeline_result[14],         # ensemble-table-container
            pipeline_result[15],         # drawdown-chart
            pipeline_result[16],         # performance-chart
            pipeline_result[17],         # store-ensemble-results
            pipeline_result[18],         # store-prices-data
        ]

    # ── Callback: Sentiment / News + Fundamental Analysis (manual) ──
    @app.callback(
        Output("sentiment-output", "children"),
        Input("btn-refresh-news", "n_clicks"),
        [
            State("store-tickers", "data"),
            State("store-optimal-weights", "data"),
            State("store-annual-stats", "data"),
            State("store-ensemble-results", "data"),
            State("input-rf", "value"),
            State("input-var-level", "value"),
        ],
        prevent_initial_call=True,
    )
    def refresh_news(n_clicks, tickers, opt_data, stats_data, ensemble_data, rf_pct, var_level_pct):
        if not n_clicks:
            return no_update
        return _build_sentiment_output(tickers, opt_data, stats_data, ensemble_data, rf_pct, var_level_pct)

    # ── Callback: Sentiment auto-trigger on portfolio generation ────
    @app.callback(
        Output("sentiment-output", "children", allow_duplicate=True),
        Input("store-optimal-weights", "data"),
        [
            State("store-tickers", "data"),
            State("store-annual-stats", "data"),
            State("store-ensemble-results", "data"),
            State("input-rf", "value"),
            State("input-var-level", "value"),
        ],
        prevent_initial_call=True,
    )
    def auto_refresh_news(opt_data, tickers, stats_data, ensemble_data, rf_pct, var_level_pct):
        if opt_data is None:
            return no_update
        return _build_sentiment_output(tickers, opt_data, stats_data, ensemble_data, rf_pct, var_level_pct)

    # ── Callback: Chat init (generate thread_id on page load) ───────
    @app.callback(
        Output("store-chat-thread-id", "data"),
        Input("store-chat-thread-id", "data"),
    )
    def init_chat_thread(current_id):
        if current_id:
            return no_update
        return str(uuid.uuid4())

    # ── Callback: Chat send message ─────────────────────────────────
    @app.callback(
        [
            Output("chat-messages", "children"),
            Output("chat-input", "value"),
            Output("store-chat-history", "data"),
            Output("chat-loading-target", "children"),
        ],
        [Input("chat-send", "n_clicks"), Input("chat-input", "n_submit")],
        [
            State("chat-input", "value"),
            State("store-chat-history", "data"),
            State("store-chat-thread-id", "data"),
            State("store-optimal-weights", "data"),
            State("store-annual-stats", "data"),
            State("store-ensemble-results", "data"),
            State("store-tickers", "data"),
        ],
        prevent_initial_call=True,
    )
    def send_chat_message(
        n_clicks, n_submit, text, history, thread_id,
        opt_data, stats_data, ensemble_data, tickers,
    ):
        if not text or not text.strip():
            return no_update, no_update, no_update, no_update

        text = text.strip()
        history = history or []

        # Add user message
        history.append({"role": "user", "content": text})

        # Check if portfolio exists
        if opt_data is None:
            reply = "Primero ejecuta una simulacion (boton **EJECUTAR** o usa el **Constructor NL**) para que pueda ayudarte con tu portafolio."
            history.append({"role": "assistant", "content": reply})
            return _build_chat_bubbles(history), "", history, ""

        # Build portfolio context
        portfolio_context = _build_portfolio_context(
            opt_data, stats_data, ensemble_data, tickers
        )

        # Chat with agent
        try:
            from agents.chatbot import ChatbotAgent
            chatbot = ChatbotAgent()
            reply = chatbot.chat(text, portfolio_context, thread_id or "default")
        except ImportError:
            reply = "**Error:** Dependencias de LangGraph/LangChain no instaladas."
        except Exception as exc:
            reply = f"**Error:** `{exc}`"

        history.append({"role": "assistant", "content": reply})
        return _build_chat_bubbles(history), "", history, ""


def _build_sentiment_output(tickers, opt_data, stats_data, ensemble_data, rf_pct, var_level_pct):
    """Fetch news, score sentiment, and optionally run fundamental analysis.

    Shared by the manual button callback and the auto-trigger callback.
    """
    tickers = tickers or TICKERS

    try:
        sentiment_data = fetch_all_news(tickers, max_per_ticker=cfg.MAX_NEWS_PER_TICKER)
    except Exception as exc:
        return html.Div(
            f"Error obteniendo noticias: {exc}",
            style={"color": "#fb923c", "fontSize": "0.75rem"},
        )

    # Build formatted news output
    children = []
    score_general = sentiment_data.get("score_general", 0)

    # General score header
    if score_general > 0.1:
        general_color = "#00d4aa"
        general_label = "POSITIVO"
    elif score_general < -0.1:
        general_color = "#f87171"
        general_label = "NEGATIVO"
    else:
        general_color = "#fbbf24"
        general_label = "NEUTRAL"

    children.append(
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "16px"},
            children=[
                html.Span(
                    f"Sentimiento General: {general_label}",
                    style={"color": general_color, "fontSize": "0.8rem", "fontWeight": "700"},
                ),
                html.Span(
                    f"({score_general:+.3f})",
                    style={"color": general_color, "fontSize": "0.75rem"},
                ),
            ],
        )
    )

    # Per-ticker news
    for ticker in tickers:
        ticker_data = sentiment_data.get("por_ticker", {}).get(ticker, {})
        avg_score = ticker_data.get("score_promedio", 0)
        noticias = ticker_data.get("noticias", [])

        score_color = "#00d4aa" if avg_score > 0.05 else "#f87171" if avg_score < -0.05 else "#fbbf24"

        ticker_header = html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "10px"},
            children=[
                html.Span(
                    ticker,
                    style={
                        "color": get_ticker_color(ticker),
                        "fontWeight": "700",
                        "fontSize": "0.75rem",
                    },
                ),
                html.Span(
                    f"{avg_score:+.2f}",
                    className="sentiment-score-positive" if avg_score >= 0 else "sentiment-score-negative",
                    style={"color": score_color},
                ),
            ],
        )
        children.append(ticker_header)

        for noticia in noticias:
            s = noticia.get("score", 0)
            s_color = "#00d4aa" if s > 0 else "#f87171" if s < 0 else "#8b949e"
            children.append(
                html.Div(
                    className="sentiment-item",
                    children=[
                        html.Span(
                            f"[{s:+.1f}]",
                            style={"color": s_color, "fontSize": "0.65rem", "minWidth": "40px"},
                        ),
                        html.Span(
                            noticia.get("title", "")[:80],
                            style={"color": "#c9d1d9", "fontSize": "0.7rem", "marginLeft": "6px"},
                        ),
                    ],
                )
            )

    # Run fundamental analysis agent if portfolio data is available
    if opt_data and stats_data:
        stats_tickers = stats_data.get("tickers", tickers)
        opt_weights = np.array(opt_data.get("weights", []))
        mean_ret = np.array(stats_data["mean_returns"])
        cov_mat = np.array(stats_data["cov_matrix"])
        rf = (rf_pct or 4) / 100.0
        var_conf = (var_level_pct or 95) / 100.0

        metrics = calc_portfolio_metrics(opt_weights, mean_ret, cov_mat, rf, var_conf)
        portfolio_data = {
            "weights": {t: float(w) for t, w in zip(stats_tickers, opt_weights)},
            "tickers": stats_tickers,
            "expected_return": metrics["expected_return"],
            "volatility": metrics["volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "var": metrics["var"],
        }

        try:
            from agents.fundamental_analyst import run_fundamental_analysis
            analysis = run_fundamental_analysis(
                portfolio_data, sentiment_data, ensemble_data
            )
        except Exception as exc:
            analysis = f"_Error en analisis fundamental: {exc}_"

        # Separator + fundamental analysis
        children.append(
            html.Hr(style={"borderColor": "#30363d", "margin": "16px 0"})
        )
        children.append(
            html.Div(
                "ANALISIS FUNDAMENTAL AI",
                style={
                    "color": "#00d4aa",
                    "fontSize": "0.75rem",
                    "fontWeight": "700",
                    "marginBottom": "8px",
                },
            )
        )
        children.append(
            dcc.Markdown(
                analysis,
                style={"color": "#c9d1d9", "fontSize": "0.75rem", "lineHeight": "1.6"},
            )
        )
    else:
        children.append(
            html.Div(
                "Ejecuta la simulacion primero para obtener el analisis fundamental AI.",
                style={"color": "#8b949e", "fontSize": "0.7rem", "marginTop": "12px"},
            )
        )

    return html.Div(children)


def _build_chat_bubbles(history):
    """Convert chat history list into Dash div elements."""
    bubbles = []
    for msg in history:
        if msg["role"] == "user":
            bubbles.append(
                html.Div(
                    msg["content"],
                    className="chat-bubble chat-bubble-user",
                )
            )
        else:
            bubbles.append(
                html.Div(
                    dcc.Markdown(
                        msg["content"],
                        style={"margin": "0", "fontSize": "inherit", "lineHeight": "inherit"},
                    ),
                    className="chat-bubble chat-bubble-assistant",
                )
            )
    return bubbles


def _build_portfolio_context(opt_data, stats_data, ensemble_data, tickers):
    """Build a context dict from dashboard stores for the chatbot."""
    context = {}

    if opt_data:
        context["tickers"] = opt_data.get("tickers", tickers or [])
        context["weights"] = opt_data.get("weights", [])

    if stats_data and opt_data:
        stats_tickers = stats_data.get("tickers", tickers or [])
        opt_weights = np.array(opt_data.get("weights", []))
        mean_ret = np.array(stats_data.get("mean_returns", []))
        cov_mat = np.array(stats_data.get("cov_matrix", []))

        if len(opt_weights) > 0 and len(mean_ret) > 0:
            try:
                metrics = calc_portfolio_metrics(opt_weights, mean_ret, cov_mat, 0.04, 0.95)
                context["metrics"] = metrics
            except Exception:
                pass

    if ensemble_data:
        context["ensemble"] = ensemble_data

    return context


def _execute_agent_analysis(portfolio_data, ensemble_data):
    """Run the multi-agent debate system on portfolio data.

    Shared by the manual button (CB7) and auto-trigger (CB7b).
    """
    tickers = portfolio_data.get("tickers", [])

    # Fetch sentiment for agents
    try:
        sentiment_data = fetch_all_news(tickers, max_per_ticker=cfg.MAX_NEWS_PER_TICKER)
    except Exception:
        sentiment_data = {"por_ticker": {}, "score_general": 0, "resumen": ""}

    try:
        from agents import PortfolioAgentsGraph
        graph = PortfolioAgentsGraph()
        result = graph.run(
            portfolio_data,
            ensemble_data=ensemble_data,
            sentiment_data=sentiment_data,
        )
        return result
    except ImportError:
        return _fallback_analysis(portfolio_data)
    except Exception as exc:
        error_msg = str(exc)
        if "api" in error_msg.lower() or "key" in error_msg.lower() or "auth" in error_msg.lower():
            return (
                "**Error de API:** No se pudo conectar al servicio LLM. "
                "Verifica que tu API key esta configurada en `.env`.\n\n"
                f"Detalle: `{error_msg}`"
            )
        return f"**Error al ejecutar agentes:** `{error_msg}`"


def _build_sliders_with_weights(tickers, slider_vals, lock_states=None):
    """Build the sliders panel with specific weight values pre-set."""
    from dash import html, dcc
    import dash_bootstrap_components as dbc

    if lock_states is None:
        lock_states = {}

    rows = []
    for t, val in zip(tickers, slider_vals):
        color = get_ticker_color(t)
        locked = lock_states.get(t, False)
        rows.append(
            html.Div(
                className="slider-row",
                children=[
                    html.Span(t, className="slider-ticker-label", style={"color": color}, title=get_ticker_name(t)),
                    dcc.Checklist(
                        id={"type": "lock", "index": t},
                        options=[{"label": "", "value": "locked"}],
                        value=["locked"] if locked else [],
                        className="lock-checkbox",
                        style={"minWidth": "24px"},
                        inputStyle={"marginRight": "0"},
                    ),
                    html.Div(
                        dcc.Slider(
                            id={"type": "slider", "index": t},
                            min=0, max=100, step=0.1, value=val,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        className="slider-container",
                    ),
                    html.Span(
                        f"{val:.1f}%",
                        id={"type": "weight-display", "index": t},
                        className="slider-weight-display",
                    ),
                ],
            )
        )

    total = sum(slider_vals)
    total_bar = html.Div(
        className="total-bar-container",
        children=[
            html.Div(
                className="total-bar-track",
                children=[
                    html.Div(
                        id="total-bar-fill",
                        className="total-bar-fill",
                        style={"width": f"{min(total, 100):.1f}%"},
                    ),
                ],
            ),
            html.Div(
                id="total-bar-label",
                className="total-bar-label",
                children=f"Total: {total:.1f}%",
            ),
        ],
    )

    action_buttons = dbc.Row(
        [
            dbc.Col(html.Button("IGUAL", id="btn-equal", className="btn-equal", n_clicks=0), width=4),
            dbc.Col(html.Button("RANDOM", id="btn-random", className="btn-random", n_clicks=0), width=4),
            dbc.Col(html.Button("NORM.", id="btn-normalize", className="btn-normalize", n_clicks=0), width=4),
        ],
        className="g-2",
    )

    return html.Div(
        className="card",
        children=[
            html.Div("PESOS DEL PORTAFOLIO", className="section-title"),
            *rows,
            total_bar,
            action_buttons,
        ],
    )


def _fallback_analysis(portfolio_data):
    """Generate a basic analysis when the agent graph is not available."""
    weights = portfolio_data["weights"]
    er = portfolio_data["expected_return"]
    vol = portfolio_data["volatility"]
    sharpe = portfolio_data["sharpe_ratio"]
    var = portfolio_data["var"]

    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_w[:3]
    top3_conc = sum(w for _, w in top3)

    lines = [
        "## Analisis del Portafolio\n",
        "### Metricas Clave",
        f"- **Sharpe Ratio:** {sharpe:.4f}",
        f"- **Retorno Esperado:** {er:.2%}",
        f"- **Volatilidad:** {vol:.2%}",
        f"- **VaR (diario):** {var:.2%}\n",
        "### Asignacion",
    ]
    for t, w in sorted_w:
        bar = "\u2588" * int(w * 40)
        lines.append(f"- **{t}:** {w:.2%} {bar}")

    lines.append(f"\n### Concentracion")
    lines.append(f"Top 3 posiciones ({', '.join(t for t,_ in top3)}): **{top3_conc:.1%}** del portafolio.")

    if sharpe > 1.0:
        lines.append("\n> El Sharpe Ratio es atractivo (>1.0), indicando buen rendimiento ajustado al riesgo.")
    elif sharpe > 0.5:
        lines.append("\n> El Sharpe Ratio es moderado. Considera optimizar la asignacion.")
    else:
        lines.append("\n> El Sharpe Ratio es bajo (<0.5). La relacion riesgo-rendimiento podria mejorarse.")

    lines.append("\n---\n_Analisis generado sin agentes AI. Configura tu API key para un analisis completo._")

    return "\n".join(lines)
