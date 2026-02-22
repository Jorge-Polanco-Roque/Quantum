"""Portfolio drawdown chart builder."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import THEME


def create_drawdown_figure(prices_data=None, tickers=None, weights=None):
    """Build a drawdown area chart for the weighted portfolio.

    *prices_data* should be a dict with 'dates' (list of str) and
    'prices' (dict of ticker -> list of floats).
    """
    fig = go.Figure()

    if prices_data is None or tickers is None or weights is None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            title=dict(
                text="DRAWDOWN DEL PORTAFOLIO",
                font=dict(size=14, color="#8b949e"),
            ),
            annotations=[
                dict(
                    text="Esperando resultados...",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="#8b949e"),
                )
            ],
            margin=dict(l=50, r=20, t=50, b=50),
        )
        return fig

    dates = prices_data["dates"]
    prices_dict = prices_data["prices"]

    # Build price matrix and compute weighted portfolio value
    n_dates = len(dates)
    portfolio_value = np.zeros(n_dates)

    for i, t in enumerate(tickers):
        if t in prices_dict:
            p = np.array(prices_dict[t])
            # Normalize to 1.0 at start
            p_norm = p / p[0] if p[0] > 0 else p
            portfolio_value += weights[i] * p_norm

    # Compute drawdown
    running_max = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - running_max) / np.where(running_max > 0, running_max, 1)
    drawdown_pct = drawdown * 100

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown_pct,
            fill="tozeroy",
            fillcolor="rgba(251, 146, 60, 0.2)",
            line=dict(color="#fb923c", width=1.5),
            name="Drawdown",
            hovertemplate="Fecha: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    # Max drawdown annotation
    max_dd_idx = np.argmin(drawdown_pct)
    max_dd_val = drawdown_pct[max_dd_idx]

    fig.add_annotation(
        x=dates[max_dd_idx],
        y=max_dd_val,
        text=f"Max: {max_dd_val:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#fb923c",
        font=dict(size=10, color="#fb923c"),
        bgcolor="#161b22",
        bordercolor="#fb923c",
        borderwidth=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        title=dict(
            text="DRAWDOWN DEL PORTAFOLIO",
            font=dict(size=14, color="#e6edf3"),
        ),
        xaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
        ),
        yaxis=dict(
            title="Drawdown (%)",
            color="#8b949e",
            gridcolor="#21262d",
            zeroline=True,
            zerolinecolor="#30363d",
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        showlegend=False,
    )

    return fig
