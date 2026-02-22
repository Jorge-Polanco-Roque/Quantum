"""Historical performance chart — normalized asset lines + portfolio."""

import numpy as np
import plotly.graph_objects as go

from config import get_ticker_color


def create_performance_figure(prices_data=None, tickers=None, weights=None):
    """Build a performance chart with normalized lines per asset + portfolio.

    *prices_data* should be a dict with 'dates' and 'prices' (ticker -> list).
    """
    fig = go.Figure()

    if prices_data is None or tickers is None or weights is None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            title=dict(
                text="RENDIMIENTO HISTORICO",
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

    # Individual asset lines (thin, colored)
    portfolio_value = np.zeros(len(dates))

    for i, t in enumerate(tickers):
        if t not in prices_dict:
            continue
        p = np.array(prices_dict[t])
        p_norm = (p / p[0] - 1) * 100 if p[0] > 0 else np.zeros_like(p)

        color = get_ticker_color(t)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=p_norm,
                mode="lines",
                line=dict(color=color, width=1.2),
                name=t,
                opacity=0.6,
                hovertemplate=f"{t}: " + "%{y:.1f}%<extra></extra>",
            )
        )

        # Accumulate for portfolio
        p_ratio = p / p[0] if p[0] > 0 else np.ones_like(p)
        portfolio_value += weights[i] * p_ratio

    # Portfolio line (thick)
    portfolio_return = (portfolio_value - 1) * 100

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_return,
            mode="lines",
            line=dict(color="#00d4aa", width=3),
            name="Portafolio",
            hovertemplate="Portafolio: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        title=dict(
            text="RENDIMIENTO HISTORICO NORMALIZADO",
            font=dict(size=14, color="#e6edf3"),
        ),
        xaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
        ),
        yaxis=dict(
            title="Rendimiento (%)",
            color="#8b949e",
            gridcolor="#21262d",
            zeroline=True,
            zerolinecolor="#30363d",
        ),
        legend=dict(
            font=dict(size=9, color="#8b949e"),
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363d",
            borderwidth=1,
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        hovermode="x unified",
    )

    return fig
