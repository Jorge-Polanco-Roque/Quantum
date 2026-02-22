"""Weights horizontal bar chart builder."""

import plotly.graph_objects as go

from config import get_ticker_color, get_ticker_name


def create_weights_figure(tickers=None, weights=None):
    """Build a horizontal bar chart for portfolio weights.

    *weights* should be in 0-1 decimal form; chart displays as percentages.
    """
    fig = go.Figure()

    if tickers is None or weights is None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            title=dict(
                text="PESOS OPTIMOS",
                font=dict(size=14, color="#8b949e"),
            ),
            xaxis=dict(title="Peso (%)", color="#8b949e"),
            yaxis=dict(color="#8b949e"),
            annotations=[
                dict(
                    text="Esperando resultados...",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="#8b949e"),
                )
            ],
            margin=dict(l=60, r=20, t=50, b=50),
        )
        return fig

    # Sort by weight descending
    paired = sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)
    sorted_tickers = [t for t, _ in paired]
    sorted_weights = [w * 100 for _, w in paired]
    colors = [get_ticker_color(t) for t in sorted_tickers]

    names = [get_ticker_name(t) for t in sorted_tickers]
    hover_texts = [
        f"{t}: {n}<br>Peso: {w:.2f}%"
        for t, n, w in zip(sorted_tickers, names, sorted_weights)
    ]

    fig.add_trace(
        go.Bar(
            x=sorted_weights,
            y=sorted_tickers,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{w:.1f}%" for w in sorted_weights],
            textposition="auto",
            textfont=dict(size=11, color="#0d1117", family="JetBrains Mono"),
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        title=dict(
            text="PESOS OPTIMOS",
            font=dict(size=14, color="#e6edf3"),
        ),
        xaxis=dict(
            title="Peso (%)",
            color="#8b949e",
            gridcolor="#21262d",
            range=[0, max(sorted_weights) * 1.15] if sorted_weights else [0, 100],
        ),
        yaxis=dict(
            color="#e6edf3",
            autorange="reversed",
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        bargap=0.3,
    )

    return fig
