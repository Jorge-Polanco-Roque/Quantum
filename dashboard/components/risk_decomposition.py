"""Risk decomposition chart — allocation vs risk contribution."""

import numpy as np
import plotly.graph_objects as go


def create_risk_decomposition_figure(
    tickers=None, weights=None, risk_contributions=None
):
    """Build grouped bar chart comparing % allocated vs % risk contribution."""
    fig = go.Figure()

    if tickers is None or weights is None or risk_contributions is None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            title=dict(
                text="DESCOMPOSICION DE RIESGO",
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

    weights_pct = np.array(weights) * 100
    risk_pct = np.array(risk_contributions) * 100

    fig.add_trace(
        go.Bar(
            x=tickers,
            y=weights_pct,
            name="Asignacion (%)",
            marker=dict(color="#00d4aa", opacity=0.85),
            text=[f"{v:.1f}%" for v in weights_pct],
            textposition="auto",
            textfont=dict(size=9, color="#0d1117"),
            hovertemplate="%{x}: %{y:.1f}% asignado<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=tickers,
            y=risk_pct,
            name="Contrib. Riesgo (%)",
            marker=dict(color="#f472b6", opacity=0.85),
            text=[f"{v:.1f}%" for v in risk_pct],
            textposition="auto",
            textfont=dict(size=9, color="#0d1117"),
            hovertemplate="%{x}: %{y:.1f}% riesgo<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        title=dict(
            text="ASIGNACION vs CONTRIBUCION AL RIESGO",
            font=dict(size=14, color="#e6edf3"),
        ),
        xaxis=dict(color="#8b949e"),
        yaxis=dict(
            title="Porcentaje (%)",
            color="#8b949e",
            gridcolor="#21262d",
        ),
        barmode="group",
        bargap=0.3,
        bargroupgap=0.1,
        legend=dict(
            font=dict(size=10, color="#8b949e"),
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363d",
            borderwidth=1,
        ),
        margin=dict(l=50, r=20, t=50, b=50),
    )

    return fig
