"""Correlation heatmap chart builder."""

import numpy as np
import plotly.graph_objects as go


def create_correlation_figure(tickers=None, cov_matrix=None):
    """Build a correlation heatmap from the covariance matrix."""
    fig = go.Figure()

    if tickers is None or cov_matrix is None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            title=dict(
                text="CORRELACION",
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

    cov = np.array(cov_matrix)
    vols = np.sqrt(np.diag(cov))
    vols_safe = np.where(vols > 0, vols, 1e-10)
    corr = cov / np.outer(vols_safe, vols_safe)
    corr = np.clip(corr, -1, 1)

    # Annotations: show correlation values
    annotations = []
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f"{corr[i, j]:.2f}",
                    font=dict(size=10, color="#e6edf3" if abs(corr[i, j]) < 0.7 else "#0d1117"),
                    showarrow=False,
                )
            )

    fig.add_trace(
        go.Heatmap(
            z=corr,
            x=tickers,
            y=tickers,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(
                title=dict(text="Corr", font=dict(size=10, color="#8b949e")),
                tickfont=dict(size=9, color="#8b949e"),
                thickness=12,
                len=0.8,
            ),
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        title=dict(
            text="MATRIZ DE CORRELACION",
            font=dict(size=14, color="#e6edf3"),
        ),
        xaxis=dict(color="#8b949e", tickfont=dict(size=10)),
        yaxis=dict(color="#8b949e", autorange="reversed", tickfont=dict(size=10)),
        annotations=annotations,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig
