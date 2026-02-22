"""Efficient Frontier chart builder."""

import plotly.graph_objects as go


def create_frontier_figure(
    mc_results=None,
    ef_vols=None,
    ef_rets=None,
    cml_x=None,
    cml_y=None,
    optimal=None,
    current_weights=None,
    current_metrics=None,
    risk_free_rate=0.04,
):
    """Build the efficient frontier go.Figure.

    All arrays should be in decimal form (0.25 = 25%).
    The chart displays values as percentages.
    """
    fig = go.Figure()

    if mc_results is None:
        # Empty placeholder
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            title=dict(
                text="FRONTERA EFICIENTE",
                font=dict(size=14, color="#8b949e"),
            ),
            xaxis=dict(title="Volatilidad Anualizada (%)", color="#8b949e"),
            yaxis=dict(title="Retorno Esperado (%)", color="#8b949e"),
            annotations=[
                dict(
                    text="Haz clic en EJECUTAR para iniciar la simulacion",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="#8b949e"),
                )
            ],
            margin=dict(l=50, r=20, t=50, b=50),
        )
        return fig

    # Monte Carlo scatter
    vols_pct = mc_results["volatilities"] * 100
    rets_pct = mc_results["returns"] * 100
    sharpes = mc_results["sharpe_ratios"]

    fig.add_trace(
        go.Scattergl(
            x=vols_pct,
            y=rets_pct,
            mode="markers",
            marker=dict(
                size=3,
                color=sharpes,
                colorscale="Viridis",
                opacity=0.6,
                colorbar=dict(
                    title=dict(text="Sharpe", font=dict(size=10, color="#8b949e")),
                    tickfont=dict(size=9, color="#8b949e"),
                    thickness=12,
                    len=0.6,
                ),
            ),
            name="Simulaciones MC",
            hovertemplate=(
                "Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<br>"
                "Sharpe: %{marker.color:.3f}<extra></extra>"
            ),
        )
    )

    # Efficient Frontier line
    if ef_vols is not None and ef_rets is not None:
        fig.add_trace(
            go.Scatter(
                x=ef_vols * 100,
                y=ef_rets * 100,
                mode="lines",
                line=dict(color="#00d4aa", width=3),
                name="Frontera Eficiente",
            )
        )

    # Capital Market Line
    if cml_x is not None and cml_y is not None:
        fig.add_trace(
            go.Scatter(
                x=cml_x * 100,
                y=cml_y * 100,
                mode="lines",
                line=dict(color="#fbbf24", width=2, dash="dash"),
                name="Linea del Mercado de Capitales",
            )
        )

    # Risk-free point
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[risk_free_rate * 100],
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=10,
                color="#fbbf24",
                line=dict(width=1, color="#0d1117"),
            ),
            name=f"Tasa Libre ({risk_free_rate*100:.1f}%)",
            hovertemplate="Tasa Libre: %{y:.2f}%<extra></extra>",
        )
    )

    # Optimal portfolio star
    if optimal is not None:
        fig.add_trace(
            go.Scatter(
                x=[optimal["volatility"] * 100],
                y=[optimal["return"] * 100],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=15,
                    color="#ffffff",
                    line=dict(width=1, color="#00d4aa"),
                ),
                name=f"Optimo (Sharpe {optimal['sharpe']:.3f})",
                hovertemplate=(
                    "OPTIMO<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>"
                ),
            )
        )

    # Current portfolio marker
    if current_metrics is not None:
        fig.add_trace(
            go.Scatter(
                x=[current_metrics["volatility"] * 100],
                y=[current_metrics["expected_return"] * 100],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=12,
                    color="#f472b6",
                    line=dict(width=2, color="#ffffff"),
                ),
                name="Tu Portafolio",
                hovertemplate=(
                    "TU PORTAFOLIO<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        title=dict(
            text="FRONTERA EFICIENTE — Monte Carlo",
            font=dict(size=14, color="#e6edf3"),
        ),
        xaxis=dict(
            title="Volatilidad Anualizada (%)",
            color="#8b949e",
            gridcolor="#21262d",
            zeroline=False,
        ),
        yaxis=dict(
            title="Retorno Esperado (%)",
            color="#8b949e",
            gridcolor="#21262d",
            zeroline=False,
        ),
        legend=dict(
            font=dict(size=10, color="#8b949e"),
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363d",
            borderwidth=1,
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        hovermode="closest",
    )

    return fig
