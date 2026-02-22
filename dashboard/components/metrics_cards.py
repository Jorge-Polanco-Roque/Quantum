"""Metric cards — Sharpe, Return, Volatility, VaR."""

from dash import html
import dash_bootstrap_components as dbc


def _metric_card(card_id, label, subtitle, css_class):
    """Build a single metric card."""
    return html.Div(
        id=card_id,
        className=f"metric-card {css_class}",
        children=[
            html.Div(label, className="metric-label"),
            html.Div("\u2014", className="metric-value", id=f"{card_id}-value"),
            html.Div(subtitle, className="metric-subtitle"),
        ],
    )


def create_metrics_cards():
    """Return a row of 4 metric cards."""
    return dbc.Row(
        [
            dbc.Col(
                _metric_card(
                    "metric-sharpe",
                    "SHARPE RATIO",
                    "Rendimiento ajustado al riesgo",
                    "sharpe",
                ),
                xs=6,
                md=3,
            ),
            dbc.Col(
                _metric_card(
                    "metric-return",
                    "RETORNO ESPERADO",
                    "Rendimiento anualizado",
                    "return",
                ),
                xs=6,
                md=3,
            ),
            dbc.Col(
                _metric_card(
                    "metric-volatility",
                    "VOLATILIDAD",
                    "Desviacion estandar anualizada",
                    "volatility",
                ),
                xs=6,
                md=3,
            ),
            dbc.Col(
                _metric_card(
                    "metric-var",
                    "VAR 95%",
                    "Perdida maxima diaria esperada",
                    "var",
                ),
                xs=6,
                md=3,
            ),
        ],
        className="g-3",
    )
