"""Dashboard layout — assembles all components into the page."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from config import TICKERS
from dashboard.components.parameters import create_parameters_panel
from dashboard.components.metrics_cards import create_metrics_cards
from dashboard.components.sliders import create_sliders_panel
from dashboard.components.frontier_chart import create_frontier_figure
from dashboard.components.weights_chart import create_weights_figure
from dashboard.components.agent_panel import create_agent_panel
from dashboard.components.nl_input import create_nl_input_panel
from dashboard.components.correlation_chart import create_correlation_figure
from dashboard.components.risk_decomposition import create_risk_decomposition_figure
from dashboard.components.ensemble_table import create_ensemble_table
from dashboard.components.drawdown_chart import create_drawdown_figure
from dashboard.components.performance_chart import create_performance_figure
from dashboard.components.sentiment_panel import create_sentiment_panel


def _header():
    """Build the page header with title and badges."""
    return html.Div(
        style={"marginBottom": "24px"},
        children=[
            html.H1("QUANT PORTFOLIO OPTIMIZER", className="header-title"),
            html.Div(
                "PORTAFOLIO DINAMICO  ·  SIMULACION MONTE CARLO  ·  TEORIA MODERNA DE PORTAFOLIOS",
                className="header-subtitle",
            ),
            html.Div(
                style={"marginTop": "10px"},
                children=[
                    html.Span("10,000 SIMS", className="header-badge"),
                    html.Span("VAR 95%", className="header-badge"),
                    html.Span("MAX SHARPE", className="header-badge"),
                    html.Span("ENSEMBLE", className="header-badge"),
                    html.Span("SENTIMIENTO", className="header-badge"),
                    html.Span("NL BUILDER", className="header-badge"),
                ],
            ),
        ],
    )


def create_layout():
    """Return the full app layout."""
    return dbc.Container(
        fluid=True,
        style={"padding": "24px 32px", "maxWidth": "1600px"},
        children=[
            # Row 1 — Header
            dbc.Row(dbc.Col(_header(), width=12), className="mb-3"),

            # Row 2 — NL Builder + Parameters + Metrics
            dbc.Row(
                [
                    dbc.Col(create_nl_input_panel(), xs=12, lg=3),
                    dbc.Col(create_parameters_panel(), xs=12, lg=3),
                    dbc.Col(create_metrics_cards(), xs=12, lg=6),
                ],
                className="mb-3 g-3",
            ),

            # Row 3 — Sliders + Charts
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            id="sliders-container",
                            children=create_sliders_panel(),
                        ),
                        xs=12,
                        lg=3,
                    ),
                    dbc.Col(
                        html.Div(
                            className="graph-card",
                            children=[
                                dcc.Graph(
                                    id="frontier-chart",
                                    figure=create_frontier_figure(),
                                    config={"displayModeBar": True, "displaylogo": False},
                                    style={"height": "480px"},
                                ),
                            ],
                        ),
                        xs=12,
                        lg=5,
                    ),
                    dbc.Col(
                        html.Div(
                            className="graph-card",
                            children=[
                                dcc.Graph(
                                    id="weights-chart",
                                    figure=create_weights_figure(),
                                    config={"displayModeBar": False},
                                    style={"height": "480px"},
                                ),
                            ],
                        ),
                        xs=12,
                        lg=4,
                    ),
                ],
                className="mb-3 g-3",
            ),

            # Row 4 — Correlation + Risk Decomposition + Ensemble Table
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            className="graph-card",
                            children=[
                                dcc.Graph(
                                    id="correlation-chart",
                                    figure=create_correlation_figure(),
                                    config={"displayModeBar": False},
                                    style={"height": "400px"},
                                ),
                            ],
                        ),
                        xs=12,
                        lg=4,
                    ),
                    dbc.Col(
                        html.Div(
                            className="graph-card",
                            children=[
                                dcc.Graph(
                                    id="risk-decomposition-chart",
                                    figure=create_risk_decomposition_figure(),
                                    config={"displayModeBar": False},
                                    style={"height": "400px"},
                                ),
                            ],
                        ),
                        xs=12,
                        lg=4,
                    ),
                    dbc.Col(
                        html.Div(
                            id="ensemble-table-container",
                            children=create_ensemble_table(),
                        ),
                        xs=12,
                        lg=4,
                    ),
                ],
                className="mb-3 g-3",
            ),

            # Row 5 — Drawdown + Historical Performance
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            className="graph-card",
                            children=[
                                dcc.Graph(
                                    id="drawdown-chart",
                                    figure=create_drawdown_figure(),
                                    config={"displayModeBar": True, "displaylogo": False},
                                    style={"height": "380px"},
                                ),
                            ],
                        ),
                        xs=12,
                        lg=6,
                    ),
                    dbc.Col(
                        html.Div(
                            className="graph-card",
                            children=[
                                dcc.Graph(
                                    id="performance-chart",
                                    figure=create_performance_figure(),
                                    config={"displayModeBar": True, "displaylogo": False},
                                    style={"height": "380px"},
                                ),
                            ],
                        ),
                        xs=12,
                        lg=6,
                    ),
                ],
                className="mb-3 g-3",
            ),

            # Row 6 — Sentiment Panel
            dbc.Row(
                dbc.Col(create_sentiment_panel(), xs=12),
                className="mb-3",
            ),

            # Row 7 — Agent panel
            dbc.Row(
                dbc.Col(create_agent_panel(), xs=12),
                className="mb-4",
            ),

            # Hidden stores
            dcc.Store(id="store-mc-results"),
            dcc.Store(id="store-optimal-weights"),
            dcc.Store(id="store-annual-stats"),
            dcc.Store(id="store-tickers", data=TICKERS),
            dcc.Store(id="store-ensemble-results"),
            dcc.Store(id="store-prices-data"),
        ],
    )
