"""Dashboard layout — assembles all components into the page."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from config import TICKERS
from dashboard.components.parameters import create_parameters_content
from dashboard.components.metrics_cards import create_metrics_cards
from dashboard.components.sliders import create_sliders_panel
from dashboard.components.frontier_chart import create_frontier_figure
from dashboard.components.weights_chart import create_weights_figure
from dashboard.components.agent_panel import create_agent_panel
from dashboard.components.nl_input import create_nl_input_content
from dashboard.components.correlation_chart import create_correlation_figure
from dashboard.components.risk_decomposition import create_risk_decomposition_figure
from dashboard.components.ensemble_table import create_ensemble_table
from dashboard.components.drawdown_chart import create_drawdown_figure
from dashboard.components.performance_chart import create_performance_figure
from dashboard.components.sentiment_panel import create_sentiment_panel
from dashboard.components.chat_widget import create_chat_panel


def _header():
    """Build the compact page header."""
    return html.Div(
        className="dash-header",
        children=[
            html.Div(
                children=[
                    html.H1("QUANT PORTFOLIO OPTIMIZER", className="header-title"),
                    html.Div(
                        "MONTE CARLO  ·  TEORIA MODERNA DE PORTAFOLIOS  ·  ENSEMBLE",
                        className="header-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="header-badges-row",
                children=[
                    html.Span("10K SIMS", className="header-badge"),
                    html.Span("VAR 95%", className="header-badge"),
                    html.Span("SHARPE", className="header-badge"),
                    html.Span("ENSEMBLE", className="header-badge"),
                    html.Span("SENTIMENT", className="header-badge"),
                    html.Span("NL BUILD", className="header-badge"),
                    html.Span("CHATBOT", className="header-badge"),
                ],
            ),
        ],
    )


def _dashboard_content():
    """Build the main dashboard content (left 2/3)."""
    return html.Div(
        id="dashboard-main",
        className="dashboard-main",
        children=[
            # Header
            _header(),

            # ── Section 1: Controls ─────────────────────────────────
            html.Div(
                className="dash-section",
                children=[
                    dbc.Row(
                        [
                            # Combined card: NL Builder + Parameters
                            dbc.Col(
                                html.Div(
                                    className="card controls-combined",
                                    children=[
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    create_nl_input_content(),
                                                    xs=12, md=5,
                                                    className="controls-left",
                                                ),
                                                dbc.Col(
                                                    create_parameters_content(),
                                                    xs=12, md=7,
                                                    className="controls-right",
                                                ),
                                            ],
                                            className="g-0 h-100",
                                        ),
                                    ],
                                ),
                                xs=12, md=8, className="d-flex",
                            ),
                            dbc.Col(create_metrics_cards(), xs=12, md=4, className="d-flex"),
                        ],
                        className="g-2",
                    ),
                ],
            ),

            # ── Section 2: Portfolio Weights + Frontier + Bar Chart ──
            html.Div(
                className="dash-section",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    id="sliders-container",
                                    children=create_sliders_panel(),
                                ),
                                xs=12,
                                md=3,
                            ),
                            dbc.Col(
                                html.Div(
                                    className="graph-card",
                                    children=[
                                        dcc.Graph(
                                            id="frontier-chart",
                                            figure=create_frontier_figure(),
                                            config={"displayModeBar": True, "displaylogo": False},
                                            style={"height": "420px"},
                                        ),
                                    ],
                                ),
                                xs=12,
                                md=5,
                            ),
                            dbc.Col(
                                html.Div(
                                    className="graph-card",
                                    children=[
                                        dcc.Graph(
                                            id="weights-chart",
                                            figure=create_weights_figure(),
                                            config={"displayModeBar": False},
                                            style={"height": "420px"},
                                        ),
                                    ],
                                ),
                                xs=12,
                                md=4,
                            ),
                        ],
                        className="g-2",
                    ),
                ],
            ),

            # ── Section 3: Correlation + Risk + Ensemble ────────────
            html.Div(
                className="dash-section",
                children=[
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
                                            style={"height": "350px"},
                                        ),
                                    ],
                                ),
                                xs=12,
                                md=4,
                            ),
                            dbc.Col(
                                html.Div(
                                    className="graph-card",
                                    children=[
                                        dcc.Graph(
                                            id="risk-decomposition-chart",
                                            figure=create_risk_decomposition_figure(),
                                            config={"displayModeBar": False},
                                            style={"height": "350px"},
                                        ),
                                    ],
                                ),
                                xs=12,
                                md=4,
                            ),
                            dbc.Col(
                                html.Div(
                                    id="ensemble-table-container",
                                    children=create_ensemble_table(),
                                ),
                                xs=12,
                                md=4,
                            ),
                        ],
                        className="g-2",
                    ),
                ],
            ),

            # ── Section 4: Drawdown + Performance ───────────────────
            html.Div(
                className="dash-section",
                children=[
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
                                            style={"height": "320px"},
                                        ),
                                    ],
                                ),
                                xs=12,
                                md=6,
                            ),
                            dbc.Col(
                                html.Div(
                                    className="graph-card",
                                    children=[
                                        dcc.Graph(
                                            id="performance-chart",
                                            figure=create_performance_figure(),
                                            config={"displayModeBar": True, "displaylogo": False},
                                            style={"height": "320px"},
                                        ),
                                    ],
                                ),
                                xs=12,
                                md=6,
                            ),
                        ],
                        className="g-2",
                    ),
                ],
            ),

            # ── Section 5: Sentiment ────────────────────────────────
            html.Div(
                className="dash-section",
                children=[create_sentiment_panel()],
            ),

            # ── Section 6: Agent panel ──────────────────────────────
            html.Div(
                className="dash-section",
                style={"marginBottom": "24px"},
                children=[create_agent_panel()],
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


def create_layout():
    """Return the full app layout — dashboard (left 2/3) + chat (right 1/3)."""
    return html.Div(
        id="app-layout",
        className="app-layout",
        children=[
            _dashboard_content(),
            create_chat_panel(),
        ],
    )
