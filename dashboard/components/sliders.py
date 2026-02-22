"""Portfolio weight sliders with lock toggles — dynamic pattern-matching IDs."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from config import TICKERS, get_ticker_color, get_ticker_name


def _slider_row(ticker, equal_weight):
    """Build a single slider row for a ticker using pattern-matching IDs."""
    color = get_ticker_color(ticker)
    name = get_ticker_name(ticker)
    return html.Div(
        className="slider-row",
        children=[
            html.Span(
                ticker,
                className="slider-ticker-label",
                style={"color": color},
                title=name,
            ),
            dcc.Checklist(
                id={"type": "lock", "index": ticker},
                options=[{"label": "", "value": "locked"}],
                value=[],
                className="lock-checkbox",
                style={"minWidth": "24px"},
                inputStyle={"marginRight": "0"},
            ),
            html.Div(
                dcc.Slider(
                    id={"type": "slider", "index": ticker},
                    min=0,
                    max=100,
                    step=0.1,
                    value=equal_weight,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                className="slider-container",
            ),
            html.Span(
                f"{equal_weight:.1f}%",
                id={"type": "weight-display", "index": ticker},
                className="slider-weight-display",
            ),
        ],
    )


def create_sliders_panel(tickers=None):
    """Return the full sliders panel with lock toggles and action buttons.

    If *tickers* is None, defaults to config.TICKERS.
    """
    if tickers is None:
        tickers = TICKERS

    equal_weight = round(100 / len(tickers), 2) if tickers else 0
    slider_rows = [_slider_row(t, equal_weight) for t in tickers]

    total_bar = html.Div(
        className="total-bar-container",
        children=[
            html.Div(
                className="total-bar-track",
                children=[
                    html.Div(
                        id="total-bar-fill",
                        className="total-bar-fill",
                        style={"width": "100%"},
                    ),
                ],
            ),
            html.Div(
                id="total-bar-label",
                className="total-bar-label",
                children="Total: 100.0%",
            ),
        ],
    )

    action_buttons = dbc.Row(
        [
            dbc.Col(
                html.Button("IGUAL", id="btn-equal", className="btn-equal", n_clicks=0),
                width=4,
            ),
            dbc.Col(
                html.Button("RANDOM", id="btn-random", className="btn-random", n_clicks=0),
                width=4,
            ),
            dbc.Col(
                html.Button("NORM.", id="btn-normalize", className="btn-normalize", n_clicks=0),
                width=4,
            ),
        ],
        className="g-2",
    )

    return html.Div(
        className="card",
        children=[
            html.Div("PESOS DEL PORTAFOLIO", className="section-title"),
            *slider_rows,
            total_bar,
            action_buttons,
        ],
    )
