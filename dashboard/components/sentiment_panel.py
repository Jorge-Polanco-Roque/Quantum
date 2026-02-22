"""Sentiment / news panel — button + output area."""

from dash import html, dcc


def create_sentiment_panel():
    """Return the sentiment panel with refresh button and output area."""
    return html.Div(
        className="card",
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "12px",
                },
                children=[
                    html.Div("SENTIMIENTO DE MERCADO", className="section-title", style={"marginBottom": "0"}),
                    html.Button(
                        "ACTUALIZAR NOTICIAS",
                        id="btn-refresh-news",
                        className="btn-optimo",
                        n_clicks=0,
                        style={"width": "auto", "padding": "8px 20px"},
                    ),
                ],
            ),
            dcc.Loading(
                type="dot",
                color="#00d4aa",
                children=html.Div(
                    id="sentiment-output",
                    children=html.Div(
                        "Haz clic en 'ACTUALIZAR NOTICIAS' para ver el sentimiento del mercado.",
                        style={"color": "#8b949e", "fontSize": "0.75rem"},
                    ),
                ),
            ),
        ],
    )
