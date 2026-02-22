"""AI Agent analysis panel."""

from dash import html, dcc


def create_agent_panel():
    """Return the agent analysis panel with a run button and output area."""
    return html.Div(
        className="agent-panel",
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"},
                children=[
                    html.Div("ANALISIS AI", className="section-title", style={"marginBottom": "0"}),
                    html.Button(
                        "RE-EJECUTAR ANALISIS AI",
                        id="btn-run-agents",
                        className="btn-optimo",
                        n_clicks=0,
                        style={"width": "auto", "padding": "8px 20px"},
                    ),
                ],
            ),
            dcc.Loading(
                type="dot",
                color="#00d4aa",
                children=dcc.Markdown(
                    id="agent-output",
                    children="_El analisis AI se ejecuta automaticamente al generar un portafolio. Tambien puedes re-ejecutarlo manualmente._",
                    style={"color": "#c9d1d9", "fontSize": "0.8rem", "lineHeight": "1.7"},
                ),
            ),
        ],
    )
