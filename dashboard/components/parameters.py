"""Parameters panel — risk-free rate, simulation count, VaR level inputs."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_parameters_content():
    """Return parameter inputs + buttons (no card wrapper — for combined card)."""
    return html.Div(
        className="params-content",
        children=[
            html.Div(
                className="section-title",
                children="PARAMETROS",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("TASA LIBRE (%)", className="param-label"),
                            dcc.Input(
                                id="input-rf",
                                type="number",
                                value=4,
                                min=0,
                                max=20,
                                step=0.1,
                                debounce=True,
                                className="param-input",
                                style={"width": "100%"},
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("SIMULACIONES", className="param-label"),
                            dcc.Input(
                                id="input-sims",
                                type="number",
                                value=10000,
                                min=1000,
                                max=100000,
                                step=1000,
                                debounce=True,
                                className="param-input",
                                style={"width": "100%"},
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("VAR NIVEL (%)", className="param-label"),
                            dcc.Input(
                                id="input-var-level",
                                type="number",
                                value=95,
                                min=90,
                                max=99.9,
                                step=0.5,
                                debounce=True,
                                className="param-input",
                                style={"width": "100%"},
                            ),
                        ],
                        width=4,
                    ),
                ],
                className="g-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Button(
                            "EJECUTAR",
                            id="btn-ejecutar",
                            className="btn-ejecutar",
                            n_clicks=0,
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        html.Button(
                            "OPTIMO",
                            id="btn-optimo",
                            className="btn-optimo",
                            n_clicks=0,
                        ),
                        width=6,
                    ),
                ],
                className="g-2",
            ),
        ],
    )


def create_parameters_panel():
    """Return the parameter inputs + action buttons panel (standalone card)."""
    panel = create_parameters_content()
    panel.className = "card params-content"
    return panel
