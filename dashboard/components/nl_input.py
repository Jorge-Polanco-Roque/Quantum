"""Natural Language portfolio input panel."""

from dash import html, dcc


def create_nl_input_content():
    """Return the NL input content (no card wrapper — used inside combined card)."""
    return html.Div(
        className="nl-input-panel",
        children=[
            html.Div("CONSTRUCTOR DE PORTAFOLIOS AI", className="section-title"),
            dcc.Textarea(
                id="nl-input",
                className="nl-textarea",
                placeholder=(
                    "Describe tu portafolio ideal...\n"
                    'Ej: "Construye un portafolio tech-heavy con algo de energia"'
                ),
                style={"width": "100%"},
            ),
            dcc.Loading(
                type="dot",
                color="#00d4aa",
                children=html.Div(id="nl-build-output"),
            ),
            html.Button(
                "CONSTRUIR PORTAFOLIO",
                id="btn-build-portfolio",
                className="btn-build",
                n_clicks=0,
            ),
        ],
    )


def create_nl_input_panel():
    """Return the NL input panel (standalone card — backwards compatible)."""
    panel = create_nl_input_content()
    panel.className = "card nl-input-panel"
    return panel
