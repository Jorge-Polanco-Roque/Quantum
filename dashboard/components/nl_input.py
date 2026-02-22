"""Natural Language portfolio input panel."""

from dash import html, dcc


def create_nl_input_panel():
    """Return the NL input panel with textarea and build button."""
    return html.Div(
        className="card nl-input-panel",
        children=[
            html.Div("CONSTRUCTOR DE PORTAFOLIOS AI", className="section-title"),
            dcc.Textarea(
                id="nl-input",
                className="nl-textarea",
                placeholder=(
                    "Describe tu portafolio ideal...\n"
                    'Ej: "Construye un portafolio tech-heavy con algo de energia"'
                ),
                style={
                    "width": "100%",
                    "height": "80px",
                    "resize": "vertical",
                },
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
