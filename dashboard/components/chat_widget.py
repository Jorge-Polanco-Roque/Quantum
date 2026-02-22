"""Chat panel — full-height sidebar (position toggleable: right/left/top/bottom)."""

from dash import html, dcc


def create_chat_panel():
    """Return the chat sidebar component (always visible, full height)."""
    return html.Div(
        id="chat-sidebar",
        className="chat-sidebar",
        children=[
            # Header
            html.Div(
                className="chat-header",
                children=[
                    html.Span("ASISTENTE AI", className="chat-header-title"),
                    # Position toggle buttons
                    html.Div(
                        className="chat-position-toggles",
                        children=[
                            html.Button(
                                "\u25C0", id="chat-pos-left", n_clicks=0,
                                className="chat-pos-btn",
                                title="Chat a la izquierda",
                            ),
                            html.Button(
                                "\u25B2", id="chat-pos-top", n_clicks=0,
                                className="chat-pos-btn",
                                title="Chat arriba",
                            ),
                            html.Button(
                                "\u25BC", id="chat-pos-bottom", n_clicks=0,
                                className="chat-pos-btn",
                                title="Chat abajo",
                            ),
                            html.Button(
                                "\u25B6", id="chat-pos-right", n_clicks=0,
                                className="chat-pos-btn chat-pos-btn-active",
                                title="Chat a la derecha",
                            ),
                        ],
                    ),
                    html.Div(
                        className="chat-header-badges",
                        children=[
                            html.Span("CONTEXTO", className="chat-badge"),
                            html.Span("MULTI-TURN", className="chat-badge"),
                        ],
                    ),
                ],
            ),
            # Messages area
            html.Div(
                id="chat-messages",
                className="chat-messages",
                children=[
                    html.Div(
                        className="chat-bubble chat-bubble-assistant",
                        children=dcc.Markdown(
                            "Hola! Soy tu asistente financiero. "
                            "Ejecuta una simulacion y preguntame lo que quieras "
                            "sobre tu portafolio.\n\n"
                            "Puedo ayudarte con:\n"
                            "- Explicar metricas (Sharpe, VaR, volatilidad)\n"
                            "- Analizar la composicion del portafolio\n"
                            "- Comparar metodos del ensemble\n"
                            "- Conceptos financieros (frontera eficiente, HRP, CML)",
                            style={"margin": "0", "fontSize": "inherit", "lineHeight": "inherit"},
                        ),
                    ),
                ],
            ),
            # Loading spinner
            dcc.Loading(
                id="chat-loading",
                type="dot",
                color="#00d4aa",
                children=html.Div(id="chat-loading-target", style={"display": "none"}),
                style={"minHeight": "0"},
            ),
            # Input area
            html.Div(
                className="chat-input-area",
                children=[
                    dcc.Input(
                        id="chat-input",
                        type="text",
                        placeholder="Escribe tu pregunta...",
                        debounce=False,
                        n_submit=0,
                        className="chat-input-field",
                    ),
                    html.Button(
                        "Enviar",
                        id="chat-send",
                        n_clicks=0,
                        className="chat-send-btn",
                    ),
                ],
            ),
            # Stores
            dcc.Store(id="store-chat-history", data=[]),
            dcc.Store(id="store-chat-thread-id", data=None),
            dcc.Store(id="store-chat-position", data="right"),
        ],
    )
