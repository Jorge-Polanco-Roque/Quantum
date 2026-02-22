"""Ensemble comparison table — DataTable showing all methods side by side."""

from dash import html, dash_table


def create_ensemble_table(ensemble_data=None):
    """Build a DataTable comparing all optimization methods.

    *ensemble_data* is the dict from run_all_methods() + ensemble_vote().
    """
    if ensemble_data is None:
        return html.Div(
            className="card",
            children=[
                html.Div("COMPARACION ENSEMBLE", className="section-title"),
                html.Div(
                    "Esperando resultados...",
                    style={"color": "#8b949e", "fontSize": "0.75rem", "textAlign": "center", "padding": "40px 0"},
                ),
            ],
        )

    rows = []
    best_sharpe = -999
    best_key = ""

    # Collect all methods
    for key, data in ensemble_data.items():
        metrics = data.get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_key = key

        rows.append({
            "Metodo": data.get("nombre", key),
            "Retorno": f"{metrics.get('expected_return', 0) * 100:.2f}%",
            "Volatilidad": f"{metrics.get('volatility', 0) * 100:.2f}%",
            "Sharpe": f"{sharpe:.4f}",
            "VaR 95%": f"{metrics.get('var', 0) * 100:.2f}%",
        })

    columns = [
        {"name": "Metodo", "id": "Metodo"},
        {"name": "Retorno", "id": "Retorno"},
        {"name": "Volatilidad", "id": "Volatilidad"},
        {"name": "Sharpe", "id": "Sharpe"},
        {"name": "VaR 95%", "id": "VaR 95%"},
    ]

    # Highlight the row with best Sharpe
    style_data_conditional = [
        {
            "if": {"row_index": i},
            "backgroundColor": "rgba(0, 212, 170, 0.15)",
            "fontWeight": "700",
        }
        for i, row in enumerate(rows)
        if row["Metodo"] == ensemble_data.get(best_key, {}).get("nombre", "")
    ]

    table = dash_table.DataTable(
        data=rows,
        columns=columns,
        style_table={
            "overflowX": "auto",
            "borderRadius": "8px",
        },
        style_header={
            "backgroundColor": "#0d1117",
            "color": "#8b949e",
            "fontWeight": "600",
            "fontSize": "0.7rem",
            "textTransform": "uppercase",
            "letterSpacing": "1px",
            "border": "1px solid #30363d",
            "fontFamily": "'JetBrains Mono', monospace",
        },
        style_cell={
            "backgroundColor": "#161b22",
            "color": "#e6edf3",
            "border": "1px solid #30363d",
            "fontFamily": "'JetBrains Mono', monospace",
            "fontSize": "0.75rem",
            "textAlign": "center",
            "padding": "8px 12px",
        },
        style_data_conditional=style_data_conditional,
        style_as_list_view=False,
    )

    return html.Div(
        className="card",
        children=[
            html.Div("COMPARACION ENSEMBLE", className="section-title"),
            table,
        ],
    )
