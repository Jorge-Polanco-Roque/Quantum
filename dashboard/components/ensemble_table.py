"""Ensemble comparison table — professional ranked HTML table with Sharpe bars."""

from dash import html

# Abbreviate long method names to prevent wrapping in tight md=4 column
_NAME_SHORT = {
    "Paridad de Riesgo": "Par. Riesgo",
    "Max. Diversificacion": "Max. Divers.",
    "Ensemble (Promedio Simple)": "Ens. Promedio",
    "Ensemble (Pond. Sharpe)": "Ens. Ponderado",
}


def create_ensemble_table(ensemble_data=None):
    """Build a professional ranked HTML table comparing all optimization methods."""
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

    # Collect rows
    items = []
    for key, data in ensemble_data.items():
        metrics = data.get("metrics", {})
        name = data.get("nombre", key)
        items.append({
            "nombre": _NAME_SHORT.get(name, name),
            "retorno": metrics.get("expected_return", 0),
            "volatilidad": metrics.get("volatility", 0),
            "sharpe": metrics.get("sharpe_ratio", 0),
            "var": metrics.get("var", 0),
        })

    # Sort by Sharpe descending (ranking)
    items.sort(key=lambda x: x["sharpe"], reverse=True)
    max_sharpe = items[0]["sharpe"] if items and items[0]["sharpe"] > 0 else 1

    # Header
    header = html.Thead(html.Tr([
        html.Th("#", className="et-col-rank"),
        html.Th("Metodo", className="et-col-name"),
        html.Th("Ret.", className="et-col-num"),
        html.Th("Vol.", className="et-col-num"),
        html.Th("Sharpe", className="et-col-sharpe"),
        html.Th("VaR", className="et-col-num"),
    ]))

    # Body rows
    body_rows = []
    for rank, item in enumerate(items, 1):
        is_best = rank == 1
        sharpe_pct = max(0, item["sharpe"] / max_sharpe * 100) if max_sharpe > 0 else 0

        # Sharpe cell with proportional background bar
        sharpe_cell = html.Td(
            className="et-col-sharpe et-sharpe-cell",
            children=[
                html.Div(className="et-sharpe-bar", style={"width": f"{sharpe_pct:.0f}%"}),
                html.Span(f"{item['sharpe']:.2f}", className="et-sharpe-value"),
            ],
        )

        rank_badge_cls = "et-rank-badge et-rank-best" if is_best else "et-rank-badge"
        row_cls = "et-row et-row-best" if is_best else "et-row"

        body_rows.append(html.Tr(
            className=row_cls,
            children=[
                html.Td(html.Span(str(rank), className=rank_badge_cls), className="et-col-rank"),
                html.Td(item["nombre"], className="et-col-name"),
                html.Td(f"{item['retorno'] * 100:.2f}%", className="et-col-num"),
                html.Td(f"{item['volatilidad'] * 100:.2f}%", className="et-col-num"),
                sharpe_cell,
                html.Td(f"{item['var'] * 100:.2f}%", className="et-col-num"),
            ],
        ))

    table = html.Table(
        className="ensemble-table",
        children=[header, html.Tbody(body_rows)],
    )

    return html.Div(
        className="card",
        children=[
            html.Div("COMPARACION ENSEMBLE", className="section-title"),
            html.Div(table, style={"overflowX": "auto"}),
        ],
    )
