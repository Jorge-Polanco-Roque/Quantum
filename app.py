"""Quant Portfolio Optimizer — Dash application entry point."""

import dash
from dash import Dash
import dash_bootstrap_components as dbc

from dashboard.layout import create_layout
from dashboard.callbacks import register_callbacks

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    assets_folder="dashboard/assets",
)
app.title = "Quant Portfolio Optimizer"
app.layout = create_layout()
register_callbacks(app)

if __name__ == "__main__":
    from config import DASH_DEBUG, DASH_PORT
    app.run(debug=DASH_DEBUG, port=DASH_PORT)
