"""Analista Fundamental — combina metricas cuantitativas con sentimiento de noticias.

Hace una sola llamada al LLM para ponderar analisis cuantitativo con fundamental.
"""

import config as cfg
from agents.graph import _create_llm

FUNDAMENTAL_PROMPT = """\
Eres un Analista Fundamental Senior. Tu trabajo es combinar el analisis \
cuantitativo del portafolio con las noticias recientes del mercado para dar \
una ponderacion integral.

=== DATOS DEL PORTAFOLIO ===
Tickers: {tickers}
Pesos: {weights}
Retorno Esperado: {expected_return:.2%}
Volatilidad: {volatility:.2%}
Sharpe Ratio: {sharpe_ratio:.3f}
VaR (95%): {var:.2%}

=== NOTICIAS Y SENTIMIENTO ===
Sentimiento General: {score_general:+.3f}

{noticias_detalle}

{ensemble_section}

Analiza:
1. Como las noticias recientes afectan las perspectivas de cada activo del portafolio
2. Si el sentimiento del mercado respalda o contradice la asignacion cuantitativa
3. Riesgos fundamentales no capturados por los modelos cuantitativos
4. Recomendacion final: mantener la asignacion actual, ajustar pesos, o revisar \
la seleccion de activos

Se conciso y directo. Responde en español. Usa formato Markdown con encabezados.
"""


def run_fundamental_analysis(
    portfolio_data: dict,
    sentiment_data: dict,
    ensemble_data: dict | None = None,
) -> str:
    """Run a single LLM call combining quantitative metrics with news sentiment.

    Parameters
    ----------
    portfolio_data : dict
        Must contain: tickers, weights (dict), expected_return, volatility,
        sharpe_ratio, var.
    sentiment_data : dict
        From fetch_all_news(): por_ticker, score_general, resumen.
    ensemble_data : dict, optional
        Ensemble method results for additional context.

    Returns
    -------
    str
        Markdown-formatted fundamental analysis.
    """
    # Resolve API key
    provider = cfg.LLM_PROVIDER.lower()
    if provider == "anthropic":
        api_key = cfg.ANTHROPIC_API_KEY
    else:
        api_key = cfg.OPENAI_API_KEY

    if not api_key:
        return (
            "_No hay API key configurada. Configura OPENAI_API_KEY o "
            "ANTHROPIC_API_KEY en tu archivo .env para el analisis fundamental._"
        )

    # Build prompt components
    tickers = portfolio_data.get("tickers", list(portfolio_data.get("weights", {}).keys()))
    weights = portfolio_data.get("weights", {})
    if isinstance(weights, dict):
        weights_str = ", ".join(f"{t}: {w:.1%}" for t, w in weights.items())
    else:
        weights_str = str(weights)

    # News detail
    noticias_lines = []
    por_ticker = sentiment_data.get("por_ticker", {})
    for ticker in tickers:
        info = por_ticker.get(ticker, {})
        score = info.get("score_promedio", 0)
        noticias_lines.append(f"**{ticker}** (sentimiento: {score:+.2f}):")
        for n in info.get("noticias", []):
            s = n.get("score", 0)
            noticias_lines.append(f"  [{s:+.1f}] {n.get('title', 'Sin titulo')}")

    noticias_detalle = "\n".join(noticias_lines) if noticias_lines else "No hay noticias disponibles."

    # Ensemble section
    ensemble_section = ""
    if ensemble_data:
        lines = ["=== METODOS DE OPTIMIZACION (ENSEMBLE) ==="]
        for key, data in ensemble_data.items():
            nombre = data.get("nombre", key)
            m = data.get("metrics", {})
            sharpe = m.get("sharpe_ratio", 0)
            ret = m.get("expected_return", 0)
            lines.append(f"- {nombre}: Sharpe {sharpe:.3f}, Retorno {ret:.2%}")
        ensemble_section = "\n".join(lines)

    prompt = FUNDAMENTAL_PROMPT.format(
        tickers=", ".join(tickers),
        weights=weights_str,
        expected_return=portfolio_data.get("expected_return", 0),
        volatility=portfolio_data.get("volatility", 0),
        sharpe_ratio=portfolio_data.get("sharpe_ratio", 0),
        var=portfolio_data.get("var", 0),
        score_general=sentiment_data.get("score_general", 0),
        noticias_detalle=noticias_detalle,
        ensemble_section=ensemble_section,
    )

    try:
        llm = _create_llm(cfg.LLM_PROVIDER, cfg.LLM_MODEL, api_key)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        return f"_Error en analisis fundamental: {exc}_"
