"""Analista de Mercado — el debatidor optimista que destaca potencial de
crecimiento, fundamentales fuertes y momentum positivo del mercado."""

import yfinance as yf


def fetch_market_context(tickers: list[str]) -> str:
    """Obtiene datos de mercado actuales para cada ticker usando yfinance.

    Retorna un string formateado con precio, rango 52 semanas, market cap y
    P/E trailing para cada ticker. Maneja campos faltantes gracefully.
    """
    lines: list[str] = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
            low_52 = info.get("fiftyTwoWeekLow", "N/A")
            high_52 = info.get("fiftyTwoWeekHigh", "N/A")
            mcap = info.get("marketCap")
            mcap_str = f"${mcap / 1e9:.1f}B" if mcap else "N/A"
            pe = info.get("trailingPE")
            pe_str = f"{pe:.1f}" if pe else "N/A"
            lines.append(
                f"{ticker}: Precio=${price}  |  52sem {low_52}-{high_52}  |  "
                f"MCap {mcap_str}  |  P/E {pe_str}"
            )
        except Exception:
            lines.append(f"{ticker}: datos de mercado no disponibles")
    return "\n".join(lines)


def create_market_analyst(llm):
    """Factory que retorna una funcion nodo LangGraph para el debatidor optimista."""

    def market_analyst_node(state: dict) -> dict:
        quant_report = state.get("quant_report", "")
        debate_state = state.get("portfolio_debate_state", {})
        portfolio_data = state.get("portfolio_data", {})
        market_context = state.get("market_context", "")
        sentiment_data = state.get("sentiment_data", {})

        history = debate_state.get("history", "")
        optimistic_history = debate_state.get("optimistic_history", "")
        current_response = debate_state.get("current_response", "")
        count = debate_state.get("count", 0)

        weights = portfolio_data.get("weights", {})
        expected_return = portfolio_data.get("expected_return", 0)
        sharpe = portfolio_data.get("sharpe_ratio", 0)

        weight_summary = "\n".join(
            f"  - {t}: {w:.2%}" for t, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)
        )

        if not market_context:
            tickers = portfolio_data.get("tickers", list(weights.keys()))
            market_context = fetch_market_context(tickers)

        # Sentiment context
        sentiment_section = ""
        if sentiment_data:
            resumen = sentiment_data.get("resumen", "")
            score = sentiment_data.get("score_general", 0)
            if resumen:
                sentiment_section = (
                    f"\nSentimiento de mercado (score: {score:+.3f}):\n{resumen}"
                )

        prompt = f"""Eres un Analista de Mercado Optimista participando en un debate de portafolio. Tu rol es argumentar el caso alcista para la asignacion propuesta.

Reporte cuantitativo:
{quant_report}

Asignacion actual del portafolio:
{weight_summary}

Retorno esperado: {expected_return:.2%}
Sharpe Ratio: {sharpe:.4f}

Datos de mercado en vivo:
{market_context}
{sentiment_section}

Historial del debate hasta ahora:
{history}

Ultimo argumento cauteloso:
{current_response}

Enfoca tu argumento en:
1. Potencial de crecimiento — revolucion AI, cloud computing, dominio en publicidad digital.
2. Fundamentales fuertes — crecimiento de ingresos, generacion de flujo de caja libre, ventajas competitivas.
3. Momentum — accion de precio reciente, superacion de estimados, mejoras de analistas.
4. Eficiencia del portafolio — el Sharpe Ratio demuestra fuertes retornos ajustados al riesgo.
5. Si hay noticias de sentimiento positivo, usalas como soporte para tus argumentos alcistas.
6. Contraargumenta las preocupaciones del analista cauteloso con datos especificos del contexto de mercado.

Interactua directamente con los puntos del analista cauteloso si hay alguno en el historial del debate. Se especifico y basado en evidencia.
Manten tu respuesta entre 200-300 palabras. Responde en espanol."""

        try:
            response = llm.invoke(prompt)
            argument = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            argument = f"[Analista de Mercado] Error LLM — no se pudo generar argumento optimista: {exc}"

        prefixed = f"Optimistic: {argument}"

        new_debate_state = {
            "optimistic_history": optimistic_history + "\n" + prefixed,
            "cautious_history": debate_state.get("cautious_history", ""),
            "history": history + "\n" + prefixed,
            "current_response": prefixed,
            "count": count + 1,
        }

        return {
            "portfolio_debate_state": new_debate_state,
            "market_context": market_context,
        }

    return market_analyst_node
