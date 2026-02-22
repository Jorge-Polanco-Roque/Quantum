"""Analista Cuantitativo — genera analisis detallado del portafolio
optimizado por el motor Monte Carlo, incluyendo resultados del ensemble
y datos de sentimiento."""


def create_quant_analyst(llm):
    """Factory que retorna una funcion nodo LangGraph para el analista cuantitativo."""

    def quant_analyst_node(state: dict) -> dict:
        portfolio_data = state.get("portfolio_data", {})
        ensemble_data = state.get("ensemble_data", {})
        sentiment_data = state.get("sentiment_data", {})

        weights = portfolio_data.get("weights", {})
        expected_return = portfolio_data.get("expected_return", 0)
        volatility = portfolio_data.get("volatility", 0)
        sharpe_ratio = portfolio_data.get("sharpe_ratio", 0)
        var_95 = portfolio_data.get("var", 0)
        tickers = portfolio_data.get("tickers", list(weights.keys()))

        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        weight_summary = "\n".join(
            f"  - {t}: {w:.2%}" for t, w in sorted_weights
        )

        top_holdings = [t for t, w in sorted_weights if w >= 0.15]
        concentration = sum(w for _, w in sorted_weights[:3])

        # Ensemble summary
        ensemble_section = ""
        if ensemble_data:
            lines = []
            for key, data in ensemble_data.items():
                nombre = data.get("nombre", key)
                m = data.get("metrics", {})
                lines.append(
                    f"  - {nombre}: Sharpe={m.get('sharpe_ratio', 0):.4f}, "
                    f"Ret={m.get('expected_return', 0):.2%}, "
                    f"Vol={m.get('volatility', 0):.2%}"
                )
            ensemble_section = "\n=== RESULTADOS DEL ENSEMBLE ===\n" + "\n".join(lines)

        # Sentiment summary
        sentiment_section = ""
        if sentiment_data:
            resumen = sentiment_data.get("resumen", "")
            score = sentiment_data.get("score_general", 0)
            if resumen:
                sentiment_section = (
                    f"\n=== SENTIMIENTO DE MERCADO ===\n"
                    f"Score general: {score:+.3f}\n{resumen}"
                )

        prompt = f"""Eres un Analista Cuantitativo Senior revisando un portafolio optimizado.

Asignacion del portafolio (optimizada via simulacion Monte Carlo + SLSQP):
{weight_summary}

Metricas clave:
- Retorno Anual Esperado: {expected_return:.2%}
- Volatilidad Anualizada: {volatility:.2%}
- Sharpe Ratio: {sharpe_ratio:.4f}
- Value at Risk (95%): {var_95:.2%}

Tickers en el universo: {', '.join(tickers)}

Concentracion Top-3: {concentration:.2%}
{f'Posiciones altamente concentradas (>=15%): {", ".join(top_holdings)}' if top_holdings else 'Ninguna posicion supera el 15%.'}
{ensemble_section}
{sentiment_section}

Proporciona un analisis cuantitativo detallado cubriendo:
1. Eficiencia riesgo-retorno — es el Sharpe Ratio atractivo respecto a un enfoque de pesos iguales?
2. Analisis de concentracion — el portafolio esta excesivamente concentrado o bien diversificado?
3. Evaluacion de volatilidad — es la vol anualizada razonable para este universo de acciones?
4. Interpretacion del VaR — que implica el VaR 95% en terminos de perdidas potenciales diarias/anuales?
5. Anomalias de asignacion (pesos cercanos a cero, sobreasignaciones extremas).
6. Si hay datos ensemble, compara el portafolio actual con los otros metodos de optimizacion.
7. Si hay datos de sentimiento, menciona como el contexto noticioso podria afectar las perspectivas.

Manten el analisis conciso (300-400 palabras), basado en datos y accionable.
Responde en espanol."""

        try:
            response = llm.invoke(prompt)
            report = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            report = (
                f"[Analista Cuantitativo] No se pudo generar el analisis — Error LLM: {exc}"
            )

        return {"quant_report": report}

    return quant_analyst_node
