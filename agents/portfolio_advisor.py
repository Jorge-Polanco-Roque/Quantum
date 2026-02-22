"""Asesor de Portafolios — el juez imparcial que sintetiza el debate
y el analisis cuantitativo en una recomendacion final y accionable."""


def create_portfolio_advisor(llm):
    """Factory que retorna una funcion nodo LangGraph para el asesor de portafolios."""

    def portfolio_advisor_node(state: dict) -> dict:
        quant_report = state.get("quant_report", "")
        market_context = state.get("market_context", "")
        debate_state = state.get("portfolio_debate_state", {})
        portfolio_data = state.get("portfolio_data", {})
        sentiment_data = state.get("sentiment_data", {})
        ensemble_data = state.get("ensemble_data", {})

        history = debate_state.get("history", "")
        optimistic_history = debate_state.get("optimistic_history", "")
        cautious_history = debate_state.get("cautious_history", "")

        weights = portfolio_data.get("weights", {})
        expected_return = portfolio_data.get("expected_return", 0)
        volatility = portfolio_data.get("volatility", 0)
        sharpe = portfolio_data.get("sharpe_ratio", 0)
        var_95 = portfolio_data.get("var", 0)

        weight_summary = "\n".join(
            f"  - {t}: {w:.2%}" for t, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)
        )

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

        prompt = f"""Eres un Asesor Senior de Portafolios actuando como juez imparcial en un debate de analisis de portafolio. Debes sintetizar el analisis cuantitativo, el contexto de mercado, el sentimiento, los resultados del ensemble y el debate alcista/bajista en una recomendacion clara y accionable.

=== ANALISIS CUANTITATIVO ===
{quant_report}

=== CONTEXTO DE MERCADO ===
{market_context if market_context else "Datos de mercado en vivo no disponibles."}
{sentiment_section}

=== ARGUMENTOS OPTIMISTAS ===
{optimistic_history if optimistic_history else "No se registraron argumentos optimistas."}

=== ARGUMENTOS CAUTELOSOS ===
{cautious_history if cautious_history else "No se registraron argumentos cautelosos."}

=== TRANSCRIPCION COMPLETA DEL DEBATE ===
{history if history else "No hubo debate."}
{ensemble_section}

=== DETALLES DEL PORTAFOLIO ===
Asignacion:
{weight_summary}

Metricas:
- Retorno Esperado: {expected_return:.2%}
- Volatilidad: {volatility:.2%}
- Sharpe Ratio: {sharpe:.4f}
- VaR (95%): {var_95:.2%}

Proporciona tu recomendacion final cubriendo:
1. **Veredicto**: Deberia un inversionista adoptar esta asignacion optimizada, ajustarla o usar un enfoque mas simple de pesos iguales?
2. **Fortalezas clave** del portafolio propuesto (2-3 puntos).
3. **Riesgos clave** a monitorear (2-3 puntos).
4. **Ajustes sugeridos** si los hay (ej: limitar peso maximo por accion, agregar coberturas).
5. **Perspectiva de mercado** — vision breve de condiciones a corto plazo que afectan estas posiciones.
6. **Comparacion ensemble** — si hay datos, como se compara la asignacion actual con otros metodos de optimizacion.
7. **Contexto de sentimiento** — como las noticias recientes apoyan o contradicen la tesis de inversion.

Se equilibrado, concreto y conciso (400-500 palabras). Referencia datos especificos del debate y analisis anterior.
Responde en espanol."""

        try:
            response = llm.invoke(prompt)
            recommendation = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            recommendation = (
                f"[Asesor de Portafolios] No se pudo generar la recomendacion — Error LLM: {exc}"
            )

        return {"final_recommendation": recommendation}

    return portfolio_advisor_node
