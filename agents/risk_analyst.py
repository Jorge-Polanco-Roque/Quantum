"""Analista de Riesgo — el debatidor cauteloso que destaca riesgos a la baja,
preocupaciones de concentracion y escenarios de riesgo de cola."""


def create_risk_analyst(llm):
    """Factory que retorna una funcion nodo LangGraph para el debatidor cauteloso."""

    def risk_analyst_node(state: dict) -> dict:
        quant_report = state.get("quant_report", "")
        debate_state = state.get("portfolio_debate_state", {})
        portfolio_data = state.get("portfolio_data", {})
        sentiment_data = state.get("sentiment_data", {})

        history = debate_state.get("history", "")
        cautious_history = debate_state.get("cautious_history", "")
        current_response = debate_state.get("current_response", "")
        count = debate_state.get("count", 0)

        weights = portfolio_data.get("weights", {})
        var_95 = portfolio_data.get("var", 0)
        volatility = portfolio_data.get("volatility", 0)

        weight_summary = "\n".join(
            f"  - {t}: {w:.2%}" for t, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)
        )

        # Sentiment context
        sentiment_section = ""
        if sentiment_data:
            resumen = sentiment_data.get("resumen", "")
            score = sentiment_data.get("score_general", 0)
            if resumen:
                sentiment_section = (
                    f"\nSentimiento de mercado (score: {score:+.3f}):\n{resumen}"
                )

        prompt = f"""Eres un Analista de Riesgo Cauteloso participando en un debate de portafolio. Tu rol es argumentar el lado conservador, destacando riesgos y vulnerabilidades en la asignacion propuesta.

Reporte cuantitativo:
{quant_report}

Asignacion actual del portafolio:
{weight_summary}

Volatilidad del portafolio: {volatility:.2%}
Value at Risk (95%): {var_95:.2%}
{sentiment_section}

Historial del debate hasta ahora:
{history}

Ultimo argumento del lado contrario:
{current_response}

Enfoca tu argumento en:
1. Preocupaciones del VaR — que podria significar un evento de cola del 95% en terminos de dolares para un portafolio real?
2. Riesgo de concentracion — correlacion sectorial y geografica extrema.
3. Riesgo de cola y quiebre de correlaciones — durante estres de mercado (COVID 2020, subidas de tasas 2022), las acciones tech se movieron en bloque.
4. Vientos macro en contra — entorno de tasas de interes, riesgo regulatorio, ciclo de hype de AI.
5. Si hay datos de sentimiento negativos, usalos para reforzar tus argumentos de riesgo.
6. Contraargumenta los puntos optimistas con preocupaciones especificas basadas en datos.

Interactua directamente con los puntos del analista optimista si hay alguno en el historial del debate. Se especifico y orientado a datos.
Manten tu respuesta entre 200-300 palabras. Responde en espanol."""

        try:
            response = llm.invoke(prompt)
            argument = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            argument = f"[Analista de Riesgo] Error LLM — no se pudo generar argumento cauteloso: {exc}"

        prefixed = f"Cautious: {argument}"

        new_debate_state = {
            "optimistic_history": debate_state.get("optimistic_history", ""),
            "cautious_history": cautious_history + "\n" + prefixed,
            "history": history + "\n" + prefixed,
            "current_response": prefixed,
            "count": count + 1,
        }

        return {"portfolio_debate_state": new_debate_state}

    return risk_analyst_node
