"""Chatbot conversacional — asistente financiero con contexto del portafolio.

Usa create_react_agent + InMemorySaver para mantener memoria multi-turn
dentro de la sesion del dashboard.
"""

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from agents.graph import _create_llm
import config as cfg


SYSTEM_PROMPT = """\
Eres un asistente financiero experto integrado en el Quant Portfolio Optimizer.
Tu rol es ayudar al usuario a entender y analizar su portafolio de inversiones.

CAPACIDADES:
- Explicar metricas del portafolio (Sharpe Ratio, retorno esperado, volatilidad, VaR)
- Explicar conceptos financieros (frontera eficiente, CML, HRP, CVaR, paridad de riesgo, \
diversificacion, correlacion, drawdown) en el contexto del portafolio actual
- Analizar la composicion del portafolio (por que ciertos activos tienen mas peso)
- Comparar metodos del ensemble (Max Sharpe, Min Varianza, Risk Parity, HRP, etc.)
- Interpretar datos de sentimiento de noticias
- Sugerir ajustes o consideraciones basadas en los datos disponibles

REGLAS:
- SIEMPRE responde en espanol
- Usa los datos del contexto del portafolio para dar respuestas especificas, no genericas
- Si no hay datos de portafolio disponibles, indica que primero debe ejecutar una simulacion
- Se conciso pero informativo. Usa markdown para formatear (negritas, listas, tablas)
- No inventes datos. Si no tienes la informacion, dilo claramente
- Cuando expliques conceptos, relacionalos con los valores actuales del portafolio del usuario
- Si el usuario pregunta algo fuera del ambito financiero/portafolio, redirige amablemente
"""


class ChatbotAgent:
    """Agente conversacional con memoria para el dashboard de portafolios."""

    def __init__(
        self,
        llm_provider: str = cfg.LLM_PROVIDER,
        model: str = cfg.LLM_MODEL,
        api_key: str = "",
    ):
        if not api_key:
            if llm_provider.lower() == "anthropic":
                api_key = cfg.ANTHROPIC_API_KEY
            else:
                api_key = cfg.OPENAI_API_KEY

        if not api_key:
            self._no_key = True
            self._agent = None
            self._memory = None
        else:
            self._no_key = False
            self._memory = MemorySaver()
            llm = _create_llm(llm_provider, model, api_key)
            self._agent = create_react_agent(
                llm,
                tools=[],
                prompt=SYSTEM_PROMPT,
                checkpointer=self._memory,
            )

    def chat(self, user_message: str, portfolio_context: dict, thread_id: str) -> str:
        """Envia un mensaje al chatbot con contexto del portafolio.

        Parameters
        ----------
        user_message : str
            Mensaje del usuario.
        portfolio_context : dict
            Estado actual del portafolio (tickers, pesos, metricas, ensemble, etc.).
        thread_id : str
            ID unico de la sesion para mantener memoria multi-turn.

        Returns
        -------
        str
            Respuesta del asistente en markdown.
        """
        if self._no_key:
            return (
                "**Error:** No hay API key configurada. "
                "Configura `OPENAI_API_KEY` o `ANTHROPIC_API_KEY` en tu archivo `.env`."
            )

        # Build context block
        context_block = self._format_context(portfolio_context)
        full_message = f"{context_block}\n\n{user_message}"

        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": full_message}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            final_msg = result["messages"][-1].content
            if isinstance(final_msg, list):
                final_msg = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in final_msg
                )
            return final_msg
        except Exception as exc:
            return f"**Error del asistente:** `{exc}`"

    @staticmethod
    def _format_context(ctx: dict) -> str:
        """Formatea el contexto del portafolio como bloque de texto."""
        if not ctx:
            return "[CONTEXTO DEL PORTAFOLIO]\nNo hay portafolio generado aun."

        lines = ["[CONTEXTO DEL PORTAFOLIO]"]

        # Tickers and weights
        tickers = ctx.get("tickers", [])
        weights = ctx.get("weights", [])
        if tickers and weights:
            pairs = [f"  {t}: {w*100:.1f}%" for t, w in zip(tickers, weights)]
            lines.append("Activos y pesos:")
            lines.extend(pairs)

        # Metrics
        metrics = ctx.get("metrics", {})
        if metrics:
            lines.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
            er = metrics.get("expected_return")
            if er is not None:
                lines.append(f"Retorno Esperado: {er*100:.2f}%")
            vol = metrics.get("volatility")
            if vol is not None:
                lines.append(f"Volatilidad: {vol*100:.2f}%")
            var = metrics.get("var")
            if var is not None:
                lines.append(f"VaR diario: {var*100:.2f}%")

        # Ensemble summary
        ensemble = ctx.get("ensemble", {})
        if ensemble:
            lines.append("Metodos del ensemble:")
            for key, data in ensemble.items():
                name = data.get("nombre", key)
                m = data.get("metrics", {})
                sr = m.get("sharpe_ratio", "N/A")
                lines.append(f"  {name}: Sharpe={sr}")

        # Sentiment summary
        sentiment = ctx.get("sentiment_score")
        if sentiment is not None:
            lines.append(f"Sentimiento general: {sentiment:+.3f}")

        lines.append("[FIN CONTEXTO]")
        return "\n".join(lines)
