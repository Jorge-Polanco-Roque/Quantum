"""Chatbot conversacional v2 — asistente financiero con contexto enriquecido y herramientas.

Usa create_react_agent + InMemorySaver para mantener memoria multi-turn
dentro de la sesion del dashboard.

v2: Contexto enriquecido (correlacion, risk contribution, drawdown, MC stats,
sentimiento detallado, rendimiento historico) + herramientas activas (busqueda
de sectores, validacion de tickers) + acciones via marcadores (update_weights,
add_ticker, remove_ticker, rerun_pipeline).
"""

import json
import re

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from agents.graph import _create_llm
from agents.tools import CHATBOT_TOOLS
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
- Analizar la matriz de correlacion entre activos (pares mas/menos correlacionados)
- Analizar la contribucion al riesgo de cada activo (quien aporta mas riesgo)
- Interpretar datos de sentimiento POR TICKER (score, titulares de noticias)
- Analizar drawdown historico (maximo y actual) del portafolio
- Comparar rendimiento historico por activo (retorno del periodo)
- Interpretar la distribucion Monte Carlo (percentiles de retorno y Sharpe)
- Buscar tickers por sector, pais o tema usando la herramienta search_tickers_by_sector
- Validar tickers con la herramienta validate_tickers
- Obtener estadisticas de activos con fetch_and_analyze
- Sugerir ajustes o consideraciones basadas en los datos disponibles

ACCIONES:
Puedes ejecutar cambios en el portafolio del usuario. Cuando el usuario pida EXPLICITAMENTE
un cambio (modificar pesos, agregar/eliminar activos, re-ejecutar), incluye UN marcador
de accion al FINAL de tu respuesta, despues de tu analisis.

Formatos de accion (incluir EXACTAMENTE asi, en una sola linea):
- Cambiar pesos: [ACTION:update_weights:{"AAPL": 0.25, "MSFT": 0.35, "GOOGL": 0.40}]
  IMPORTANTE: incluir TODOS los tickers actuales con pesos que sumen 1.0
- Agregar ticker: [ACTION:add_ticker:{"ticker": "BTC-USD"}]
- Eliminar ticker: [ACTION:remove_ticker:{"ticker": "TSLA"}]
- Re-ejecutar pipeline: [ACTION:rerun_pipeline:{}]

REGLAS DE ACCIONES:
- Solo emite una accion si el usuario pide un cambio EXPLICITO (no por sugerencias)
- Solo UNA accion por respuesta
- Para update_weights, incluye TODOS los tickers del portafolio actual con pesos que sumen 1.0
- Primero explica lo que vas a hacer, luego coloca el marcador al final
- Si el usuario pregunta "que pasaria si..." NO emitas accion, solo analiza
- Si no estas seguro de la intencion del usuario, pregunta antes de actuar

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
    """Agente conversacional v2 con memoria, contexto enriquecido y herramientas."""

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
                tools=CHATBOT_TOOLS,
                prompt=SYSTEM_PROMPT,
                checkpointer=self._memory,
            )

    def chat(self, user_message: str, portfolio_context: dict, thread_id: str) -> dict:
        """Envia un mensaje al chatbot con contexto del portafolio.

        Parameters
        ----------
        user_message : str
            Mensaje del usuario.
        portfolio_context : dict
            Estado actual del portafolio (tickers, pesos, metricas, ensemble,
            correlacion, risk_contribution, sentiment, drawdown, etc.).
        thread_id : str
            ID unico de la sesion para mantener memoria multi-turn.

        Returns
        -------
        dict
            ``{"response": str, "action": dict|None}`` donde action contiene
            ``{"type": str, "payload": dict}`` si el LLM emitio un marcador.
        """
        if self._no_key:
            return {
                "response": (
                    "**Error:** No hay API key configurada. "
                    "Configura `OPENAI_API_KEY` o `ANTHROPIC_API_KEY` en tu archivo `.env`."
                ),
                "action": None,
            }

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

            # Extract action marker and clean response
            action = self._extract_action(final_msg)
            if action:
                # Remove the action marker from visible response
                clean_response = re.sub(
                    r'\[ACTION:\w+:\{[^]]*?\}\]', '', final_msg
                ).strip()
            else:
                clean_response = final_msg

            return {"response": clean_response, "action": action}
        except Exception as exc:
            return {
                "response": f"**Error del asistente:** `{exc}`",
                "action": None,
            }

    @staticmethod
    def _extract_action(text: str) -> dict | None:
        """Parse an action marker from the LLM response.

        Looks for ``[ACTION:type:{json}]`` and returns
        ``{"type": str, "payload": dict}`` or ``None``.
        """
        pattern = r'\[ACTION:(\w+):(\{[^]]*?\})\]'
        match = re.search(pattern, text)
        if not match:
            return None

        action_type = match.group(1)
        valid_types = {"update_weights", "add_ticker", "remove_ticker", "rerun_pipeline"}
        if action_type not in valid_types:
            return None

        try:
            payload = json.loads(match.group(2))
        except json.JSONDecodeError:
            return None

        return {"type": action_type, "payload": payload}

    @staticmethod
    def _format_context(ctx: dict) -> str:
        """Formatea el contexto del portafolio como bloque de texto."""
        if not ctx:
            return "[CONTEXTO DEL PORTAFOLIO]\nNo hay portafolio generado aun."

        lines = ["[CONTEXTO DEL PORTAFOLIO]"]

        # Parameters
        params = ctx.get("parameters", {})
        if params:
            rf = params.get("risk_free_rate")
            vc = params.get("var_confidence")
            ns = params.get("num_simulations")
            lines.append("Parametros de simulacion:")
            if rf is not None:
                lines.append(f"  Tasa libre de riesgo: {rf*100:.1f}%")
            if vc is not None:
                lines.append(f"  Nivel de confianza VaR: {vc*100:.1f}%")
            if ns is not None:
                lines.append(f"  Simulaciones Monte Carlo: {ns:,}")

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

        # Ensemble — full detail: weights per ticker + metrics per method
        ensemble = ctx.get("ensemble", {})
        if ensemble and tickers:
            lines.append("Resultados del ensemble (pesos % por ticker y metricas por metodo):")
            for key, data in ensemble.items():
                name = data.get("nombre", key)
                m = data.get("metrics", {})
                w = data.get("weights", [])
                sr = m.get("sharpe_ratio", "N/A")
                ret = m.get("expected_return")
                vol = m.get("volatility")
                met_parts = [f"Sharpe={sr}"]
                if ret is not None:
                    met_parts.append(f"Ret={ret*100:.2f}%")
                if vol is not None:
                    met_parts.append(f"Vol={vol*100:.2f}%")
                lines.append(f"  {name} ({', '.join(met_parts)}):")
                if w and len(w) == len(tickers):
                    for t, wi in zip(tickers, w):
                        lines.append(f"    {t}: {wi*100:.1f}%")
        elif ensemble:
            lines.append("Metodos del ensemble:")
            for key, data in ensemble.items():
                name = data.get("nombre", key)
                m = data.get("metrics", {})
                sr = m.get("sharpe_ratio", "N/A")
                lines.append(f"  {name}: Sharpe={sr}")

        # ── Correlation matrix (only high pairs |r| > 0.5) ──
        corr_data = ctx.get("correlation_matrix", {})
        corr_tickers = corr_data.get("tickers", [])
        corr_matrix = corr_data.get("matrix", [])
        if corr_tickers and corr_matrix:
            lines.append("Correlaciones relevantes (|r| > 0.5):")
            for i, t1 in enumerate(corr_tickers):
                for j, t2 in enumerate(corr_tickers):
                    if j > i and len(corr_matrix) > i and len(corr_matrix[i]) > j:
                        r = corr_matrix[i][j]
                        if abs(r) > 0.5:
                            lines.append(f"  {t1}-{t2}: {r:.3f}")

        # ── Risk contribution ──
        risk_contrib = ctx.get("risk_contribution", {})
        if risk_contrib:
            sorted_rc = sorted(risk_contrib.items(), key=lambda x: x[1], reverse=True)
            lines.append("Contribucion al riesgo (mayor a menor):")
            for t, rc in sorted_rc:
                lines.append(f"  {t}: {rc*100:.1f}%")

        # ── Sentiment detailed ──
        sentiment = ctx.get("sentiment", {})
        if sentiment:
            sg = sentiment.get("score_general")
            if sg is not None:
                lines.append(f"Sentimiento general del mercado: {sg:+.3f}")
            por_ticker = sentiment.get("por_ticker", {})
            if por_ticker:
                lines.append("Sentimiento por ticker:")
                for t, tdata in por_ticker.items():
                    score = tdata.get("score_promedio", 0)
                    noticias = tdata.get("noticias", [])
                    lines.append(f"  {t}: score={score:+.2f}")
                    for n in noticias[:3]:
                        title = n.get("title", "")[:60]
                        ns = n.get("score", 0)
                        lines.append(f"    [{ns:+.1f}] {title}")

        # ── Drawdown ──
        drawdown = ctx.get("drawdown", {})
        if drawdown:
            max_dd = drawdown.get("max_drawdown")
            curr_dd = drawdown.get("current_drawdown")
            if max_dd is not None:
                lines.append(f"Max Drawdown historico: {max_dd*100:.2f}%")
            if curr_dd is not None:
                lines.append(f"Drawdown actual: {curr_dd*100:.2f}%")

        # ── Historical performance ──
        hist_perf = ctx.get("historical_performance", {})
        if hist_perf:
            port_ret = hist_perf.get("portfolio_period_return")
            if port_ret is not None:
                lines.append(f"Retorno del periodo (portafolio): {port_ret*100:.2f}%")
            per_ticker = hist_perf.get("per_ticker", {})
            if per_ticker:
                sorted_pt = sorted(per_ticker.items(), key=lambda x: x[1], reverse=True)
                lines.append("Retorno historico por activo:")
                for t, r in sorted_pt:
                    lines.append(f"  {t}: {r*100:.2f}%")

        # ── Monte Carlo distribution ──
        mc_dist = ctx.get("mc_distribution", {})
        if mc_dist:
            ret_pcts = mc_dist.get("return_percentiles", {})
            sharpe_pcts = mc_dist.get("sharpe_percentiles", {})
            if ret_pcts:
                lines.append("Distribucion Monte Carlo (retornos):")
                for k, v in ret_pcts.items():
                    lines.append(f"  {k}: {v*100:.2f}%")
            if sharpe_pcts:
                lines.append("Distribucion Monte Carlo (Sharpe):")
                for k, v in sharpe_pcts.items():
                    lines.append(f"  {k}: {v:.3f}")

        # Legacy sentiment score (backward compat)
        sentiment_score = ctx.get("sentiment_score")
        if sentiment_score is not None and not sentiment:
            lines.append(f"Sentimiento general: {sentiment_score:+.3f}")

        lines.append("[FIN CONTEXTO]")
        return "\n".join(lines)
