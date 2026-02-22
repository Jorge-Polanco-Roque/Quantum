"""Grafo de analisis de portafolios LangGraph — conecta analista cuantitativo,
analista de riesgo, analista de mercado y asesor de portafolios en un flujo
de trabajo impulsado por debate."""

from langgraph.graph import END, StateGraph, START

from agents.states import PortfolioAgentState
from agents.quant_analyst import create_quant_analyst
from agents.risk_analyst import create_risk_analyst
from agents.market_analyst import create_market_analyst, fetch_market_context
from agents.portfolio_advisor import create_portfolio_advisor

import config as cfg


def _create_llm(provider: str, model: str, api_key: str, temperature: float = 0.0):
    """Instancia un modelo de chat LangChain para el proveedor dado.

    ``temperature=0.0`` por defecto para obtener resultados reproducibles
    (greedy decoding).  Los llamadores pueden sobreescribirlo si necesitan
    mas creatividad.
    """
    provider = provider.lower()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, anthropic_api_key=api_key, temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model, openai_api_key=api_key, temperature=temperature,
        )


class PortfolioAgentsGraph:
    """Construye y compila el grafo multi-agente de analisis de portafolios."""

    MAX_DEBATE_ROUNDS = cfg.MAX_DEBATE_ROUNDS  # from config.py

    def __init__(
        self,
        llm_provider: str = cfg.LLM_PROVIDER,
        model: str = cfg.LLM_MODEL,
        api_key: str = "",
    ):
        # Resolve API key: explicit arg > env-based config
        if not api_key:
            if llm_provider.lower() == "anthropic":
                api_key = cfg.ANTHROPIC_API_KEY
            else:
                api_key = cfg.OPENAI_API_KEY

        if not api_key:
            self._no_key = True
            self._llm = None
        else:
            self._no_key = False
            self._llm = _create_llm(llm_provider, model, api_key)

        self._graph = self._build_graph() if not self._no_key else None

    # ── Graph construction ───────────────────────────────────────────

    def _build_graph(self):
        quant_node = create_quant_analyst(self._llm)
        risk_node = create_risk_analyst(self._llm)
        market_node = create_market_analyst(self._llm)
        advisor_node = create_portfolio_advisor(self._llm)

        graph = StateGraph(PortfolioAgentState)

        graph.add_node("quant_analyst", quant_node)
        graph.add_node("risk_analyst", risk_node)
        graph.add_node("market_analyst", market_node)
        graph.add_node("portfolio_advisor", advisor_node)

        # START -> quant_analyst -> risk_analyst -> conditional debate loop
        graph.add_edge(START, "quant_analyst")
        graph.add_edge("quant_analyst", "risk_analyst")
        graph.add_conditional_edges(
            "risk_analyst",
            self._should_continue_debate,
            {
                "market_analyst": "market_analyst",
                "portfolio_advisor": "portfolio_advisor",
            },
        )
        graph.add_conditional_edges(
            "market_analyst",
            self._should_continue_debate,
            {
                "risk_analyst": "risk_analyst",
                "portfolio_advisor": "portfolio_advisor",
            },
        )
        graph.add_edge("portfolio_advisor", END)

        return graph.compile()

    # ── Conditional routing ──────────────────────────────────────────

    def _should_continue_debate(self, state: dict) -> str:
        debate_state = state.get("portfolio_debate_state", {})
        count = debate_state.get("count", 0)
        current = debate_state.get("current_response", "")

        # Each round has 2 turns (one cautious + one optimistic)
        if count >= 2 * self.MAX_DEBATE_ROUNDS:
            return "portfolio_advisor"

        if current.startswith("Cautious"):
            return "market_analyst"

        return "risk_analyst"

    # ── Public API ───────────────────────────────────────────────────

    def run(
        self,
        portfolio_data: dict,
        ensemble_data: dict | None = None,
        sentiment_data: dict | None = None,
    ) -> str:
        """Ejecuta el pipeline completo de agentes sobre los datos del portafolio.

        Parameters
        ----------
        portfolio_data : dict
            Debe contener: weights (dict), expected_return, volatility,
            sharpe_ratio, var, tickers (list[str]).
        ensemble_data : dict, optional
            Resultados de todos los metodos de optimizacion.
        sentiment_data : dict, optional
            Datos de sentimiento/noticias por ticker.

        Returns
        -------
        str
            La recomendacion final del asesor de portafolios.
        """
        if self._no_key:
            return (
                "[Error de Agente] No hay API key configurada. "
                "Configura OPENAI_API_KEY o ANTHROPIC_API_KEY en tu archivo .env "
                "y asegurate de que LLM_PROVIDER este correctamente configurado en config.py."
            )

        tickers = portfolio_data.get("tickers", list(portfolio_data.get("weights", {}).keys()))

        # Fetch market context
        market_context = fetch_market_context(tickers)

        # Fetch sentiment if not provided
        if sentiment_data is None:
            try:
                from engine.sentiment import fetch_all_news
                sentiment_data = fetch_all_news(tickers, max_per_ticker=cfg.MAX_NEWS_PER_TICKER)
            except Exception:
                sentiment_data = {}

        initial_state: PortfolioAgentState = {
            "messages": [],
            "portfolio_data": portfolio_data,
            "quant_report": "",
            "market_context": market_context,
            "portfolio_debate_state": {
                "optimistic_history": "",
                "cautious_history": "",
                "history": "",
                "current_response": "",
                "count": 0,
            },
            "final_recommendation": "",
            "sentiment_data": sentiment_data or {},
            "ensemble_data": ensemble_data or {},
        }

        try:
            final_state = self._graph.invoke(initial_state)
            return final_state.get("final_recommendation", "[No se genero recomendacion]")
        except Exception as exc:
            return f"[Error de Agente] Fallo la ejecucion del grafo: {exc}"
