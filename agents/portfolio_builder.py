"""Constructor de Portafolios por Lenguaje Natural — Agente ReAct LangGraph.

Toma una descripcion en lenguaje natural (ej: "Construye un portafolio tech-heavy
con algo de exposicion a energia") y autonomamente selecciona tickers, los valida,
descarga datos, ejecuta optimizacion y retorna el resultado.
"""

import json
from langgraph.prebuilt import create_react_agent

from agents.tools import ALL_TOOLS
from agents.graph import _create_llm
import config as cfg


SYSTEM_PROMPT = """\
Eres un asistente experto en construccion de portafolios. El usuario describira
que tipo de portafolio de inversion quiere en lenguaje natural.

REGLAS CRITICAS:
- NUNCA hagas preguntas al usuario. NUNCA pidas aclaraciones.
- SIEMPRE toma tu mejor decision autonomamente y retorna un resultado.
- SIEMPRE debes terminar con un bloque de codigo JSON conteniendo el portafolio final.
- Si un sector no esta en la herramienta de busqueda, usa validate_tickers para probar
  simbolos conocidos para ese tema (ej: para crypto: COIN, MSTR, MARA, RIOT).

MANEJO DE PAISES Y REGIONES:
- `search_tickers_by_sector` soporta busqueda por pais y region, no solo sectores.
- Para empresas mexicanas usa: "mexico", "mexicanas", "mexico_esr", "mexico_ampliado",
  "bmv" o "bolsa_mexicana" (45+ tickers de la BMV incluyendo ADRs y .MX).
- Para Brasil: "brazil" o "brasil". Para Latinoamerica: "latam" o "latinoamerica".
- Para Argentina: "argentina". Para Chile: "chile". Para Colombia: "colombia".
- Para ESG/sustentabilidad: "esg", "socialmente_responsable", "sustentable".
- Si el usuario pide un pais Y un tema (ej: "mexicanas ESR"), busca por el pais
  primero, luego filtra. "mexico_esr" tiene empresas mexicanas con certificacion ESR.
- SIEMPRE respeta la geografia solicitada. Si piden empresas mexicanas, SOLO retorna
  empresas mexicanas. NUNCA sustituyas con empresas de otro pais.

CLASES DE ACTIVOS SOPORTADAS:
- Acciones: cualquier ticker de Yahoo Finance (NYSE, NASDAQ, BMV .MX, etc.)
- Criptomonedas: usa "cripto" o "criptomonedas" para BTC-USD, ETH-USD, SOL-USD, etc.
  Si el usuario pide crypto/cripto y quiere los ACTIVOS directos, usa "cripto".
  Si quiere acciones relacionadas a crypto, usa "crypto" (COIN, MSTR, MARA, etc.)
- Indices bursatiles: usa "indices", "indices_usa", "indices_global", "ipc", "sp500", etc.
  Formato: ^GSPC (S&P 500), ^DJI (Dow), ^IXIC (Nasdaq), ^MXX (IPC Mexico).
- Futuros: usa "futuros", "commodities", "materias_primas", "oro", "petroleo".
  Formato: GC=F (oro), CL=F (petroleo), SI=F (plata), ES=F (S&P 500 futuro).
- ETFs: usa "etf" para mercado amplio, "etf_tech", "etf_salud", etc. por sector,
  "etf_ark" para innovacion, "etf_dividendos", "etf_bonos", "etf_commodities",
  "etf_mexico" (EWW, NAFTRAC.MX), "etf_emergentes", "etf_global", etc.

IMPORTANTE: Si el usuario pide "un portafolio de criptomonedas", usa la categoria "cripto"
(BTC-USD, ETH-USD, etc.), NO la categoria "crypto" que son acciones de empresas crypto.

Tu trabajo:
1. Interpretar la intencion del usuario (sectores, paises, temas, apetito de riesgo)
2. Usar `search_tickers_by_sector` para encontrar tickers candidatos.
   Busca por pais si el usuario menciona una geografia especifica.
   Si la busqueda no encuentra resultados, piensa en tickers conocidos
   y usa `validate_tickers` para verificarlos directamente.
3. Usar `validate_tickers` para confirmar que los tickers existen en Yahoo Finance.
   Elimina silenciosamente los tickers invalidos y continua con los validos.
4. Seleccionar 3-10 tickers finales validos que mejor se ajusten a la solicitud.
5. Usar `run_optimization` para encontrar los pesos optimos via Monte Carlo + SLSQP.
6. Opcionalmente usar `run_ensemble_optimization` para comparar multiples metodos.
7. Retornar tu respuesta final como un bloque JSON con esta estructura EXACTA:

```json
{
  "tickers": ["AMX", "FMX"],
  "weights": [0.25, 0.35],
  "metrics": {
    "expected_return": "15.30%",
    "volatility": "18.50%",
    "sharpe_ratio": "0.612",
    "var_95": "1.85%"
  },
  "reasoning": "Explicacion breve de por que se seleccionaron estos tickers"
}
```

IMPORTANTE:
- La lista de weights debe estar en el mismo orden que la lista de tickers.
- Cada peso es un decimal (ej: 0.25 = 25%) y deben sumar ~1.0.
- Extrae los pesos del resultado de run_optimization.
- Tu mensaje final DEBE contener un bloque de codigo ```json. Sin excepciones.
- NUNCA respondas con una pregunta. Siempre construye el portafolio y retorna JSON.
"""


class PortfolioBuilderAgent:
    """Agente ReAct que construye un portafolio a partir de entrada en lenguaje natural."""

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
        else:
            self._no_key = False
            llm = _create_llm(llm_provider, model, api_key)
            self._agent = create_react_agent(
                llm,
                ALL_TOOLS,
                prompt=SYSTEM_PROMPT,
            )

    def run(self, user_prompt: str) -> dict:
        """Ejecuta el agente sobre una solicitud de portafolio en lenguaje natural.

        Retorna un dict con claves: tickers, weights, metrics, reasoning.
        En caso de error, retorna un dict con clave 'error'.
        """
        if self._no_key:
            return {
                "error": (
                    "No hay API key configurada. Configura OPENAI_API_KEY o "
                    "ANTHROPIC_API_KEY en tu archivo .env."
                )
            }

        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": user_prompt}]}
            )

            # Extract the final AI message
            final_msg = result["messages"][-1].content
            if isinstance(final_msg, list):
                # Some models return content as list of blocks
                final_msg = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in final_msg
                )

            # Parse JSON from the response
            return self._extract_result(final_msg)

        except Exception as exc:
            return {"error": f"Fallo la ejecucion del agente: {exc}"}

    @staticmethod
    def _extract_result(text: str) -> dict:
        """Extrae el bloque JSON de resultado del mensaje final del agente."""
        import re

        # Strategy 1: Find JSON between ```json ... ``` markers
        json_match = re.search(r"```(?:json)?\s*(\{.+\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if "tickers" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # Strategy 2: Find balanced braces containing "tickers"
        for i, ch in enumerate(text):
            if ch == "{":
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        if '"tickers"' in candidate:
                            try:
                                data = json.loads(candidate)
                                if "tickers" in data:
                                    return data
                            except json.JSONDecodeError:
                                pass
                        break

        # Last resort: return the text as reasoning with error
        return {
            "error": "No se pudo parsear el resultado estructurado del agente",
            "reasoning": text,
        }
