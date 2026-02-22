# Quantum — Quant Portfolio Optimizer

Optimizador de portafolios interactivo con simulacion Monte Carlo, optimizacion ensemble (7 metodos), analisis de sentimiento de mercado y un sistema multi-agente de IA para analisis cuantitativo y fundamental.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-Plotly-00d4aa?logo=plotly&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agents-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## Caracteristicas

- **Monte Carlo + SLSQP** — 10,000 simulaciones vectorizadas + optimizacion del Sharpe Ratio
- **Ensemble de 7 metodos** — Max Sharpe, Min Varianza, Paridad de Riesgo, Max Diversificacion, HRP, Min CVaR, Pesos Iguales + votacion ensemble
- **334 activos pre-configurados** — Acciones (NYSE, NASDAQ, BMV), criptomonedas, indices, futuros y 80+ ETFs organizados en 90 categorias
- **Sentimiento de mercado** — Noticias en tiempo real via yfinance con scoring por keywords (ES/EN)
- **4 agentes de IA** — Analista Cuantitativo, Analista de Riesgo, Analista de Mercado y Asesor de Portafolios con sistema de debate (LangGraph)
- **Analista Fundamental AI** — Combina metricas cuantitativas con noticias para una ponderacion fundamental
- **Constructor de Portafolios NL** — Describe tu portafolio en lenguaje natural y el agente lo construye automaticamente
- **Dashboard interactivo** — Dark mode, 10+ graficos Plotly, sliders con locks, metricas en tiempo real
- **UI en Espanol** — Toda la interfaz y reportes de agentes en espanol

## Quick Start

```bash
# Clonar el repositorio
git clone https://github.com/Jorge-Polanco-Roque/Quantum.git
cd Quantum

# Instalar dependencias
pip install -r requirements.txt

# Configurar API keys
cp .env.template .env
# Editar .env con tu OPENAI_API_KEY o ANTHROPIC_API_KEY

# Ejecutar
python app.py
# Abrir http://localhost:8050
```

## Requisitos

- Python 3.10+
- Una API key de OpenAI o Anthropic (para los agentes de IA)

## Uso

### Optimizacion Basica
1. Haz click en **EJECUTAR** para correr Monte Carlo con los Magnificent 7
2. Ajusta los sliders de peso manualmente o usa los botones IGUAL/RANDOM/NORM
3. Las metricas (Sharpe, Retorno, Volatilidad, VaR) se actualizan en tiempo real
4. El analisis de IA se ejecuta automaticamente despues de cada optimizacion

### Constructor de Portafolios por Lenguaje Natural
Escribe solicitudes como:
- *"Portafolio de criptomonedas"*
- *"Empresas mexicanas del BMV"*
- *"ETFs de dividendos con renta fija"*
- *"Portafolio tech con algo de energia renovable"*
- *"Futuros de commodities: oro, petroleo y maiz"*

### Clases de Activos Soportadas
| Clase | Ejemplos | Formato |
|-------|----------|---------|
| Acciones USA | AAPL, MSFT, NVDA | Ticker directo |
| Acciones BMV | BIMBOA.MX, WALMEX.MX | Ticker.MX |
| ADRs Mexico | AMX, FMX, KOF | Ticker directo |
| Criptomonedas | BTC-USD, ETH-USD, SOL-USD | TICKER-USD |
| Indices | ^GSPC, ^MXX, ^DJI | ^TICKER |
| Futuros | GC=F, CL=F, SI=F | TICKER=F |
| ETFs | SPY, QQQ, NAFTRAC.MX | Ticker directo |

## Arquitectura

```
quantum/
├── config.py              # Configuracion centralizada
├── app.py                 # Entry point (Dash)
├── engine/                # Motor cuantitativo (NumPy/SciPy)
│   ├── data.py            # Descarga de precios (yfinance)
│   ├── monte_carlo.py     # Simulacion Monte Carlo (Dirichlet)
│   ├── optimizer.py       # SLSQP, frontera eficiente, CML
│   ├── risk.py            # Metricas, VaR, contribucion al riesgo
│   ├── ensemble.py        # 7 metodos de optimizacion + voting
│   └── sentiment.py       # Noticias + scoring de sentimiento
├── agents/                # Sistema multi-agente (LangGraph)
│   ├── graph.py           # Grafo de debate con 4 agentes
│   ├── fundamental_analyst.py  # Analisis fundamental con LLM
│   ├── portfolio_builder.py    # Agente ReAct para NL
│   └── tools.py           # 90 categorias, 334 tickers
├── dashboard/             # UI (Dash + Plotly, dark mode)
│   ├── layout.py          # Layout de 7 filas
│   ├── callbacks.py       # 11 callbacks
│   └── components/        # 12 componentes modulares
└── referencias/           # Material de referencia
```

## Stack Tecnologico

| Capa | Tecnologia |
|------|-----------|
| Simulacion | NumPy (Dirichlet + einsum) |
| Optimizacion | SciPy SLSQP + scipy.cluster.hierarchy |
| Datos | yfinance |
| Dashboard | Dash + Plotly + Bootstrap |
| Agentes IA | LangGraph + LangChain |
| LLM | OpenAI / Anthropic (configurable) |

## Metodos de Optimizacion Ensemble

| Metodo | Descripcion |
|--------|-------------|
| Max Sharpe | Maximiza el ratio riesgo-retorno (SLSQP) |
| Min. Varianza | Minimiza la varianza del portafolio |
| Paridad de Riesgo | Iguala la contribucion al riesgo por activo |
| Max. Diversificacion | Maximiza el ratio de diversificacion |
| HRP | Hierarchical Risk Parity (Lopez de Prado) |
| Min. CVaR | Minimiza el Conditional Value at Risk |
| Pesos Iguales | Baseline 1/N |
| **Ensemble Promedio** | Promedio simple de todos los metodos |
| **Ensemble Sharpe** | Promedio ponderado por Sharpe Ratio |

## Configuracion

Edita `.env` para configurar:

```env
LLM_PROVIDER=openai          # openai o anthropic
OPENAI_API_KEY=sk-...        # Tu API key
LLM_MODEL=gpt-4o-mini        # Modelo a usar
DASH_DEBUG=true               # Modo debug
DASH_PORT=8050                # Puerto del servidor
```

## Contribuir

1. Fork el repositorio
2. Crea tu branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agrega nueva funcionalidad'`)
4. Push al branch (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Licencia

MIT License - ver [LICENSE](LICENSE) para mas detalles.

---

Desarrollado con Claude Code (claude.ai/code)
