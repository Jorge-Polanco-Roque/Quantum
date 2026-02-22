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

## Como se Seleccionan los Pesos (%) del Portafolio

El sistema tiene **4 caminos** para determinar los porcentajes de cada activo en el portafolio. Todos convergen en el mismo pipeline de metricas y visualizacion.

### Diagrama General

```
                    ┌─────────────────────────────────────────────────────┐
                    │                ENTRADA DEL USUARIO                  │
                    └──────────┬──────────────┬──────────────┬────────────┘
                               │              │              │
                    ┌──────────▼──┐  ┌────────▼────────┐  ┌─▼──────────────┐
                    │  EJECUTAR   │  │  CONSTRUIR       │  │  SLIDERS       │
                    │  (boton)    │  │  PORTAFOLIO (NL) │  │  (manual)      │
                    └──────┬──────┘  └────────┬─────────┘  └─┬──────────────┘
                           │                  │              │
                    ┌──────▼──────┐  ┌────────▼─────────┐   │
                    │ yfinance    │  │ Agente ReAct LLM  │   │
                    │ precios     │  │ (LangGraph)       │   │
                    │ historicos  │  │                    │   │
                    └──────┬──────┘  │ Selecciona:       │   │
                           │         │ - tickers          │   │
                    ┌──────▼──────┐  │ - metodo           │   │
                    │ Retornos    │  │ (NUNCA pesos)      │   │
                    │ anualizados │  └────────┬───────────┘   │
                    │ + Covarianza│           │               │
                    └──────┬──────┘  ┌────────▼───────────┐   │
                           │         │ Seleccion de metodo │   │
                           │         │ (deterministico)    │   │
                    ┌──────▼──────┐  │                     │   │
                    │ SLSQP       │  │ optimize ──► SLSQP  │   │
                    │ Max Sharpe  │  │ equal_weight ► 1/N  │   │
                    │             │  │ risk_parity ► RP    │   │
                    │ max(Sharpe) │  │ min_variance ► MV   │   │
                    │ s.t. Σw=1   │  │ split ──► grupos    │   │
                    │ 0 ≤ w ≤ 1   │  └────────┬───────────┘   │
                    └──────┬──────┘           │               │
                           │                  │               │
                           └───────┬──────────┘               │
                                   │                          │
                          ┌────────▼─────────┐                │
                          │  WEIGHT FLOOR    │                │
                          │  min 2% por      │                │
                          │  ticker          │                │
                          │  (redistribuye   │                │
                          │  proporcionalmente│               │
                          │  de los mayores) │                │
                          └────────┬─────────┘                │
                                   │                          │
                          ┌────────▼──────────────────────────▼──┐
                          │         SLIDERS DEL DASHBOARD        │
                          │    (valores en % por ticker)         │
                          │                                      │
                          │  Botones:                             │
                          │  [OPTIMO] → restaura pesos optimales │
                          │  [IGUAL]  → 1/N (100%/n)             │
                          │  [RANDOM] → Dirichlet aleatorio      │
                          │  [NORM.]  → normaliza a 100%         │
                          │            (respeta locks)            │
                          └────────────────┬─────────────────────┘
                                           │
                          ┌────────────────▼─────────────────────┐
                          │        CALCULO DE METRICAS           │
                          │   w · μ = retorno esperado           │
                          │   √(w·Σ·wᵀ) = volatilidad           │
                          │   (ret - rf) / vol = Sharpe          │
                          │   VaR parametrico diario             │
                          └────────────────┬─────────────────────┘
                                           │
              ┌────────────────────────────▼──────────────────────────┐
              │                  VISUALIZACION                        │
              │  Frontera Eficiente + CML │ Barras de Pesos           │
              │  Correlacion (heatmap)    │ Contribucion al Riesgo    │
              │  Tabla Ensemble (9 metodos)│ Drawdown + Performance   │
              │  KPI cards (Sharpe, Ret, Vol, VaR)                    │
              └───────────────────────────────────────────────────────┘
```

### Camino 1: Boton EJECUTAR (Monte Carlo + SLSQP)

```
EJECUTAR ──► yfinance (1 ano de precios)
         ──► compute_returns (retornos logaritmicos)
         ──► get_annual_stats (μ anualizado, Σ covarianza)
         ──► run_monte_carlo (10,000 portafolios Dirichlet)
         ──► optimize_max_sharpe (SLSQP)
              │
              │  Minimiza: -(w·μ - rf) / √(w·Σ·wᵀ)
              │  Sujeto a: Σwᵢ = 1, 0 ≤ wᵢ ≤ 1
              │
         ──► _apply_weight_floor (min 2% por ticker)
         ──► Ensemble: corre los 7 metodos en paralelo
         ──► Actualiza sliders + graficos + metricas
```

El optimizer SLSQP (Sequential Least Squares Programming) encuentra los pesos que **maximizan el Sharpe Ratio** — el retorno por unidad de riesgo. Usa el solver de SciPy con restricciones: pesos suman 1.0, cada peso entre 0 y 1.

### Camino 2: Constructor NL (Lenguaje Natural)

```
"Portafolio 70% tech, 30% bonos"
         │
         ▼
  PortfolioBuilderAgent (ReAct LLM)
         │
         │  Tools disponibles (3):
         │  - validate_tickers: verifica existencia en Yahoo Finance
         │  - search_tickers_by_sector: busca en 106 categorias
         │  - fetch_and_analyze: estadisticas basicas
         │
         │  El LLM NUNCA asigna pesos. Solo retorna:
         │  {tickers: [...], method: "split", split: {groups: {...}}}
         │
         ▼
  Callback deterministico (segun method):
         │
         ├─ "ensemble"      → Ensemble Shrinkage δ adaptativo (default)
         ├─ "optimize"      → SLSQP Max Sharpe
         ├─ "equal_weight"  → 1/N (100% / num_tickers)
         ├─ "risk_parity"   → Iguala contribucion al riesgo
         ├─ "min_variance"  → Minimiza varianza total
         └─ "split"         → Distribucion por grupos:
                              Ej: equity 70% / 3 tickers = 23.3% c/u
                                  bonos  30% / 2 tickers = 15.0% c/u
         │
         ▼
  _apply_weight_floor (min 2%)
         │
         ▼
  Pipeline completo (MC + ensemble + graficos)
```

**Principio clave**: El LLM decide *que* activos incluir y *como* optimizar, pero **nunca** decide los porcentajes. Los pesos siempre se calculan con algoritmos deterministicos — el mismo input produce el mismo output.

### Camino 3: Ajuste Manual (Sliders)

```
  Usuario mueve slider de AAPL a 35%
         │
         ▼
  Callback pattern-matching (MATCH/ALL)
         │
         │  Lee TODOS los sliders simultáneamente
         │  Construye vector de pesos: [0.35, 0.15, ...]
         │  Normaliza: wᵢ = wᵢ / Σwⱼ
         │
         ▼
  calc_portfolio_metrics (recalculo en tiempo real)
         │
         ▼
  Actualiza KPI cards (Sharpe, Ret, Vol, VaR)
```

Botones de accion rapida:
- **OPTIMO**: restaura los pesos del ultimo resultado de optimizacion
- **IGUAL**: asigna 100%/N a cada ticker (pesos iguales)
- **RANDOM**: genera pesos aleatorios con distribucion Dirichlet(1,...,1)
- **NORM.**: normaliza a 100% respetando sliders bloqueados (lock)

### Camino 4: Ensemble (7 Metodos + Votacion)

El ensemble corre automaticamente con cada EJECUTAR y produce 9 conjuntos de pesos:

```
  Datos de entrada: μ (retornos), Σ (covarianza), rf (tasa libre)
         │
         ├──► Max Sharpe ──────── max (w·μ - rf) / √(w·Σ·wᵀ)
         ├──► Min. Varianza ───── min w·Σ·wᵀ
         ├──► Paridad de Riesgo ─ igualar RCᵢ = wᵢ(Σw)ᵢ / σₚ
         ├──► Max. Diversificacion  max Σ(wᵢσᵢ) / σₚ
         ├──► HRP ─────────────── clustering jerarquico + biseccion
         ├──► Min. CVaR ───────── min E[pérdida | pérdida > VaR]
         └──► Pesos Iguales ──── 1/N (baseline)
                    │
                    ▼
         ┌─────────────────────────────────┐
         │  ENSEMBLE VOTING                │
         │                                 │
         │  Promedio Simple:               │
         │    wₑ = (1/7) Σ wₖ             │
         │                                 │
         │  Promedio Ponderado por Sharpe:  │
         │    wₑ = Σ (Sₖ/ΣSⱼ) · wₖ      │
         │    (Sₖ = Sharpe del metodo k)   │
         └─────────────────────────────────┘
```

Los 10 resultados (7 metodos + 3 ensembles) se muestran en la **Tabla Ensemble** del dashboard, permitiendo comparar los pesos y metricas de cada enfoque lado a lado.

#### Ensemble Shrinkage (delta adaptativo)

Ademas del promedio simple y Sharpe-weighted, el ensemble incluye un metodo de **shrinkage adaptativo** que combina el promedio ponderado por Sharpe con 1/N:

```
w_final = δ · w_sharpe_weighted + (1 - δ) · w_equal
```

El coeficiente δ se calcula automaticamente a partir del HHI (Indice Herfindahl-Hirschman) de los pesos Sharpe:
- **HHI bajo** (los metodos contribuyen de forma equilibrada) → δ alto (~0.7) → mayor confianza en el ensemble
- **HHI alto** (un metodo domina) → δ bajo (~0.3) → mayor ancla a 1/N para evitar sobreajuste

Este es el **metodo default** del Constructor NL, basado en la literatura academica (DeMiguel et al. 2009, Tu & Zhou 2011) que demuestra que combinar metodos con shrinkage hacia 1/N es mas robusto que cualquier metodo individual.

### Weight Floor (Piso Minimo)

Todos los caminos de optimizacion aplican `_apply_weight_floor()` que garantiza un **minimo de 2%** por ticker. Esto evita posiciones negligibles y asegura diversificacion minima.

```
Antes del floor:  AAPL=45%, MSFT=30%, GOOGL=20%, META=4%, AMZN=1%, TSLA=0%, NVDA=0%
                                                                     ▲         ▲
                                                              deficit: 1%     2%

Redistribucion:   Se toma 3% de {AAPL, MSFT, GOOGL, META} proporcionalmente
                  y se distribuye a {TSLA=2%, NVDA=2%, AMZN=2%}

Despues del floor: AAPL=43.6%, MSFT=29.1%, GOOGL=19.4%, META=3.9%, AMZN=2%, TSLA=2%, NVDA=2%
                   Σ = 100% ✓   Todos ≥ 2% ✓
```

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
| **Ensemble Shrinkage** | Shrinkage adaptativo: δ · Sharpe-weighted + (1-δ) · 1/N |

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
