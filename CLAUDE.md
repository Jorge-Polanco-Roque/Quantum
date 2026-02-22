# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Quant Portfolio Optimizer** — an interactive Monte Carlo portfolio optimization tool with ensemble optimization (7 methods), market sentiment analysis, and a multi-agent AI debate system. Supports any set of stock tickers (defaults to the Magnificent 7) with a natural language portfolio construction agent that handles sector, geographic, and thematic queries (including Mexican, Brazilian, LatAm companies and ESG themes). UI entirely in Spanish.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.template .env  # edit with your API keys
python app.py          # open http://localhost:8050
```

## Architecture

```
quantum/
├── config.py                       # Centralized config (tickers, theme, LLM, defaults)
│                                   #   get_ticker_color(), get_ticker_name() (cached via yfinance)
├── app.py                          # Dash entry point
├── requirements.txt
├── .env.example
├── engine/                         # Quantitative engine (pure NumPy/SciPy)
│   ├── data.py                     # fetch_stock_data, compute_returns, get_annual_stats
│   ├── monte_carlo.py              # run_monte_carlo (Dirichlet + einsum), get_optimal_portfolio
│   ├── optimizer.py                # optimize_max_sharpe (SLSQP), compute_efficient_frontier, CML
│   ├── risk.py                     # calc_portfolio_metrics, normalize_weights, calc_risk_contribution
│   ├── ensemble.py                 # 7 optimization methods + ensemble voting + shrinkage
│   │                               #   run_all_methods, ensemble_vote, ensemble_shrinkage
│   └── sentiment.py                # News via yfinance .news + keyword scoring (fetch_all_news)
│                                   #   Handles yfinance >= 1.x nested content format
├── agents/                         # Multi-agent system (LangGraph, TradingAgents pattern)
│   ├── states.py                   # PortfolioAgentState (with sentiment_data, ensemble_data)
│   ├── quant_analyst.py            # Analyzes MC results + ensemble + sentiment (Spanish prompts)
│   ├── risk_analyst.py             # Cautious debater — VaR, concentration, tail risk (Spanish)
│   ├── market_analyst.py           # Optimistic debater + fetch_market_context (Spanish)
│   ├── portfolio_advisor.py        # Judge: synthesizes debate + ensemble + sentiment (Spanish)
│   ├── graph.py                    # PortfolioAgentsGraph (passes sentiment_data + ensemble_data)
│   │                               #   _create_llm(provider, model, key, temperature=0.0)
│   │                               #   All agents share this factory (deterministic by default)
│   ├── fundamental_analyst.py      # Single-call LLM: combines quant metrics + news → fundamental analysis
│   ├── tools.py                    # 6 @tool functions incl. run_ensemble_optimization (Spanish docstrings)
│   │                               #   SECTOR_TICKERS: ~106 categories, ~380 unique tickers
│   │                               #   TICKER_ALIASES: short fund names → Yahoo Finance .MX tickers
│   │                               #   Covers: sectors, countries, cryptos, indices, futures, ETFs, UCITS, SIC
│   │                               #   ALL_TOOLS (6): full set for chatbot v2 / other consumers
│   │                               #   BUILDER_TOOLS (3): validate, search, analyze (no optimization)
│   ├── portfolio_builder.py        # PortfolioBuilderAgent (deterministic: selects tickers + method only)
│   │                               #   LLM returns {tickers, method, reasoning} — NO weights
│   │                               #   Exception: method="preset" includes user-specified weights
│   │                               #   Uses BUILDER_TOOLS (3) — no access to optimization tools
│   │                               #   Weight computation happens in dashboard callback
│   └── chatbot.py                  # ChatbotAgent (conversational, InMemorySaver, portfolio context)
│                                   #   _format_context: full ensemble weights per ticker + metrics
├── dashboard/                      # Dash UI (dark mode, Plotly charts, all in Spanish)
│   ├── layout.py                   # Adaptive layout: dashboard (3/4) + chat sidebar (1/4)
│   │                               #   Chat dynamically repositionable: right/left/top/bottom
│   │                               #   IDs: app-layout, dashboard-main (for inline style switching)
│   │                               #   Dashboard has 7 rows: header, NL+params+metrics,
│   │                               #   sliders+charts, corr+risk_decomp+ensemble,
│   │                               #   drawdown+performance, sentiment, agents.
│   │                               #   Stores: mc, optimal, stats, tickers, ensemble-results,
│   │                               #   prices-data, chat-history, chat-thread-id, chat-position.
│   ├── callbacks.py                # 16 callbacks: chat-position (CB0), ejecutar, sliders,
│   │                               #   optimo, igual, random, norm, agents (manual),
│   │                               #   auto-agents (store-triggered), NL builder, weight
│   │                               #   displays, sentiment+fundamental (manual + auto),
│   │                               #   chat-init, chat-send
│   │                               #   Helpers: _compute_position_styles(pos) → 3 style dicts
│   │                               #   _compute_split_weights(tickers, split_data) → deterministic split
│   │                               #   _apply_weight_floor(weight_map, min_floor) → floor redistribution
│   │                               #   _build_sentiment_output() → shared by manual + auto sentiment
│   ├── components/
│   │   ├── parameters.py           # Tasa Libre%, Simulaciones, VaR Nivel% inputs
│   │   ├── metrics_cards.py        # 4 KPI cards (Sharpe, Return, Vol, VaR)
│   │   ├── sliders.py              # Dynamic sliders with pattern-matching IDs + lock toggles
│   │   │                           #   Ticker labels show company name on hover (title attr)
│   │   ├── frontier_chart.py       # Scatter cloud + EF line + CML + markers (Spanish legends)
│   │   ├── weights_chart.py        # Horizontal bar chart (hover shows company name)
│   │   ├── correlation_chart.py    # Heatmap de correlacion (RdBu colorscale)
│   │   ├── risk_decomposition.py   # Asignacion vs Contribucion al Riesgo (grouped bars)
│   │   ├── ensemble_table.py       # DataTable: 10 methods side-by-side, highlights best Sharpe
│   │   ├── drawdown_chart.py       # Portfolio drawdown area chart (max drawdown annotated)
│   │   ├── performance_chart.py    # Normalized historical lines per asset + portfolio
│   │   ├── sentiment_panel.py      # News button + per-ticker sentiment + fundamental analysis
│   │   ├── agent_panel.py          # AI analysis output panel (auto-triggered + manual re-run)
│   │   ├── nl_input.py             # Natural language portfolio input panel
│   │   └── chat_widget.py          # Chat sidebar panel (1/4, dynamic position toggles)
│   │                               #   id=chat-sidebar, 4 toggle buttons (◀▲▼▶)
│   │                               #   store-chat-position (right|left|top|bottom)
│   └── assets/
│       └── style.css               # Dark mode theme (#0d1117, accent #00d4aa)
│                                   #   Scoped heading styles for agent-panel, #sentiment-output, chat
│                                   #   .chat-pos-btn / .chat-pos-btn-active toggle styles
└── referencias/                    # Reference materials (do not modify)
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Simulation | NumPy (Dirichlet + einsum) | 10K portfolios vectorized |
| Optimization | SciPy SLSQP + hierarchy | Max Sharpe, Min Var, Risk Parity, HRP, Max Div, Min CVaR, Ensemble |
| Data | yfinance | Historical prices, fundamentals, news (.news content format) |
| Sentiment | yfinance .news + keyword scoring | Bilingual ES/EN sentiment per ticker |
| Dashboard | Dash + Plotly | Interactive dark mode UI (Spanish) |
| Agents | LangGraph + LangChain | Multi-agent debate + NL builder + fundamental analyst + chatbot |
| LLM | OpenAI / Anthropic (configurable) | AI portfolio analysis |

## Data Flow

1. **EJECUTAR** click → `engine/data.py` fetches prices → `monte_carlo.py` runs 10K sims → `optimizer.py` finds optimal via SLSQP → `ensemble.py` runs all 7 methods + voting + shrinkage → `risk.py` calculates metrics + risk contribution → dashboard updates all 7 chart sections + sliders + ensemble table → **auto-triggers AI Analysis** (see #4) + **auto-triggers Sentiment/Fundamental** (see #3)
2. **Slider changes** → `risk.py` recalculates metrics in real-time → updates metric cards
3. **Sentimiento + Fundamental (auto-triggered + manual)** → fires automatically when `store-optimal-weights` changes (also available via ACTUALIZAR NOTICIAS button) → `engine/sentiment.py` fetches yfinance news → keyword scoring → **`agents/fundamental_analyst.py`** combines quant metrics + news into a ponderacion cuantitativo-fundamental → renders per-ticker sentiment + AI fundamental analysis
4. **Analisis AI (auto-triggered)** → fires automatically when `store-optimal-weights` changes (from EJECUTAR or NL Builder) → `agents/graph.py` runs LangGraph workflow with ensemble + sentiment data: Quant Analyst → Risk Analyst ↔ Market Analyst (debate) → Portfolio Advisor → displays recommendation. Also available as manual re-run via button.
5. **CONSTRUIR PORTAFOLIO** → `agents/portfolio_builder.py` runs ReAct agent with BUILDER_TOOLS (3): interprets NL prompt → selects tickers → validates → returns `{tickers, method, reasoning}` (NO weights, except `"preset"`). Callback computes weights deterministicaly: `method="preset"` → user-specified percentages passed through, `"ensemble"` → ensemble shrinkage (default), `"optimize"` → SLSQP, `"equal_weight"` → 1/N, `"risk_parity"` → engine risk parity, `"min_variance"` → engine min variance, `"split"` → `_compute_split_weights()`. Then `_run_full_pipeline(preset_weights=weight_map, method=method)` → `_apply_weight_floor()` ensures min 2% per ticker → updates `store-tickers` + runs full pipeline (MC + ensemble + all charts) → **auto-triggers AI Analysis** (see #4) + **auto-triggers Sentiment** (see #3). Same prompt always produces identical weights.
6. **Chatbot** → always-visible sidebar (1/4, repositionable via toggle buttons) → user types question → callback reads all portfolio stores → builds context dict → `agents/chatbot.py` ChatbotAgent.chat() with InMemorySaver for multi-turn memory → returns markdown response rendered in chat bubble.
7. **Chat position toggle** → click ◀▲▼▶ buttons → `switch_chat_position` callback updates inline styles on `#app-layout`, `#dashboard-main`, `#chat-sidebar` via `_compute_position_styles(pos)` → CSS `flex-direction` + `order` switch without DOM rebuild → chat history and all stores preserved.

## Ensemble Optimization (engine/ensemble.py)

7 methods with unified signature `(mean_returns, cov_matrix, rf) -> np.ndarray`:
- **Max Sharpe** — maximizes Sharpe ratio (existing SLSQP)
- **Min. Varianza** — minimizes portfolio variance
- **Paridad de Riesgo** — equalizes risk contribution per asset (Spinu 2013)
- **Max. Diversificacion** — maximizes diversification ratio
- **HRP** — Hierarchical Risk Parity (Lopez de Prado, scipy.cluster.hierarchy)
- **Min. CVaR** — minimizes Conditional VaR with parametric scenarios
- **Igual Peso** — baseline 1/N

Ensemble combines via: simple average + Sharpe-weighted average + **shrinkage ensemble** (adaptive delta blending Sharpe-weighted with 1/N based on HHI concentration).

## Agent Systems

### TradingAgents Debate System (Spanish prompts)
Graph flow: `START → quant_analyst → risk_analyst → conditional_edge → [market_analyst | risk_analyst | portfolio_advisor] → END`

Debate routing: `count >= 2 * MAX_DEBATE_ROUNDS` → advisor; "Cautious:" prefix → market_analyst; else → risk_analyst.

All agents receive `sentiment_data` and `ensemble_data` in state. Prompts are entirely in Spanish. The debate auto-triggers when `store-optimal-weights` changes (no manual click needed).

### Fundamental Analyst (agents/fundamental_analyst.py)
Single LLM call that combines quantitative portfolio metrics with news sentiment. Triggered by the "ACTUALIZAR NOTICIAS" button. Produces a markdown analysis covering: news impact per asset, sentiment vs quantitative alignment, uncaptured fundamental risks, and a final recommendation.

### NL Portfolio Builder (Deterministic — ReAct Agent, Spanish prompt)
Uses `langgraph.prebuilt.create_react_agent` with `BUILDER_TOOLS` (3 tools) from `agents/tools.py`:
- `validate_tickers` — checks ticker existence via yfinance
- `search_tickers_by_sector` — returns tickers for sectors, countries, cryptos, indices, futures, ETFs, UCITS, SIC (~101 categories)
- `fetch_and_analyze` — fetches data + returns summary stats (return, vol, correlation)

The LLM has NO access to optimization tools — it only selects tickers and chooses a method.
Returns `{tickers, method, reasoning}` (and `split` when method="split"). Weight computation
happens deterministically in the dashboard callback via `_compute_split_weights()`, engine
optimizers (`risk_parity_portfolio`, `min_variance_portfolio`, `optimize_max_sharpe`), or
equal weight. `_apply_weight_floor()` ensures every ticker gets >= 2%.

Methods: `"preset"` (user-specified exact percentages per ticker), `"ensemble"` (ensemble shrinkage with adaptive delta, default),
`"optimize"` (SLSQP Max Sharpe), `"equal_weight"` (1/N), `"risk_parity"`, `"min_variance"`, `"split"` (with groups specification).

### Chatbot Interactivo (agents/chatbot.py)
Conversational agent using `create_react_agent` + `InMemorySaver` for multi-turn memory. Receives full portfolio context (tickers, weights, metrics, ensemble with full per-ticker weights and metrics per method, sentiment) injected as `[CONTEXTO DEL PORTAFOLIO]` block on each message. System prompt in Spanish. No tools (v1 — pure conversational with context). Thread ID (UUID) generated per page load.

Dashboard integration: always-visible sidebar (`chat_widget.py`, 300px fixed, dynamically repositionable) with 3 callbacks — init (generate thread_id), send (build context from stores → ChatbotAgent.chat() → render markdown bubbles), position toggle (switch_chat_position). Layout uses CSS flexbox: `.app-layout` (100vw × 100vh flex row) → `.dashboard-main` (flex: 1 1 0%, fills remaining space, min-width: 0) + `.chat-sidebar` (flex: 0 0 300px, rigid). Dash internal wrappers (`#react-entry-point`, `#_dash-app-content`) are explicitly sized to 100% to prevent layout interference. Position toggle buttons (◀▲▼▶) in chat header switch between right/left/top/bottom by changing inline styles on 3 container IDs (`app-layout`, `dashboard-main`, `chat-sidebar`) — no DOM rebuild, all state preserved. `_compute_position_styles(pos)` returns the 3 style dicts per position. Left/Right: only order + border changes (no flex overrides). Top/Bottom: flex-direction column + sidebar width: 100% + height: 25vh. Responsive: stacks vertically below 1200px.

### SECTOR_TICKERS Asset Coverage (agents/tools.py)
~106 curated mappings, ~380 unique tickers:
- **Sectors** (20): technology, energy, healthcare, finance, consumer, industrial, real_estate, utilities, telecom, semiconductor, ai, ev, cloud, renewable, crypto (stocks), fintech, defense, biotech, retail, gaming
- **Countries/Regions** (14): mexico, mexicanas, mexico_esr, mexico_ampliado, bmv/bolsa_mexicana (45 tickers), brazil/brasil, latam/latinoamerica, chile, colombia, argentina
- **Themes** (6): esg, socialmente_responsable, sustainable/sustentable, magnificent7/mag7
- **Criptomonedas** (3): cripto/criptomonedas/crypto_directo — 14 pares USD (BTC-USD, ETH-USD, SOL-USD, etc.)
- **Indices** (7): indices, indices_usa, indices_global, ipc (^MXX), sp500 (^GSPC), nasdaq, dow_jones
- **Futuros** (6): futuros/futures, commodities/materias_primas, oro/gold, petroleo/oil
- **ETFs mercado** (3): etf/etfs/etf_mercado (SPY, QQQ, IWM, etc.)
- **ETFs sector** (7): etf_tech, etf_salud, etf_financiero, etf_energia, etf_industrial, etf_consumo, etf_inmobiliario
- **ETFs tematicos** (6): etf_ark/etf_innovacion, etf_dividendos, etf_dividendos_global, etf_bonos/etf_renta_fija
- **ETFs commodities** (3): etf_oro, etf_plata, etf_commodities
- **ETFs pais/region** (7): etf_mexico/naftrac, etf_brasil, etf_china, etf_emergentes, etf_europa, etf_japon, etf_global
- **UCITS** (7): ucits/ucits_equity (VWRL.L, CSPX.L, IWDA.L, etc.), ucits_bonos/ucits_renta_fija (AGBP.L, IGLT.L, etc.), ucits_dividendos, ucits_oro, ucits_commodities
- **SIC Mexico** (2): sic/sic_mexico — ~31 ETFs extranjeros disponibles en el Sistema Internacional de Cotizaciones
- **Fondos Mexico** (5): fondos_mexico/fondos_inversion, fondos_blackrock/blackrock_mexico, fondos_gbm — BlackRock (BLKEM1, GOLD5+, BLKGUB1, BLKINT, BLKDOL, BLKLIQ, BLKRFA) + GBM (GBMCRE, GBMMOD, GBMF2, GBMINT)

### TICKER_ALIASES (agents/tools.py)
Maps common short fund names to Yahoo Finance series tickers. `validate_tickers` resolves aliases automatically:
- BLKEM1 → BLKEM1C0-A.MX, GOLD5+ → GOLD5+B2-C.MX, BLKGUB1 → BLKGUB1B0-D.MX
- BLKINT → BLKINTC0-A.MX, BLKDOL → BLKDOLB0-D.MX, BLKLIQ → BLKLIQB0-D.MX, BLKRFA → BLKRFAB0-C.MX
- GBMCRE → GBMCREB2-A.MX, GBMMOD → GBMMODB2-A.MX, GBMF2 → GBMF2BO.MX, GBMINT → GBMINTB1.MX

## Dynamic Dashboard

The dashboard uses **Dash pattern-matching callbacks** (`MATCH`/`ALL`) for slider components:
- Slider IDs: `{"type": "slider", "index": ticker}`
- Lock IDs: `{"type": "lock", "index": ticker}`
- Weight display IDs: `{"type": "weight-display", "index": ticker}`

Ticker labels show the full company name on hover via HTML `title` attribute. The weights chart also shows company names in hover tooltips.

### Stores
- `store-tickers` — current ticker list
- `store-mc-results` — MC simulation results
- `store-optimal-weights` — optimal weights from SLSQP (triggers auto-agent callback)
- `store-annual-stats` — annualized mean returns + covariance matrix
- `store-ensemble-results` — all method results for agent consumption
- `store-prices-data` — serialized prices for drawdown/performance charts
- `store-chat-history` — chat message history (list of {role, content} dicts)
- `store-chat-thread-id` — UUID for chatbot memory thread (generated on page load)
- `store-chat-position` — current chat sidebar position (right|left|top|bottom)

## Key Config (config.py)

- `TICKERS`: Default Magnificent 7 (backwards compatible)
- `get_ticker_color(ticker)`: Returns deterministic color for any ticker
- `get_ticker_name(ticker)`: Returns company name via yfinance (cached in `_TICKER_NAME_CACHE`)
- `DEFAULT_NUM_SIMULATIONS`: 10,000
- `DEFAULT_RISK_FREE_RATE`: 4%
- `DEFAULT_VAR_CONFIDENCE`: 95%
- `LLM_PROVIDER` / `LLM_MODEL`: Set via .env
- `THEME`: Dark mode color palette
- `SLSQP_MAX_ITER` / `SLSQP_FTOL`: SLSQP optimizer defaults (1000 / 1e-12)
- `RISK_PARITY_MAX_ITER` / `RISK_PARITY_FTOL`: Risk parity optimizer (2000 / 1e-14)
- `MIN_WEIGHT_BOUND`: Lower bound for risk parity weights (1e-6)
- `EF_NUM_POINTS`: Efficient frontier resolution (80 points)
- `CML_MAX_VOL_MULTIPLIER`: CML extension factor (1.1)
- `CVAR_NUM_SCENARIOS` / `CVAR_CONFIDENCE`: CVaR simulation params (5000 / 0.95)
- `RANDOM_SEED`: Reproducibility seed (42)
- `MAX_NEWS_PER_TICKER`: News fetch limit per ticker (3)
- `AGENT_RECURSION_LIMIT`: ReAct agent max turns (40)
- `ENSEMBLE_DELTA_MIN` / `ENSEMBLE_DELTA_MAX`: Shrinkage delta range (0.3 / 0.7)

## CSS Scoped Styles (dashboard/assets/style.css)

Markdown rendered inside `.agent-panel`, `#sentiment-output`, and `.chat-bubble-assistant` has scoped heading sizes (h1: 0.85rem, h2: 0.8rem, h3: 0.75rem) to keep content compact within dark-mode panels. The agent panel h1 gets an accent-colored bottom border as a visual separator.

**Controls row**: Section 1 (NL + params + metrics) uses `d-flex` columns with `.card { height: 100%; display: flex; flex-direction: column }` and `.metric-card { height: 100%; overflow: hidden }` for equal-height alignment. Metric cards use `min-width: 0` and `overflow: hidden` to prevent text overflow in narrow columns.

**Layout model**: `.app-layout` uses `width: 100vw; height: 100vh; display: flex` (no `position: fixed` to avoid Dash wrapper conflicts). Dash wrappers (`#react-entry-point`, `#_dash-app-content`) are explicitly set to `width: 100%; height: 100%`. Dashboard uses `flex: 1 1 0% !important; min-width: 0 !important` — the `!important` flags prevent Bootstrap DARKLY theme overrides. Chat sidebar uses `flex: 0 0 300px !important; width: 300px !important; max-width: 300px` — rigid 300px that never grows or shrinks. `_compute_position_styles(pos)` in callbacks.py returns inline style dicts: left/right only set `order` + borders (never `flex` or `width`); top/bottom override sidebar to `width: 100%; flex: 0 0 25vh`. Position toggle buttons (`.chat-pos-btn`, `.chat-pos-btn-active`) are 22x22px compact buttons with accent highlight on active state. Responsive: `@media (max-width: 1200px)` stacks vertically with `!important` overrides.

## Development Commands

```bash
python app.py                           # Start dashboard
python -c "from engine import *"        # Verify engine (incl. ensemble, sentiment)
python -c "from agents import *"        # Verify agents (incl. PortfolioBuilderAgent)
python -c "from config import get_ticker_name; print(get_ticker_name('AMX'))"  # Test ticker name
```

## Next Steps

### Chatbot v2 — Tools y Modificaciones
Ampliar el chatbot (actualmente conversacional puro, v1) con herramientas para:
- **Modificaciones via chat**: "Sube ACWI a 10%", "Agrega BTC-USD al portafolio" — interpretar y ejecutar cambios en sliders/tickers
- **Analisis comparativo**: "Compara mi portafolio con el S&P 500", "Muestra el drawdown si hubiera invertido hace 5 anos"
- **Re-ejecutar pipeline**: tool para re-correr MC + ensemble desde el chat

## Reference Materials

- `referencias/TradingAgents/` — Multi-agent LLM framework (LangGraph patterns)
- `referencias/img/` — 7 dashboard design reference screenshots
- `referencias/prompt_inicial.txt` — Original project requirements
