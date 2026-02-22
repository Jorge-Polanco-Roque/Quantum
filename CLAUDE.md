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
│   │                               #   _clip_to_bounds, run_all_methods, ensemble_vote, ensemble_shrinkage
│   │                               #   All methods accept optional bounds=[(lo,hi),...] for constraints
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
│   │                               #   SECTOR_TICKERS: ~308 categories, ~1004 unique tickers
│   │                               #   TICKER_ALIASES: short fund names → Yahoo Finance .MX tickers
│   │                               #   Covers: sectors, countries, cryptos, indices, futures, ETFs, UCITS, SIC
│   │                               #   ALL_TOOLS (6): full set for chatbot v2 / other consumers
│   │                               #   BUILDER_TOOLS (3): validate, search, analyze (no optimization)
│   ├── portfolio_builder.py        # PortfolioBuilderAgent (deterministic: selects tickers + method only)
│   │                               #   LLM returns {tickers, method, reasoning} — NO weights
│   │                               #   Exception: method="preset" includes user-specified weights
│   │                               #   REGLA #2: rebalanceo usa SOLO tickers dados, nunca agrega extras
│   │                               #   REGLA #3: "0% en X" = exclude ticker, never max=0 constraint
│   │                               #   One-shot examples: ideal 4-call flow (search→validate→analyze→JSON)
│   │                               #   Split priority: group proportions override optimize method
│   │                               #   Uses BUILDER_TOOLS (3) — no access to optimization tools
│   │                               #   Weight computation happens in dashboard callback
│   └── chatbot.py                  # ChatbotAgent (conversational, InMemorySaver, portfolio context)
│                                   #   _format_context: full ensemble weights per ticker + metrics
├── dashboard/                      # Dash UI (dark mode, Plotly charts, all in Spanish)
│   ├── layout.py                   # Adaptive layout: dashboard (3/4) + chat sidebar (1/4)
│   │                               #   Chat dynamically repositionable: right/left/top/bottom
│   │                               #   IDs: app-layout, dashboard-main (for inline style switching)
│   │                               #   Section 1: Combined card (.controls-combined) merges
│   │                               #     NL input (md=5) + Parameters (md=7) in single card,
│   │                               #     outer split md=8 combined / md=4 metrics
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
│   │                               #   _build_bounds_from_constraints(tickers, constraints) → SLSQP bounds
│   │                               #   _apply_weight_floor(weight_map, min_floor, constraints) → floor+constraints
│   │                               #   _build_sentiment_output() → shared by manual + auto sentiment
│   ├── components/
│   │   ├── parameters.py           # Tasa Libre%, Simulaciones, VaR Nivel% inputs
│   │   │                           #   create_parameters_content() (no card wrapper, for combined card)
│   │   │                           #   create_parameters_panel() (standalone card wrapper)
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
│   │   │                           #   create_nl_input_content() (no card wrapper, for combined card)
│   │   │                           #   create_nl_input_panel() (standalone card wrapper)
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
5. **CONSTRUIR PORTAFOLIO** → `agents/portfolio_builder.py` runs ReAct agent with BUILDER_TOOLS (3): interprets NL prompt → selects tickers → validates → returns `{tickers, method, reasoning}` (NO weights, except `"preset"`). Optionally returns `"constraints"` with per-ticker min/max bounds (e.g. `{"NVDA": {"min": 0.10}, "_all": {"max": 0.25}}`). Callback computes weights deterministicaly: `method="preset"` → user-specified percentages passed through, `"ensemble"` → ensemble shrinkage (default), `"optimize"` → SLSQP, `"equal_weight"` → 1/N, `"risk_parity"` → engine risk parity, `"min_variance"` → engine min variance, `"split"` → `_compute_split_weights()`. Constraints are converted to SLSQP bounds via `_build_bounds_from_constraints()` and passed to all optimizers + ensemble. Then `_run_full_pipeline(preset_weights=weight_map, method=method, constraints=constraints)` → `_apply_weight_floor(constraints=constraints)` ensures min 2% per ticker AND respects user constraints → updates `store-tickers` + runs full pipeline (MC + ensemble + all charts) → **auto-triggers AI Analysis** (see #4) + **auto-triggers Sentiment** (see #3). Same prompt always produces identical weights.
6. **Chatbot** → always-visible sidebar (1/4, repositionable via toggle buttons) → user types question → callback reads all portfolio stores → builds context dict → `agents/chatbot.py` ChatbotAgent.chat() with InMemorySaver for multi-turn memory → returns markdown response rendered in chat bubble.
7. **Chat position toggle** → click ◀▲▼▶ buttons → `switch_chat_position` callback updates inline styles on `#app-layout`, `#dashboard-main`, `#chat-sidebar` via `_compute_position_styles(pos)` → CSS `flex-direction` + `order` switch without DOM rebuild → chat history and all stores preserved.

## Ensemble Optimization (engine/ensemble.py)

7 methods with unified signature `(mean_returns, cov_matrix, rf, bounds=None) -> np.ndarray`:
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
When the user provides a specific ticker list or portfolio, the agent uses ONLY those tickers
— never adds extras. For rebalancing requests, it keeps the same tickers (minus exclusions).
Returns `{tickers, method, reasoning}` (and `split` when method="split", `constraints` when
user specifies min/max per ticker). Weight computation happens deterministically in the
dashboard callback via `_compute_split_weights()`, engine optimizers (`risk_parity_portfolio`,
`min_variance_portfolio`, `optimize_max_sharpe`), or equal weight. User constraints are
converted to SLSQP bounds via `_build_bounds_from_constraints()` and forwarded to all
optimizers + ensemble methods. `_apply_weight_floor(constraints=constraints)` ensures every
ticker gets >= 2% AND respects user min/max constraints. The `"_all"` key applies global
bounds to all tickers; individual ticker constraints override `"_all"`.

Methods: `"preset"` (user-specified exact percentages per ticker), `"ensemble"` (ensemble shrinkage with adaptive delta, default),
`"optimize"` (SLSQP Max Sharpe), `"equal_weight"` (1/N), `"risk_parity"`, `"min_variance"`, `"split"` (with groups specification).

Constraints: Any method can include `"constraints"` with per-ticker bounds: `{"AAPL": {"min": 0.05, "max": 0.30}, "_all": {"max": 0.25}}`.
The system distinguishes between preset (exact weights), constraints (bounds for optimization), and exclusions ("0% en X" = ticker omitted from list entirely).
Constraints are enforced at both the optimizer level (SLSQP bounds) and post-processing (`_apply_weight_floor`).

Constraint normalization (callbacks.py, safety net): after parsing the agent's JSON, any ticker with `max<=0` or a lone `{"min": 0}` is removed from the ticker list (treated as exclusion). This guarantees deterministic results even if the LLM parses "0% en X" inconsistently between `{"max": 0}` and `{"min": 0}`.

### Chatbot Interactivo (agents/chatbot.py)
Conversational agent using `create_react_agent` + `InMemorySaver` for multi-turn memory. Receives full portfolio context (tickers, weights, metrics, ensemble with full per-ticker weights and metrics per method, sentiment) injected as `[CONTEXTO DEL PORTAFOLIO]` block on each message. System prompt in Spanish. No tools (v1 — pure conversational with context). Thread ID (UUID) generated per page load.

Dashboard integration: always-visible sidebar (`chat_widget.py`, 300px fixed, dynamically repositionable) with 3 callbacks — init (generate thread_id), send (build context from stores → ChatbotAgent.chat() → render markdown bubbles), position toggle (switch_chat_position). Layout uses CSS flexbox: `.app-layout` (100vw × 100vh flex row) → `.dashboard-main` (flex: 1 1 0%, fills remaining space, min-width: 0) + `.chat-sidebar` (flex: 0 0 300px, rigid). Dash internal wrappers (`#react-entry-point`, `#_dash-app-content`) are explicitly sized to 100% to prevent layout interference. Position toggle buttons (◀▲▼▶) in chat header switch between right/left/top/bottom by changing inline styles on 3 container IDs (`app-layout`, `dashboard-main`, `chat-sidebar`) — no DOM rebuild, all state preserved. `_compute_position_styles(pos)` returns the 3 style dicts per position. Left/Right: only order + border changes (no flex overrides). Top/Bottom: flex-direction column + sidebar width: 100% + height: 25vh. Responsive: stacks vertically below 1200px.

### SECTOR_TICKERS Asset Coverage (agents/tools.py)
~308 curated mappings, ~1004 unique tickers:
- **Sectors** (expanded to 15-20 tickers each): technology, energy, healthcare, finance, consumer, industrial, real_estate, utilities, telecom, semiconductor, ai, ev, cloud, renewable, crypto (stocks), fintech, defense, biotech, retail, gaming, software, materials, pharma, medtech, insurance, media, cybersecurity, food_beverage, consumer_staples, transportation, agriculture, water, cannabis, luxury, communication_services, aerospace, hospitality, payments, logistics, reits, data_center, healthcare_services, diagnostics, waste_management, construction, chemicals, steel, mining, airlines, automotive, education, pet_economy, sports_betting, hydrogen, nuclear, space, internet, ecommerce, 3d_printing, quantum_computing
- **Countries/Regions** (30+): mexico, mexicanas, mexico_esr, mexico_ampliado, bmv/bolsa_mexicana (45 tickers), brazil/brasil, latam/latinoamerica, chile, colombia, argentina, india, uk/reino_unido, europe/europa_acciones, germany/alemania, canada, australia, japan/japon_acciones, china/china_adr, korea/corea, southeast_asia, spain/espana, africa_stocks, taiwan, israel
- **Themes**: esg, socialmente_responsable, sustainable/sustentable, magnificent7/mag7
- **Criptomonedas**: cripto/criptomonedas/crypto_directo — 27 pares USD (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, DOT, MATIC, LINK, ATOM, LTC, NEAR, UNI, AAVE, FTM, ALGO, MANA, SAND, etc.)
- **Indices** (19+): indices, indices_usa, indices_global, ipc, sp500, nasdaq, dow_jones, dax, cac40, ibex, nifty, kospi, tsx, merval + more
- **Futuros** (15+): futuros/futures, commodities/materias_primas, oro/gold, petroleo/oil, cobre/copper, platino, cafe/coffee, algodon/cotton, azucar/sugar
- **ETFs mercado**: etf/etfs/etf_mercado (SPY, QQQ, IWM, etc.)
- **ETFs sector**: etf_tech, etf_salud, etf_financiero, etf_energia, etf_industrial, etf_consumo, etf_inmobiliario, etf_materials, etf_utilities, etf_communication, etf_real_estate, etf_healthcare, etf_defense
- **ETFs size/style**: etf_small_cap, etf_mid_cap, etf_value, etf_growth, etf_momentum, etf_low_vol
- **ETFs tematicos**: etf_ark, etf_esg, etf_semiconductor, etf_cybersecurity, etf_clean_energy, etf_water, etf_infrastructure, etf_ai, etf_robotics, etf_cloud, etf_fintech, etf_bitcoin, etf_ethereum, etf_cannabis, etf_space, etf_gaming, etf_luxury, etf_agriculture, etf_dividendos, etf_dividendos_global
- **ETFs renta fija**: etf_bonos, etf_treasury, etf_high_yield, etf_tips, etf_corporate, etf_muni, etf_emerging_debt
- **ETFs commodities**: etf_oro, etf_plata, etf_commodities
- **ETFs pais/region**: etf_mexico, etf_brasil, etf_china, etf_emergentes, etf_europa, etf_japon, etf_global, etf_india, etf_korea, etf_canada, etf_australia, etf_uk, etf_germany, etf_southeast_asia, etf_africa, etf_middle_east, etf_frontier, etf_latam
- **UCITS** (15+ categories): ucits/ucits_equity (15 tickers), ucits_bonos (8), ucits_esg, ucits_small_cap, ucits_emerging, ucits_usa, ucits_europe, ucits_japan, ucits_pacific, ucits_dividendos, ucits_oro, ucits_commodities
- **SIC Mexico**: sic/sic_mexico — ~31 ETFs extranjeros disponibles en el Sistema Internacional de Cotizaciones
- **Fondos Mexico**: fondos_mexico/fondos_inversion, fondos_blackrock/blackrock_mexico, fondos_gbm

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
- `AGENT_RECURSION_LIMIT`: ReAct agent max turns (80)
- `ENSEMBLE_DELTA_MIN` / `ENSEMBLE_DELTA_MAX`: Shrinkage delta range (0.3 / 0.7)

## CSS Scoped Styles (dashboard/assets/style.css)

Markdown rendered inside `.agent-panel`, `#sentiment-output`, and `.chat-bubble-assistant` has scoped heading sizes (h1: 0.85rem, h2: 0.8rem, h3: 0.75rem) to keep content compact within dark-mode panels. The agent panel h1 gets an accent-colored bottom border as a visual separator.

**Controls row**: Section 1 uses a combined card (`.controls-combined`) that merges NL Builder (left, `md=5`) and Parameters (right, `md=7`) into a single card with a vertical divider (`.controls-left` border-right). Outer split: `md=8` combined card / `md=4` metric cards. Cards use `gap: 8px` instead of `justify-content: space-between`. `.dash-section > .row { align-items: stretch }` ensures equal-height columns. `.metric-card { height: 100%; overflow: hidden }` with `min-width: 0` prevents text overflow. Buttons use `margin-top: auto` for bottom-alignment within flex columns.

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
