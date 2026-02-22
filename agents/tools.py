"""Funciones @tool de LangChain para el agente ReAct Constructor de Portafolios.

Cada tool es un wrapper delgado alrededor de funciones existentes del engine.
"""

import json
import numpy as np
import yfinance as yf
from langchain_core.tools import tool

from engine.data import fetch_stock_data, compute_returns, get_annual_stats
from engine.monte_carlo import run_monte_carlo, get_optimal_portfolio
from engine.optimizer import optimize_max_sharpe
from engine.risk import calc_portfolio_metrics

# ── Curated sector/theme → ticker mapping ────────────────────────────
SECTOR_TICKERS = {
    "technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM", "ADBE", "INTC", "AMD", "ORCL"],
    "tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM", "ADBE", "INTC", "AMD", "ORCL"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "BMY", "AMGN"],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SCHW", "USB"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SCHW", "USB"],
    "consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "WMT", "LOW"],
    "industrial": ["CAT", "BA", "HON", "UPS", "GE", "MMM", "RTX", "LMT", "DE", "UNP"],
    "industrials": ["CAT", "BA", "HON", "UPS", "GE", "MMM", "RTX", "LMT", "DE", "UNP"],
    "real_estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "DLR", "PSA", "WELL", "AVB"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
    "telecom": ["T", "VZ", "TMUS", "CMCSA", "CHTR"],
    "semiconductor": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC"],
    "ai": ["NVDA", "MSFT", "GOOGL", "META", "AMD", "PLTR", "CRM", "SNOW", "AI", "SMCI"],
    "ev": ["TSLA", "RIVN", "LCID", "NIO", "LI", "XPEV", "F", "GM"],
    "cloud": ["AMZN", "MSFT", "GOOGL", "CRM", "SNOW", "NET", "DDOG", "MDB", "ZS", "CRWD"],
    "renewable": ["ENPH", "SEDG", "FSLR", "NEE", "BEP", "RUN", "CSIQ", "DQ"],
    "crypto": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "CIFR"],
    "cryptocurrency": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "CIFR"],
    "bitcoin": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF"],
    "blockchain": ["COIN", "MSTR", "MARA", "RIOT", "IBM", "SQ", "PYPL"],
    "fintech": ["SQ", "PYPL", "AFRM", "SOFI", "NU", "COIN", "HOOD", "UPST"],
    "magnificent7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "defense": ["LMT", "RTX", "NOC", "GD", "BA", "HII", "LHX", "TDG"],
    "biotech": ["AMGN", "GILD", "REGN", "VRTX", "BIIB", "MRNA", "ILMN", "ALNY"],
    "retail": ["WMT", "COST", "TGT", "HD", "LOW", "AMZN", "TJX", "ROST"],
    "gaming": ["NVDA", "AMD", "MSFT", "SONY", "EA", "TTWO", "RBLX", "U"],

    # ── Geographic / Country mappings ──────────────────────────────
    "mexico": ["AMX", "FMX", "KOF", "CX", "BIMBOA.MX", "WALMEX.MX", "GFNORTEO.MX", "GMEXICOB.MX", "PAC", "ASR"],
    "mexicanas": ["AMX", "FMX", "KOF", "CX", "BIMBOA.MX", "WALMEX.MX", "GFNORTEO.MX", "GMEXICOB.MX", "PAC", "ASR"],
    "mexico_esr": ["AMX", "FMX", "KOF", "CX", "BIMBOA.MX", "WALMEX.MX", "GFNORTEO.MX", "GMEXICOB.MX", "AC.MX", "GRUMAB.MX"],
    "mexico_ampliado": ["AMX", "FMX", "KOF", "CX", "PAC", "ASR", "OMAB", "VLRS", "BIMBOA.MX", "WALMEX.MX",
                         "GFNORTEO.MX", "GMEXICOB.MX", "AC.MX", "GCARSOA1.MX", "ALSEA.MX", "ELEKTRA.MX",
                         "GRUMAB.MX", "ORBIA.MX", "PINFRA.MX", "LABB.MX"],
    "bmv": ["AMX", "FMX", "KOF", "CX", "PAC", "ASR", "OMAB", "VLRS",
            "BIMBOA.MX", "WALMEX.MX", "GFNORTEO.MX", "GMEXICOB.MX", "AC.MX",
            "GCARSOA1.MX", "ALSEA.MX", "ELEKTRA.MX", "GRUMAB.MX", "ORBIA.MX",
            "PINFRA.MX", "LABB.MX", "FEMSAUBD.MX", "AMXB.MX", "CEMEXCPO.MX",
            "KOFUBL.MX", "GAPB.MX", "ASURB.MX", "OMAB.MX", "MEGACPO.MX",
            "BBAJIOO.MX", "GENTERA.MX", "BOLSAA.MX", "AGUA.MX", "VESTA.MX",
            "FIBRAMQ12.MX", "KIMBERA.MX", "ALPEKA.MX", "LIVEPOL1.MX",
            "RA.MX", "SPORTS.MX", "VOLARA.MX", "Q.MX", "HOTEL.MX",
            "CMOCTEZ.MX", "TRAXIONA.MX", "CIDMEGA.MX"],
    "bolsa_mexicana": ["AMX", "FMX", "KOF", "CX", "PAC", "ASR", "OMAB", "VLRS",
                        "BIMBOA.MX", "WALMEX.MX", "GFNORTEO.MX", "GMEXICOB.MX", "AC.MX",
                        "GCARSOA1.MX", "ALSEA.MX", "ELEKTRA.MX", "GRUMAB.MX", "ORBIA.MX",
                        "PINFRA.MX", "LABB.MX", "FEMSAUBD.MX", "AMXB.MX", "CEMEXCPO.MX",
                        "KOFUBL.MX", "GAPB.MX", "ASURB.MX", "OMAB.MX", "MEGACPO.MX",
                        "BBAJIOO.MX", "GENTERA.MX", "BOLSAA.MX", "AGUA.MX", "VESTA.MX",
                        "FIBRAMQ12.MX", "KIMBERA.MX", "ALPEKA.MX", "LIVEPOL1.MX",
                        "RA.MX", "SPORTS.MX", "VOLARA.MX", "Q.MX", "HOTEL.MX",
                        "CMOCTEZ.MX", "TRAXIONA.MX", "CIDMEGA.MX"],
    "brazil": ["VALE", "PBR", "ITUB", "BBD", "NU", "ABEV", "SBS", "BSBR", "MELI", "EWZ"],
    "brasil": ["VALE", "PBR", "ITUB", "BBD", "NU", "ABEV", "SBS", "BSBR", "MELI", "EWZ"],
    "latam": ["AMX", "FMX", "MELI", "NU", "VALE", "PBR", "ITUB", "KOF", "CX", "SQM"],
    "latin_america": ["AMX", "FMX", "MELI", "NU", "VALE", "PBR", "ITUB", "KOF", "CX", "SQM"],
    "latinoamerica": ["AMX", "FMX", "MELI", "NU", "VALE", "PBR", "ITUB", "KOF", "CX", "SQM"],
    "chile": ["SQM", "BSAC", "ECL.SN"],
    "colombia": ["EC", "CIB", "AVAL"],
    "argentina": ["GGAL", "YPF", "MELI", "PAM", "TEO", "LOMA", "BMA", "SUPV"],

    # ── ESG / Sustainability ───────────────────────────────────────
    "esg": ["ESGU", "SUSA", "DSI", "KRMA", "SNPE", "AAPL", "MSFT", "CRM", "ADBE", "NEE"],
    "socialmente_responsable": ["ESGU", "SUSA", "DSI", "KRMA", "SNPE", "AAPL", "MSFT", "CRM", "ADBE", "NEE"],
    "sustainable": ["NEE", "ENPH", "FSLR", "BEP", "ESGU", "SUSA", "DSI", "KRMA", "VEGN", "SNPE"],
    "sustentable": ["NEE", "ENPH", "FSLR", "BEP", "ESGU", "SUSA", "DSI", "KRMA", "VEGN", "SNPE"],

    # ── Criptomonedas (pares USD) ─────────────────────────────────
    "cripto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
               "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "ATOM-USD",
               "LTC-USD", "NEAR-USD"],
    "criptomonedas": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
                       "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "ATOM-USD",
                       "LTC-USD", "NEAR-USD"],
    "crypto_directo": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
                        "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "ATOM-USD",
                        "LTC-USD", "NEAR-USD"],

    # ── Indices bursatiles ────────────────────────────────────────
    "indices": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^N225", "^HSI", "^MXX", "^BVSP", "^STOXX50E"],
    "indices_usa": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
    "indices_global": ["^GSPC", "^FTSE", "^N225", "^HSI", "^STOXX50E", "^BVSP", "^MXX"],
    "ipc": ["^MXX"],
    "sp500": ["^GSPC"],
    "nasdaq": ["^IXIC"],
    "dow_jones": ["^DJI"],

    # ── Futuros (commodities + indices) ───────────────────────────
    "futuros": ["GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZW=F", "ES=F", "NQ=F", "YM=F"],
    "futures": ["GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZW=F", "ES=F", "NQ=F", "YM=F"],
    "commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZW=F"],
    "materias_primas": ["GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZW=F"],
    "oro": ["GC=F"],
    "gold": ["GC=F"],
    "petroleo": ["CL=F"],
    "oil": ["CL=F"],

    # ── ETFs — Mercado amplio ─────────────────────────────────────
    "etf": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VT", "ACWI"],
    "etfs": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VT", "ACWI"],
    "etf_mercado": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VT", "ACWI"],

    # ── ETFs — Sector ─────────────────────────────────────────────
    "etf_tech": ["XLK", "VGT", "FTEC", "IGV"],
    "etf_salud": ["XLV", "VHT", "IBB", "XBI"],
    "etf_financiero": ["XLF", "VFH", "KBE", "KRE"],
    "etf_energia": ["XLE", "VDE", "OIH", "AMLP"],
    "etf_industrial": ["XLI", "VIS"],
    "etf_consumo": ["XLY", "XLP", "VCR", "VDC"],
    "etf_inmobiliario": ["VNQ", "IYR", "XLRE", "RWR"],

    # ── ETFs — Tematicos ──────────────────────────────────────────
    "etf_ark": ["ARKK", "ARKW", "ARKG", "ARKF", "ARKQ"],
    "etf_innovacion": ["ARKK", "ARKW", "ARKG", "ARKF", "ARKQ"],
    "etf_dividendos": ["VYM", "SCHD", "HDV", "DVY", "DGRO"],
    "etf_bonos": ["BND", "AGG", "TLT", "IEF", "SHY", "TIPS", "HYG", "LQD"],
    "etf_renta_fija": ["BND", "AGG", "TLT", "IEF", "SHY", "TIPS", "HYG", "LQD"],

    # ── ETFs — Commodities ────────────────────────────────────────
    "etf_oro": ["GLD", "IAU", "SGOL"],
    "etf_plata": ["SLV", "SIVR"],
    "etf_commodities": ["GLD", "IAU", "SLV", "USO", "DBC", "GSG", "PDBC"],

    # ── ETFs — Paises / Regiones ──────────────────────────────────
    "etf_mexico": ["EWW", "NAFTRAC.MX"],
    "naftrac": ["NAFTRAC.MX"],
    "etf_brasil": ["EWZ"],
    "etf_china": ["FXI", "MCHI", "KWEB", "GXC"],
    "etf_emergentes": ["EEM", "VWO", "IEMG"],
    "etf_europa": ["VGK", "EZU", "HEDJ"],
    "etf_japon": ["EWJ", "DXJ"],
    "etf_global": ["VT", "ACWI", "IXUS", "VEA", "VWO"],
}


@tool
def validate_tickers(tickers: list[str]) -> str:
    """Valida que cada simbolo de ticker exista en Yahoo Finance.

    Retorna un objeto JSON con listas de tickers 'valid' e 'invalid'.
    Usa esto para verificar tickers antes de descargar datos u optimizar.
    """
    valid, invalid = [], []
    for t in tickers:
        t = t.upper().strip()
        try:
            info = yf.Ticker(t).info
            if info and info.get("regularMarketPrice") is not None:
                valid.append(t)
            else:
                invalid.append(t)
        except Exception:
            invalid.append(t)
    return json.dumps({"valid": valid, "invalid": invalid})


@tool
def search_tickers_by_sector(sector: str) -> str:
    """Busca los principales simbolos de ticker en un sector, pais, region o tema.

    Soporta busqueda por:
    - Sector: technology, energy, healthcare, finance, consumer, industrial,
      real_estate, utilities, telecom, semiconductor, ai, ev, cloud, renewable,
      crypto (acciones cripto), fintech, defense, biotech, retail, gaming.
    - Pais/Region: mexico, mexicanas, mexico_esr, mexico_ampliado, bmv,
      bolsa_mexicana, brazil, brasil, latam, latinoamerica, chile, colombia, argentina.
    - Tema: esg, socialmente_responsable, sustainable, sustentable,
      magnificent7, mag7.
    - Criptomonedas (pares USD): cripto, criptomonedas, crypto_directo.
    - Indices bursatiles: indices, indices_usa, indices_global, ipc, sp500, nasdaq, dow_jones.
    - Futuros: futuros, futures, commodities, materias_primas, oro, gold, petroleo, oil.
    - ETFs mercado: etf, etfs, etf_mercado.
    - ETFs sector: etf_tech, etf_salud, etf_financiero, etf_energia, etf_industrial,
      etf_consumo, etf_inmobiliario.
    - ETFs tematicos: etf_ark, etf_innovacion, etf_dividendos, etf_bonos, etf_renta_fija.
    - ETFs commodities: etf_oro, etf_plata, etf_commodities.
    - ETFs pais/region: etf_mexico, naftrac, etf_brasil, etf_china, etf_emergentes,
      etf_europa, etf_japon, etf_global.

    Retorna una lista de simbolos de ticker para la busqueda dada.
    Usa espacios o guiones bajos indistintamente.
    """
    key = sector.lower().strip().replace(" ", "_")
    tickers = SECTOR_TICKERS.get(key)
    if tickers:
        return json.dumps({"sector": sector, "tickers": tickers})
    available = sorted(SECTOR_TICKERS.keys())
    return json.dumps({
        "error": f"Sector '{sector}' no encontrado",
        "available_sectors": available,
    })


@tool
def fetch_and_analyze(tickers: list[str]) -> str:
    """Descarga datos historicos de precios y retorna estadisticas resumen.

    Retorna retorno anual medio, volatilidad anualizada y correlaciones
    entre pares para cada ticker. Usa esto para evaluar candidatos antes de optimizar.
    """
    tickers = [t.upper().strip() for t in tickers]
    try:
        prices = fetch_stock_data(tickers=tickers)
    except Exception as e:
        return json.dumps({"error": f"Error descargando datos: {e}"})

    returns = compute_returns(prices)
    mean_ret, cov_mat = get_annual_stats(returns)

    vols = np.sqrt(np.diag(cov_mat))
    corr = np.corrcoef(returns.T)

    available = list(prices.columns)
    stats = {}
    for i, t in enumerate(available):
        stats[t] = {
            "annual_return": f"{mean_ret[i]:.2%}",
            "annual_volatility": f"{vols[i]:.2%}",
        }

    corr_summary = {}
    for i, t1 in enumerate(available):
        for j, t2 in enumerate(available):
            if j > i:
                corr_summary[f"{t1}-{t2}"] = f"{corr[i, j]:.3f}"

    return json.dumps({
        "tickers": available,
        "stats": stats,
        "correlations": corr_summary,
    })


@tool
def run_optimization(tickers: list[str], rf: float = 0.04, num_sims: int = 10000) -> str:
    """Ejecuta el pipeline completo de optimizacion Monte Carlo + SLSQP.

    Retorna los pesos optimos del portafolio, retorno esperado, volatilidad,
    Sharpe Ratio y VaR. Esta es la funcion principal de optimizacion.

    Args:
        tickers: Lista de simbolos de ticker (ej: ["AAPL", "MSFT", "GOOGL"])
        rf: Tasa libre de riesgo anual como decimal (default 0.04 = 4%)
        num_sims: Numero de simulaciones Monte Carlo (default 10000)
    """
    tickers = [t.upper().strip() for t in tickers]
    try:
        prices = fetch_stock_data(tickers=tickers)
    except Exception as e:
        return json.dumps({"error": f"Error descargando datos: {e}"})

    available = list(prices.columns)
    returns = compute_returns(prices)
    mean_ret, cov_mat = get_annual_stats(returns)

    mc = run_monte_carlo(mean_ret, cov_mat, num_sims=num_sims, risk_free_rate=rf)
    opt_weights = optimize_max_sharpe(mean_ret, cov_mat, risk_free_rate=rf)
    metrics = calc_portfolio_metrics(opt_weights, mean_ret, cov_mat, rf)

    weights_dict = {t: round(float(w), 4) for t, w in zip(available, opt_weights)}

    return json.dumps({
        "tickers": available,
        "optimal_weights": weights_dict,
        "expected_return": f"{metrics['expected_return']:.2%}",
        "volatility": f"{metrics['volatility']:.2%}",
        "sharpe_ratio": f"{metrics['sharpe_ratio']:.3f}",
        "var_95": f"{metrics['var']:.2%}",
        "num_simulations": num_sims,
        "risk_free_rate": f"{rf:.2%}",
    })


@tool
def get_portfolio_metrics(
    tickers: list[str],
    weights: list[float],
    rf: float = 0.04,
) -> str:
    """Calcula las metricas del portafolio para un conjunto especifico de pesos.

    Args:
        tickers: Lista de simbolos de ticker
        weights: Lista de pesos del portafolio (deben sumar 1.0)
        rf: Tasa libre de riesgo anual como decimal
    """
    tickers = [t.upper().strip() for t in tickers]
    try:
        prices = fetch_stock_data(tickers=tickers)
    except Exception as e:
        return json.dumps({"error": f"Error descargando datos: {e}"})

    available = list(prices.columns)
    returns_df = compute_returns(prices)
    mean_ret, cov_mat = get_annual_stats(returns_df)

    w = np.array(weights[:len(available)], dtype=float)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum

    metrics = calc_portfolio_metrics(w, mean_ret, cov_mat, rf)

    return json.dumps({
        "tickers": available,
        "weights": {t: round(float(wi), 4) for t, wi in zip(available, w)},
        "expected_return": f"{metrics['expected_return']:.2%}",
        "volatility": f"{metrics['volatility']:.2%}",
        "sharpe_ratio": f"{metrics['sharpe_ratio']:.3f}",
        "var_95": f"{metrics['var']:.2%}",
    })


@tool
def run_ensemble_optimization(tickers: list[str], rf: float = 0.04) -> str:
    """Ejecuta optimizacion ensemble con 7 metodos y retorna comparacion.

    Ejecuta: Max Sharpe, Min Varianza, Paridad de Riesgo, Max Diversificacion,
    HRP, Min CVaR y Pesos Iguales. Luego calcula promedios ensemble.

    Args:
        tickers: Lista de simbolos de ticker
        rf: Tasa libre de riesgo anual como decimal (default 0.04)
    """
    from engine.ensemble import run_all_methods, ensemble_vote

    tickers = [t.upper().strip() for t in tickers]
    try:
        prices = fetch_stock_data(tickers=tickers)
    except Exception as e:
        return json.dumps({"error": f"Error descargando datos: {e}"})

    available = list(prices.columns)
    returns_df = compute_returns(prices)
    mean_ret, cov_mat = get_annual_stats(returns_df)

    all_methods = run_all_methods(mean_ret, cov_mat, rf)
    ensembles = ensemble_vote(all_methods, mean_ret, cov_mat, rf)

    combined = {**all_methods, **ensembles}

    summary = {}
    for key, data in combined.items():
        m = data.get("metrics", {})
        summary[key] = {
            "nombre": data.get("nombre", key),
            "tickers": available,
            "weights": {t: round(w, 4) for t, w in zip(available, data.get("weights", []))},
            "expected_return": f"{m.get('expected_return', 0):.2%}",
            "volatility": f"{m.get('volatility', 0):.2%}",
            "sharpe_ratio": f"{m.get('sharpe_ratio', 0):.3f}",
            "var_95": f"{m.get('var', 0):.2%}",
        }

    return json.dumps(summary)


ALL_TOOLS = [
    validate_tickers,
    search_tickers_by_sector,
    fetch_and_analyze,
    run_optimization,
    get_portfolio_metrics,
    run_ensemble_optimization,
]
