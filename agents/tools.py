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

# ── Alias mapping: common short names → Yahoo Finance tickers ────────
# Mexican fondos de inversión use series-specific tickers on Yahoo Finance.
# Users type short names (e.g. "BLKEM1") but yfinance needs the full series.
TICKER_ALIASES = {
    # BlackRock Mexico fondos
    "BLKEM1": "BLKEM1C0-A.MX",
    "GOLD5+": "GOLD5+B2-C.MX",
    "BLKGUB1": "BLKGUB1B0-D.MX",
    "BLKINT": "BLKINTC0-A.MX",
    "BLKDOL": "BLKDOLB0-D.MX",
    "BLKLIQ": "BLKLIQB0-D.MX",
    "BLKRFA": "BLKRFAB0-C.MX",
    # GBM fondos
    "GBMCRE": "GBMCREB2-A.MX",
    "GBMMOD": "GBMMODB2-A.MX",
    "GBMF2": "GBMF2BO.MX",
    "GBMINT": "GBMINTB1.MX",
    # NAFTRAC alias
    "NAFTRAC": "NAFTRAC.MX",
}


def resolve_ticker_aliases(tickers: list[str]) -> list[str]:
    """Replace short fund names with their Yahoo Finance equivalents."""
    resolved = []
    for t in tickers:
        clean = t.strip()
        upper = clean.upper()
        # Check exact match first, then uppercase
        alias = TICKER_ALIASES.get(clean) or TICKER_ALIASES.get(upper)
        resolved.append(alias if alias else clean)
    return resolved


# ── Curated sector/theme → ticker mapping ────────────────────────────
SECTOR_TICKERS = {
    # ── Sectors (expanded) ───────────────────────────────────────────
    "technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM", "ADBE", "INTC", "AMD", "ORCL",
                   "IBM", "NOW", "INTU", "SHOP", "WDAY", "TEAM", "HUBS", "ZM", "PANW", "FTNT"],
    "tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM", "ADBE", "INTC", "AMD", "ORCL",
             "IBM", "NOW", "INTU", "SHOP", "WDAY", "TEAM", "HUBS", "ZM", "PANW", "FTNT"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
               "DVN", "FANG", "HES", "PXD", "BKR", "CTRA", "MRO", "APA", "TRGP", "OKE"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "BMY", "AMGN",
                   "CI", "HUM", "ISRG", "SYK", "MDT", "BSX", "EW", "DXCM", "A", "DHR"],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SCHW", "USB",
                "PNC", "TFC", "COF", "DFS", "ICE", "CME", "MCO", "SPGI", "MSCI", "FIS"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SCHW", "USB",
                   "PNC", "TFC", "COF", "DFS", "ICE", "CME", "MCO", "SPGI", "MSCI", "FIS"],
    "consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "WMT", "LOW",
                 "DG", "DLTR", "ORLY", "AZO", "LULU", "BKNG", "MAR", "HLT", "CMG", "YUM"],
    "industrial": ["CAT", "BA", "HON", "UPS", "GE", "MMM", "RTX", "LMT", "DE", "UNP",
                   "FDX", "NSC", "CSX", "WM", "RSG", "EMR", "ROK", "ETN", "ITW", "PH"],
    "industrials": ["CAT", "BA", "HON", "UPS", "GE", "MMM", "RTX", "LMT", "DE", "UNP",
                    "FDX", "NSC", "CSX", "WM", "RSG", "EMR", "ROK", "ETN", "ITW", "PH"],
    "real_estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "DLR", "PSA", "WELL", "AVB",
                    "VTR", "ARE", "MAA", "UDR", "ESS", "SUI", "CPT", "INVH", "CUBE", "EXR"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC",
                  "ES", "AWK", "ATO", "CMS", "LNT", "EVRG", "PNW", "NI"],
    "telecom": ["T", "VZ", "TMUS", "CMCSA", "CHTR", "LUMN", "FTR", "USM"],
    "semiconductor": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC",
                      "MRVL", "NXPI", "ON", "SWKS", "ASML", "TSM", "MCHP", "ADI"],
    "ai": ["NVDA", "MSFT", "GOOGL", "META", "AMD", "PLTR", "CRM", "SNOW", "AI", "SMCI",
           "PATH", "BBAI", "SOUN", "UPST", "DDOG", "MDB", "ESTC", "CFLT"],
    "ev": ["TSLA", "RIVN", "LCID", "NIO", "LI", "XPEV", "F", "GM", "PCAR", "BLNK", "CHPT", "QS"],
    "cloud": ["AMZN", "MSFT", "GOOGL", "CRM", "SNOW", "NET", "DDOG", "MDB", "ZS", "CRWD",
              "ESTC", "CFLT", "HUBS", "TEAM", "WDAY"],
    "renewable": ["ENPH", "SEDG", "FSLR", "NEE", "BEP", "RUN", "CSIQ", "DQ",
                  "SPWR", "NOVA", "ARRY", "CWEN", "AES", "PLUG", "BE"],
    "crypto": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "CIFR"],
    "cryptocurrency": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "CIFR"],
    "bitcoin": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF"],
    "blockchain": ["COIN", "MSTR", "MARA", "RIOT", "IBM", "SQ", "PYPL"],
    "fintech": ["SQ", "PYPL", "AFRM", "SOFI", "NU", "COIN", "HOOD", "UPST", "LC", "BILL"],
    "magnificent7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "defense": ["LMT", "RTX", "NOC", "GD", "BA", "HII", "LHX", "TDG", "HWM", "TXT"],
    "biotech": ["AMGN", "GILD", "REGN", "VRTX", "BIIB", "MRNA", "ILMN", "ALNY",
                "SGEN", "BMRN", "INCY", "EXEL", "IONS", "PCVX", "RARE"],
    "retail": ["WMT", "COST", "TGT", "HD", "LOW", "AMZN", "TJX", "ROST",
               "DG", "DLTR", "ORLY", "AZO", "BBY", "ULTA", "FIVE"],
    "gaming": ["NVDA", "AMD", "MSFT", "SONY", "EA", "TTWO", "RBLX", "U",
               "ATVI", "SE", "DKNG", "PENN"],

    # ── New sectors ──────────────────────────────────────────────────
    "software": ["MSFT", "CRM", "ADBE", "NOW", "INTU", "WDAY", "SNOW", "TEAM", "HUBS", "ZM",
                 "DDOG", "MDB", "PANW", "FTNT"],
    "materials": ["LIN", "APD", "ECL", "SHW", "NEM", "FCX", "GOLD", "SCCO", "BHP", "RIO",
                  "DD", "VMC", "MLM", "CF", "NUE"],
    "pharma": ["LLY", "NVO", "AZN", "MRK", "PFE", "ABBV", "BMY", "GSK", "SNY", "ZTS", "VTRS", "TAK"],
    "farmaceuticas": ["LLY", "NVO", "AZN", "MRK", "PFE", "ABBV", "BMY", "GSK", "SNY", "ZTS", "VTRS", "TAK"],
    "medtech": ["ISRG", "SYK", "MDT", "BSX", "EW", "DXCM", "BDX", "ZBH", "HOLX", "ALGN"],
    "insurance": ["BRK-B", "PGR", "TRV", "AIG", "AFL", "ALL", "MET", "PRU", "CB", "HIG"],
    "seguros": ["BRK-B", "PGR", "TRV", "AIG", "AFL", "ALL", "MET", "PRU", "CB", "HIG"],
    "media": ["DIS", "NFLX", "WBD", "PARA", "FOX", "ROKU", "SPOT", "LYV", "IMAX", "CMCSA"],
    "medios": ["DIS", "NFLX", "WBD", "PARA", "FOX", "ROKU", "SPOT", "LYV", "IMAX", "CMCSA"],
    "cybersecurity": ["CRWD", "PANW", "FTNT", "ZS", "S", "CYBR", "RPD", "QLYS"],
    "ciberseguridad": ["CRWD", "PANW", "FTNT", "ZS", "S", "CYBR", "RPD", "QLYS"],
    "food_beverage": ["PEP", "KO", "MDLZ", "GIS", "K", "HSY", "SJM", "CPB", "MKC", "STZ", "BF-B", "TAP", "SAM"],
    "alimentos_bebidas": ["PEP", "KO", "MDLZ", "GIS", "K", "HSY", "SJM", "CPB", "MKC", "STZ", "BF-B", "TAP", "SAM"],
    "consumer_staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "KMB", "EL", "CHD", "SPC", "CLX"],
    "consumo_basico": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "KMB", "EL", "CHD", "SPC", "CLX"],
    "transportation": ["UNP", "CSX", "NSC", "FDX", "JBHT", "ODFL", "XPO", "CHRW", "EXPD", "LSTR"],
    "transporte": ["UNP", "CSX", "NSC", "FDX", "JBHT", "ODFL", "XPO", "CHRW", "EXPD", "LSTR"],
    "agriculture": ["ADM", "BG", "CTVA", "FMC", "MOS", "NTR", "DE", "AGCO", "CF"],
    "agricultura": ["ADM", "BG", "CTVA", "FMC", "MOS", "NTR", "DE", "AGCO", "CF"],
    "water": ["AWK", "WTR", "XYL", "WTRG", "PNR"],
    "agua": ["AWK", "WTR", "XYL", "WTRG", "PNR"],
    "cannabis": ["TLRY", "CGC", "ACB", "CRON", "SNDL"],
    "luxury": ["LVMUY", "LULU", "TPR", "CPRI", "RL", "RH", "BIRK"],
    "lujo": ["LVMUY", "LULU", "TPR", "CPRI", "RL", "RH", "BIRK"],
    "communication_services": ["GOOGL", "META", "DIS", "NFLX", "T", "VZ", "TMUS", "CMCSA",
                               "EA", "TTWO", "RBLX", "SPOT", "PINS", "SNAP"],
    "servicios_comunicacion": ["GOOGL", "META", "DIS", "NFLX", "T", "VZ", "TMUS", "CMCSA",
                               "EA", "TTWO", "RBLX", "SPOT", "PINS", "SNAP"],
    "aerospace": ["BA", "LMT", "RTX", "NOC", "GD", "HII", "TDG", "HWM", "TXT", "SPR"],
    "aeroespacial": ["BA", "LMT", "RTX", "NOC", "GD", "HII", "TDG", "HWM", "TXT", "SPR"],
    "hospitality": ["MAR", "HLT", "H", "WH", "WYNN", "LVS", "MGM", "CZR", "RCL", "CCL", "NCLH"],
    "hoteleria": ["MAR", "HLT", "H", "WH", "WYNN", "LVS", "MGM", "CZR", "RCL", "CCL", "NCLH"],
    "payments": ["V", "MA", "PYPL", "SQ", "FIS", "FISV", "GPN", "WEX", "FOUR", "RPAY"],
    "pagos": ["V", "MA", "PYPL", "SQ", "FIS", "FISV", "GPN", "WEX", "FOUR", "RPAY"],
    "logistics": ["FDX", "UPS", "XPO", "EXPD", "CHRW", "ODFL", "JBHT", "LSTR", "SAIA", "GXO"],
    "logistica": ["FDX", "UPS", "XPO", "EXPD", "CHRW", "ODFL", "JBHT", "LSTR", "SAIA", "GXO"],
    "reits": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "DLR", "PSA", "WELL", "AVB",
              "VTR", "ARE", "MAA", "UDR", "ESS", "SUI", "CPT", "INVH", "CUBE", "EXR"],
    "fibras": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "DLR", "PSA", "WELL", "AVB",
               "VTR", "ARE", "MAA", "UDR", "ESS", "SUI", "CPT", "INVH", "CUBE", "EXR"],
    "data_center": ["EQIX", "DLR", "AMT", "CCI", "SBAC", "VNT"],
    "centro_datos": ["EQIX", "DLR", "AMT", "CCI", "SBAC", "VNT"],
    "healthcare_services": ["UNH", "CI", "HUM", "CNC", "MOH", "ELV", "CVS", "HCA", "THC", "UHS"],
    "servicios_salud": ["UNH", "CI", "HUM", "CNC", "MOH", "ELV", "CVS", "HCA", "THC", "UHS"],
    "diagnostics": ["TMO", "DHR", "A", "IQV", "CRL", "TECH", "NEOG", "MEDP", "WST"],
    "diagnosticos": ["TMO", "DHR", "A", "IQV", "CRL", "TECH", "NEOG", "MEDP", "WST"],
    "waste_management": ["WM", "RSG", "CLH", "CWST", "SRCL", "US", "GFL"],
    "gestion_residuos": ["WM", "RSG", "CLH", "CWST", "SRCL", "US", "GFL"],
    "construction": ["VMC", "MLM", "JCI", "CARR", "OTIS", "LII", "SWK", "MAS", "OC", "BLDR"],
    "construccion": ["VMC", "MLM", "JCI", "CARR", "OTIS", "LII", "SWK", "MAS", "OC", "BLDR"],
    "chemicals": ["LIN", "APD", "ECL", "DD", "DOW", "LYB", "PPG", "ALB", "CE", "EMN", "AXTA"],
    "quimicas": ["LIN", "APD", "ECL", "DD", "DOW", "LYB", "PPG", "ALB", "CE", "EMN", "AXTA"],
    "steel": ["NUE", "STLD", "X", "CLF", "RS", "CMC", "WOR"],
    "acero": ["NUE", "STLD", "X", "CLF", "RS", "CMC", "WOR"],
    "mining": ["NEM", "FCX", "GOLD", "BHP", "RIO", "SCCO", "TECK", "AA", "MP", "LAC", "ALB"],
    "mineria": ["NEM", "FCX", "GOLD", "BHP", "RIO", "SCCO", "TECK", "AA", "MP", "LAC", "ALB"],
    "airlines": ["DAL", "UAL", "LUV", "AAL", "ALK", "JBLU", "SAVE", "HA"],
    "aerolineas": ["DAL", "UAL", "LUV", "AAL", "ALK", "JBLU", "SAVE", "HA"],
    "automotive": ["TM", "F", "GM", "TSLA", "HMC", "STLA", "RIVN", "LCID", "RACE", "APTV"],
    "automotriz": ["TM", "F", "GM", "TSLA", "HMC", "STLA", "RIVN", "LCID", "RACE", "APTV"],
    "education": ["CHGG", "DUOL", "COUR", "LRN", "STRA", "PRDO"],
    "educacion": ["CHGG", "DUOL", "COUR", "LRN", "STRA", "PRDO"],
    "pet_economy": ["CHWY", "IDXX", "ZTS", "WOOF", "TRUP"],
    "mascotas": ["CHWY", "IDXX", "ZTS", "WOOF", "TRUP"],
    "sports_betting": ["DKNG", "PENN", "FLUT", "BETZ", "RSI"],
    "apuestas_deportivas": ["DKNG", "PENN", "FLUT", "BETZ", "RSI"],
    "hydrogen": ["PLUG", "BE", "BLDP", "FCEL"],
    "hidrogeno": ["PLUG", "BE", "BLDP", "FCEL"],
    "nuclear": ["CEG", "VST", "CCJ", "LEU", "SMR", "NNE"],
    "space": ["RKLB", "ASTS", "SPCE", "BA", "LMT", "BKSY", "PL"],
    "espacio": ["RKLB", "ASTS", "SPCE", "BA", "LMT", "BKSY", "PL"],
    "internet": ["GOOGL", "META", "AMZN", "NFLX", "ABNB", "UBER", "LYFT", "DASH", "SNAP", "PINS",
                 "ETSY", "EBAY", "W", "YELP", "GRPN", "ZG", "OPEN"],
    "ecommerce": ["AMZN", "BKNG", "EBAY", "ETSY", "W", "CPNG", "MELI", "SE", "SHOP", "CHWY", "WISH"],
    "comercio_electronico": ["AMZN", "BKNG", "EBAY", "ETSY", "W", "CPNG", "MELI", "SE", "SHOP", "CHWY", "WISH"],
    "3d_printing": ["DDD", "SSYS", "DM", "XONE", "MKFG"],
    "impresion_3d": ["DDD", "SSYS", "DM", "XONE", "MKFG"],
    "quantum_computing": ["IONQ", "RGTI", "QUBT"],
    "computacion_cuantica": ["IONQ", "RGTI", "QUBT"],

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
    "latam": ["AMX", "FMX", "MELI", "NU", "VALE", "PBR", "ITUB", "KOF", "CX", "SQM",
              "GGAL", "YPF", "BSAC", "EC", "ABEV", "BBD", "ILF"],
    "latin_america": ["AMX", "FMX", "MELI", "NU", "VALE", "PBR", "ITUB", "KOF", "CX", "SQM",
                      "GGAL", "YPF", "BSAC", "EC", "ABEV", "BBD", "ILF"],
    "latinoamerica": ["AMX", "FMX", "MELI", "NU", "VALE", "PBR", "ITUB", "KOF", "CX", "SQM",
                      "GGAL", "YPF", "BSAC", "EC", "ABEV", "BBD", "ILF"],
    "chile": ["SQM", "BSAC", "ECL.SN"],
    "colombia": ["EC", "CIB", "AVAL"],
    "argentina": ["GGAL", "YPF", "MELI", "PAM", "TEO", "LOMA", "BMA", "SUPV"],

    # ── New geographic categories ────────────────────────────────────
    "india": ["INFY", "WIT", "HDB", "IBN", "RDY", "TTM", "SIFY", "MMYT", "VEDL", "WNS",
              "WIPRO", "TCS.NS"],
    "uk": ["BP", "SHEL", "AZN", "GSK", "HSBC", "RIO", "UL", "DEO", "BTI", "NGG"],
    "reino_unido": ["BP", "SHEL", "AZN", "GSK", "HSBC", "RIO", "UL", "DEO", "BTI", "NGG"],
    "europe": ["ASML", "NVO", "AZN", "SAP", "SHEL", "TTE", "UL", "DEO", "SNY", "GSK",
               "NXPI", "ING", "PHG", "ABBV", "ERIC", "NOK", "SAN", "STLA", "RACE"],
    "europa_acciones": ["ASML", "NVO", "AZN", "SAP", "SHEL", "TTE", "UL", "DEO", "SNY", "GSK",
                        "NXPI", "ING", "PHG", "ABBV", "ERIC", "NOK", "SAN", "STLA", "RACE"],
    "germany": ["SAP", "SIEGY", "BASFY", "VWAGY", "MBG.DE", "BMW.DE", "DTE.DE"],
    "alemania": ["SAP", "SIEGY", "BASFY", "VWAGY", "MBG.DE", "BMW.DE", "DTE.DE"],
    "canada": ["RY", "TD", "ENB", "CNQ", "BMO", "BNS", "CP", "CNI", "MFC", "SU", "TRP", "BCE", "SHOP",
               "CM", "FFH", "GIB", "WCN", "OTEX", "BB"],
    "australia": ["BHP", "RIO", "ANZBY", "WBSFY", "NAB.AX", "MQG.AX", "WDS.AX", "FMG.AX"],
    "japan": ["TM", "SONY", "NTT", "MUFG", "SMFG", "HMC", "NTDOY", "FANUY", "KYOCY"],
    "japon_acciones": ["TM", "SONY", "NTT", "MUFG", "SMFG", "HMC", "NTDOY", "FANUY", "KYOCY"],
    "japan_stocks": ["TM", "SONY", "NTT", "MUFG", "SMFG", "HMC", "NTDOY", "FANUY", "KYOCY"],
    "china_adr": ["BABA", "JD", "PDD", "BIDU", "NIO", "LI", "XPEV", "TCEHY", "TME", "BILI",
                  "BEKE", "ZTO", "VNET", "MNSO", "TAL", "YMM", "FUTU", "TIGR", "KC", "DIDI"],
    "china": ["BABA", "JD", "PDD", "BIDU", "NIO", "LI", "XPEV", "TCEHY", "TME", "BILI",
              "BEKE", "ZTO", "TAL", "YMM", "FUTU", "TIGR", "KC", "DIDI"],
    "korea": ["PKX", "KB", "SHG", "CPNG", "LPL", "KEP", "SKM"],
    "corea": ["PKX", "KB", "SHG", "CPNG", "LPL", "KEP", "SKM"],
    "southeast_asia": ["SE", "GRAB", "GLBE"],
    "sudeste_asiatico": ["SE", "GRAB", "GLBE"],
    "taiwan": ["TSM", "UMC", "ASX", "AEHR"],
    "israel": ["NICE", "CHKP", "MNDY", "GLBE", "WIX", "TEVA"],
    "spain": ["TEF", "BBVA", "SAN", "IBE.MC"],
    "espana": ["TEF", "BBVA", "SAN", "IBE.MC"],
    "africa_stocks": ["GOLD", "ANG", "HMY", "SBSW", "SSL", "SBSWY", "IMPUY"],
    "africa_acciones": ["GOLD", "ANG", "HMY", "SBSW", "SSL", "SBSWY", "IMPUY"],

    # ── ESG / Sustainability ───────────────────────────────────────
    "esg": ["ESGU", "SUSA", "DSI", "KRMA", "SNPE", "AAPL", "MSFT", "CRM", "ADBE", "NEE"],
    "socialmente_responsable": ["ESGU", "SUSA", "DSI", "KRMA", "SNPE", "AAPL", "MSFT", "CRM", "ADBE", "NEE"],
    "sustainable": ["NEE", "ENPH", "FSLR", "BEP", "ESGU", "SUSA", "DSI", "KRMA", "VEGN", "SNPE"],
    "sustentable": ["NEE", "ENPH", "FSLR", "BEP", "ESGU", "SUSA", "DSI", "KRMA", "VEGN", "SNPE"],

    # ── Criptomonedas (pares USD — expanded) ─────────────────────────
    "cripto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
               "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "ATOM-USD",
               "LTC-USD", "NEAR-USD", "UNI-USD", "AAVE-USD", "FTM-USD", "ALGO-USD",
               "MANA-USD", "SAND-USD", "AXS-USD", "OP-USD", "ARB-USD", "APT-USD",
               "SUI-USD", "TRX-USD", "SHIB-USD"],
    "criptomonedas": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
                       "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "ATOM-USD",
                       "LTC-USD", "NEAR-USD", "UNI-USD", "AAVE-USD", "FTM-USD", "ALGO-USD",
                       "MANA-USD", "SAND-USD", "AXS-USD", "OP-USD", "ARB-USD", "APT-USD",
                       "SUI-USD", "TRX-USD", "SHIB-USD"],
    "crypto_directo": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
                        "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "ATOM-USD",
                        "LTC-USD", "NEAR-USD", "UNI-USD", "AAVE-USD", "FTM-USD", "ALGO-USD",
                        "MANA-USD", "SAND-USD", "AXS-USD", "OP-USD", "ARB-USD", "APT-USD",
                        "SUI-USD", "TRX-USD", "SHIB-USD"],

    # ── Indices bursatiles (expanded) ────────────────────────────────
    "indices": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^N225", "^HSI", "^MXX", "^BVSP",
                "^STOXX50E", "^GDAXI", "^FCHI", "^IBEX", "^KS11", "^NSEI", "^GSPTSE", "^AORD", "^MERV"],
    "indices_usa": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
    "indices_global": ["^GSPC", "^FTSE", "^N225", "^HSI", "^STOXX50E", "^BVSP", "^MXX",
                       "^GDAXI", "^FCHI", "^IBEX", "^KS11", "^NSEI", "^GSPTSE", "^AORD", "^MERV"],
    "ipc": ["^MXX"],
    "sp500": ["^GSPC"],
    "nasdaq": ["^IXIC"],
    "dow_jones": ["^DJI"],
    "dax": ["^GDAXI"],
    "cac40": ["^FCHI"],
    "ibex": ["^IBEX"],
    "nifty": ["^NSEI"],
    "kospi": ["^KS11"],
    "tsx": ["^GSPTSE"],
    "merval": ["^MERV"],

    # ── Futuros (commodities + indices — expanded) ───────────────────
    "futuros": ["GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZW=F", "ES=F", "NQ=F", "YM=F",
                "HG=F", "PL=F", "PA=F", "KC=F", "CT=F", "SB=F"],
    "futures": ["GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZW=F", "ES=F", "NQ=F", "YM=F",
                "HG=F", "PL=F", "PA=F", "KC=F", "CT=F", "SB=F"],
    "commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZW=F",
                    "HG=F", "PL=F", "PA=F", "KC=F", "CT=F", "SB=F"],
    "materias_primas": ["GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZW=F",
                        "HG=F", "PL=F", "PA=F", "KC=F", "CT=F", "SB=F"],
    "oro": ["GC=F"],
    "gold": ["GC=F"],
    "petroleo": ["CL=F"],
    "oil": ["CL=F"],
    "cobre": ["HG=F"],
    "copper": ["HG=F"],
    "platino": ["PL=F"],
    "platinum": ["PL=F"],
    "cafe": ["KC=F"],
    "coffee": ["KC=F"],
    "algodon": ["CT=F"],
    "cotton": ["CT=F"],
    "azucar": ["SB=F"],
    "sugar": ["SB=F"],

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
    "etf_materials": ["XLB", "VAW", "GNR"],
    "etf_materiales": ["XLB", "VAW", "GNR"],
    "etf_utilities": ["XLU", "VPU", "IDU"],
    "etf_servicios_publicos": ["XLU", "VPU", "IDU"],
    "etf_communication": ["XLC", "VOX", "FCOM"],
    "etf_comunicacion": ["XLC", "VOX", "FCOM"],

    # ── ETFs — Tematicos ──────────────────────────────────────────
    "etf_ark": ["ARKK", "ARKW", "ARKG", "ARKF", "ARKQ"],
    "etf_innovacion": ["ARKK", "ARKW", "ARKG", "ARKF", "ARKQ"],
    "etf_dividendos": ["VYM", "SCHD", "HDV", "DVY", "DGRO"],
    "etf_bonos": ["BND", "AGG", "TLT", "IEF", "SHY", "TIPS", "HYG", "LQD"],
    "etf_renta_fija": ["BND", "AGG", "TLT", "IEF", "SHY", "TIPS", "HYG", "LQD"],

    # ── ETFs — Size & Style ──────────────────────────────────────────
    "etf_small_cap": ["IWM", "IJR", "VB", "SCHA", "VTWO", "IWO"],
    "etf_mid_cap": ["IWR", "VO", "MDY", "IJH", "SCHM"],
    "etf_value": ["VTV", "IWD", "SCHV", "VLUE", "RPV", "VONV"],
    "etf_growth": ["VUG", "IWF", "SCHG", "VOOG", "MGK", "IVW"],
    "etf_momentum": ["MTUM", "PDP", "DWAS", "QMOM"],
    "etf_low_vol": ["USMV", "SPLV", "LGLV", "SPHD"],
    "etf_baja_volatilidad": ["USMV", "SPLV", "LGLV", "SPHD"],

    # ── ETFs — Thematic (new) ────────────────────────────────────────
    "etf_esg": ["ESGU", "ESGV", "SUSA", "ESGE", "KRMA"],
    "etf_semiconductor": ["SOXX", "SMH", "PSI", "SOXQ"],
    "etf_semiconductores": ["SOXX", "SMH", "PSI", "SOXQ"],
    "etf_cybersecurity": ["HACK", "CIBR", "BUG", "IHAK"],
    "etf_ciberseguridad": ["HACK", "CIBR", "BUG", "IHAK"],
    "etf_clean_energy": ["ICLN", "QCLN", "TAN", "PBW", "ACES"],
    "etf_energia_limpia": ["ICLN", "QCLN", "TAN", "PBW", "ACES"],
    "etf_water": ["PHO", "FIW", "CGW"],
    "etf_agua": ["PHO", "FIW", "CGW"],
    "etf_water_etf": ["PHO", "FIW", "CGW"],
    "etf_infrastructure": ["PAVE", "IFRA", "IGF", "NFRA"],
    "etf_infraestructura": ["PAVE", "IFRA", "IGF", "NFRA"],
    "etf_ai": ["BOTZ", "ROBO", "IRBO", "AIQ"],
    "etf_inteligencia_artificial": ["BOTZ", "ROBO", "IRBO", "AIQ"],
    "etf_robotics": ["BOTZ", "ROBO", "IRBO"],
    "etf_robotica": ["BOTZ", "ROBO", "IRBO"],
    "etf_cloud": ["SKYY", "CLOU", "WCLD"],
    "etf_nube": ["SKYY", "CLOU", "WCLD"],
    "etf_fintech": ["FINX", "ARKF", "IPAY"],
    "etf_bitcoin": ["IBIT", "FBTC", "BITB", "BITO"],
    "etf_ethereum": ["ETHA", "FETH"],
    "etf_cannabis": ["MJ", "MSOS", "YOLO"],
    "etf_space": ["UFO", "ARKX", "ROKT"],
    "etf_espacio": ["UFO", "ARKX", "ROKT"],
    "etf_gaming": ["HERO", "ESPO", "NERD"],
    "etf_videojuegos": ["HERO", "ESPO", "NERD"],
    "etf_luxury": ["LUXE", "XLY"],
    "etf_lujo": ["LUXE", "XLY"],
    "etf_agriculture": ["MOO", "DBA", "VEGI"],
    "etf_agricultura": ["MOO", "DBA", "VEGI"],

    # ── ETFs — Fixed Income (new) ────────────────────────────────────
    "etf_treasury": ["SHV", "BIL", "SGOV", "SCHO", "VGSH"],
    "etf_tesoro": ["SHV", "BIL", "SGOV", "SCHO", "VGSH"],
    "etf_high_yield": ["HYG", "JNK", "SHYG", "USHY"],
    "etf_alto_rendimiento": ["HYG", "JNK", "SHYG", "USHY"],
    "etf_tips": ["TIP", "SCHP", "VTIP"],
    "etf_corporate": ["LQD", "VCIT", "IGIB", "USIG"],
    "etf_corporativo": ["LQD", "VCIT", "IGIB", "USIG"],
    "etf_muni": ["MUB", "TFI", "VTEB"],
    "etf_municipal": ["MUB", "TFI", "VTEB"],
    "etf_emerging_debt": ["EMB", "VWOB", "PCY", "EMLC"],
    "etf_deuda_emergente": ["EMB", "VWOB", "PCY", "EMLC"],

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
    "etf_india": ["INDA", "INDY", "EPI"],
    "etf_korea": ["EWY", "FLKR"],
    "etf_corea": ["EWY", "FLKR"],
    "etf_canada": ["EWC", "FLCA"],
    "etf_australia": ["EWA", "FLAU"],
    "etf_uk": ["EWU", "FLGB"],
    "etf_reino_unido": ["EWU", "FLGB"],
    "etf_germany": ["EWG", "FLGR"],
    "etf_alemania": ["EWG", "FLGR"],
    "etf_southeast_asia": ["ASEA", "VNM", "EIDO"],
    "etf_sudeste_asiatico": ["ASEA", "VNM", "EIDO"],
    "etf_africa": ["AFK", "EZA"],
    "etf_middle_east": ["GULF", "KSA"],
    "etf_medio_oriente": ["GULF", "KSA"],
    "etf_frontier": ["FM", "FRN"],
    "etf_frontera": ["FM", "FRN"],
    "etf_latam": ["ILF", "EWZ", "EWW", "ECH", "GXG", "ARGT"],
    "etf_real_estate": ["VNQ", "IYR", "XLRE", "RWR", "SCHH", "REZ", "REM"],
    "etf_bienes_raices": ["VNQ", "IYR", "XLRE", "RWR", "SCHH", "REZ", "REM"],
    "etf_healthcare": ["XLV", "VHT", "IBB", "XBI", "IHI", "ARKG"],
    "etf_defense": ["ITA", "PPA", "XAR"],
    "etf_defensa": ["ITA", "PPA", "XAR"],

    # ── UCITS ETFs (London Stock Exchange .L — expanded) ─────────────
    "ucits": ["VWRL.L", "CSPX.L", "IWDA.L", "VUSA.L", "VUAG.L", "HMWO.L",
              "ISF.L", "IUSA.L", "SWDA.L", "VWRP.L", "EIMI.L",
              "EMIM.L", "VFEM.L", "WSML.L", "IUSN.L"],
    "ucits_equity": ["VWRL.L", "CSPX.L", "IWDA.L", "VUSA.L", "VUAG.L", "HMWO.L",
                     "ISF.L", "IUSA.L", "SWDA.L", "VWRP.L", "EIMI.L",
                     "EMIM.L", "VFEM.L", "WSML.L", "IUSN.L"],
    "ucits_bonos": ["AGBP.L", "IGLT.L", "VGOV.L", "SEMB.L", "IBTM.L",
                    "IGLS.L", "VAGS.L", "SUAG.L"],
    "ucits_renta_fija": ["AGBP.L", "IGLT.L", "VGOV.L", "SEMB.L", "IBTM.L",
                         "IGLS.L", "VAGS.L", "SUAG.L"],
    "ucits_dividendos": ["VHYL.L", "VWRL.L", "ISF.L"],
    "ucits_oro": ["SGLN.L"],
    "ucits_commodities": ["SGLN.L"],
    "ucits_esg": ["SUWS.L", "V3AM.L"],
    "ucits_small_cap": ["WSML.L", "IUSN.L"],
    "ucits_pequena_cap": ["WSML.L", "IUSN.L"],
    "ucits_emerging": ["EIMI.L", "EMIM.L", "VFEM.L"],
    "ucits_emergentes": ["EIMI.L", "EMIM.L", "VFEM.L"],
    "ucits_usa": ["VUSA.L", "VUAG.L", "CSPX.L", "IUSA.L"],
    "ucits_europe": ["IMEU.L", "VEUR.L", "MEUD.L"],
    "ucits_europa": ["IMEU.L", "VEUR.L", "MEUD.L"],
    "ucits_japan": ["VJPN.L", "IJPN.L"],
    "ucits_japon": ["VJPN.L", "IJPN.L"],
    "ucits_pacific": ["VAPX.L", "IPXJ.L"],
    "ucits_pacifico": ["VAPX.L", "IPXJ.L"],

    # ── SIC Mexico (ETFs extranjeros disponibles en el SIC) ──────────
    "sic": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VT", "ACWI",
            "VYM", "SCHD", "HDV", "DVY", "DGRO",
            "BND", "AGG", "TLT", "IEF", "LQD", "HYG",
            "GLD", "IAU", "SLV",
            "EWW", "EWZ", "EEM", "VWO", "FXI", "VGK", "EWJ",
            "VWRL.L", "CSPX.L", "IWDA.L"],
    "sic_mexico": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VT", "ACWI",
                   "VYM", "SCHD", "HDV", "DVY", "DGRO",
                   "BND", "AGG", "TLT", "IEF", "LQD", "HYG",
                   "GLD", "IAU", "SLV",
                   "EWW", "EWZ", "EEM", "VWO", "FXI", "VGK", "EWJ",
                   "VWRL.L", "CSPX.L", "IWDA.L"],

    # ── Dividendos global expandido ──────────────────────────────────
    "etf_dividendos_global": ["VYM", "SCHD", "HDV", "DVY", "DGRO",
                              "VHYL.L", "IDV", "SDIV", "FGD", "DEM"],

    # ── Fondos de inversion Mexico (series Yahoo Finance) ────────────
    "fondos_mexico": ["BLKEM1C0-A.MX", "GOLD5+B2-C.MX", "BLKGUB1B0-D.MX",
                      "BLKINTC0-A.MX", "BLKDOLB0-D.MX", "BLKLIQB0-D.MX",
                      "BLKRFAB0-C.MX", "GBMCREB2-A.MX", "GBMMODB2-A.MX",
                      "GBMF2BO.MX", "GBMINTB1.MX"],
    "fondos_inversion": ["BLKEM1C0-A.MX", "GOLD5+B2-C.MX", "BLKGUB1B0-D.MX",
                          "BLKINTC0-A.MX", "BLKDOLB0-D.MX", "BLKLIQB0-D.MX",
                          "BLKRFAB0-C.MX", "GBMCREB2-A.MX", "GBMMODB2-A.MX",
                          "GBMF2BO.MX", "GBMINTB1.MX"],
    "fondos_blackrock": ["BLKEM1C0-A.MX", "GOLD5+B2-C.MX", "BLKGUB1B0-D.MX",
                          "BLKINTC0-A.MX", "BLKDOLB0-D.MX", "BLKLIQB0-D.MX",
                          "BLKRFAB0-C.MX"],
    "blackrock_mexico": ["BLKEM1C0-A.MX", "GOLD5+B2-C.MX", "BLKGUB1B0-D.MX",
                          "BLKINTC0-A.MX", "BLKDOLB0-D.MX", "BLKLIQB0-D.MX",
                          "BLKRFAB0-C.MX"],
    "fondos_gbm": ["GBMCREB2-A.MX", "GBMMODB2-A.MX", "GBMF2BO.MX", "GBMINTB1.MX"],
}


@tool
def validate_tickers(tickers: list[str]) -> str:
    """Valida que cada simbolo de ticker exista en Yahoo Finance.

    Resuelve automaticamente aliases de fondos mexicanos (ej: BLKEM1 →
    BLKEM1C0-A.MX, GOLD5+ → GOLD5+B2-C.MX). Si un nombre corto tiene alias,
    retorna el ticker de Yahoo Finance en la lista 'valid'.

    Retorna un objeto JSON con:
    - 'valid': tickers encontrados (con nombre Yahoo Finance resuelto)
    - 'invalid': tickers no encontrados
    - 'resolved': mapeo nombre_original → nombre_yahoo (solo si hubo alias)
    Usa esto para verificar tickers antes de descargar datos u optimizar.
    """
    valid, invalid = [], []
    resolved_map = {}
    for t in tickers:
        original = t.strip()
        upper = original.upper()
        # Check aliases first
        alias = TICKER_ALIASES.get(original) or TICKER_ALIASES.get(upper)
        check = alias if alias else upper
        try:
            info = yf.Ticker(check).info
            if info and info.get("regularMarketPrice") is not None:
                valid.append(check)
                if alias:
                    resolved_map[original] = check
            else:
                invalid.append(original)
        except Exception:
            invalid.append(original)
    result = {"valid": valid, "invalid": invalid}
    if resolved_map:
        result["resolved"] = resolved_map
    return json.dumps(result)


@tool
def search_tickers_by_sector(sector: str) -> str:
    """Busca los principales simbolos de ticker en un sector, pais, region o tema.

    Soporta busqueda por (~200 categorias, ~1000 tickers unicos):
    - Sector: technology, energy, healthcare, finance, consumer, industrial,
      real_estate, utilities, telecom, semiconductor, ai, ev, cloud, renewable,
      crypto (acciones cripto), fintech, defense, biotech, retail, gaming.
    - Sectores nuevos: software, materials, pharma/farmaceuticas, medtech,
      insurance/seguros, media/medios, cybersecurity/ciberseguridad,
      food_beverage/alimentos_bebidas, consumer_staples/consumo_basico,
      transportation/transporte, agriculture/agricultura, water/agua,
      cannabis, luxury/lujo, communication_services/servicios_comunicacion,
      aerospace/aeroespacial.
    - Pais/Region: mexico, mexicanas, mexico_esr, mexico_ampliado, bmv,
      bolsa_mexicana, brazil, brasil, latam, latinoamerica, chile, colombia, argentina.
    - Paises nuevos: india, uk/reino_unido, europe/europa_acciones,
      germany/alemania, canada, australia, japan/japon_acciones/japan_stocks,
      china_adr, china, korea/corea, southeast_asia/sudeste_asiatico,
      spain/espana, africa_stocks/africa_acciones.
    - Tema: esg, socialmente_responsable, sustainable, sustentable,
      magnificent7, mag7.
    - Criptomonedas (27 pares USD): cripto, criptomonedas, crypto_directo.
      Incluye BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, DOT, MATIC, LINK,
      ATOM, LTC, NEAR, UNI, AAVE, FTM, ALGO, MANA, SAND, AXS, OP, ARB,
      APT, SUI, TRX, SHIB.
    - Indices bursatiles (19 indices): indices, indices_usa, indices_global,
      ipc, sp500, nasdaq, dow_jones, dax, cac40, ibex, nifty, kospi, tsx, merval.
    - Futuros (15 contratos): futuros, futures, commodities, materias_primas,
      oro, gold, petroleo, oil, cobre, copper, platino, platinum,
      cafe, coffee, algodon, cotton, azucar, sugar.
    - ETFs mercado: etf, etfs, etf_mercado.
    - ETFs sector: etf_tech, etf_salud, etf_financiero, etf_energia, etf_industrial,
      etf_consumo, etf_inmobiliario, etf_materials/etf_materiales,
      etf_utilities/etf_servicios_publicos, etf_communication/etf_comunicacion.
    - ETFs tematicos: etf_ark, etf_innovacion, etf_dividendos, etf_bonos, etf_renta_fija.
    - ETFs size/style: etf_small_cap, etf_mid_cap, etf_value, etf_growth,
      etf_momentum, etf_low_vol/etf_baja_volatilidad.
    - ETFs tematicos nuevos: etf_esg, etf_semiconductor/etf_semiconductores,
      etf_cybersecurity/etf_ciberseguridad, etf_clean_energy/etf_energia_limpia,
      etf_water/etf_agua, etf_infrastructure/etf_infraestructura,
      etf_ai/etf_inteligencia_artificial, etf_robotics/etf_robotica,
      etf_cloud/etf_nube, etf_fintech, etf_bitcoin, etf_ethereum,
      etf_cannabis, etf_space/etf_espacio, etf_gaming/etf_videojuegos,
      etf_luxury/etf_lujo, etf_agriculture/etf_agricultura.
    - ETFs renta fija nuevos: etf_treasury/etf_tesoro, etf_high_yield/etf_alto_rendimiento,
      etf_tips, etf_corporate/etf_corporativo, etf_muni/etf_municipal,
      etf_emerging_debt/etf_deuda_emergente.
    - ETFs commodities: etf_oro, etf_plata, etf_commodities.
    - ETFs pais/region: etf_mexico, naftrac, etf_brasil, etf_china, etf_emergentes,
      etf_europa, etf_japon, etf_global, etf_india, etf_korea/etf_corea,
      etf_canada, etf_australia, etf_uk/etf_reino_unido, etf_germany/etf_alemania,
      etf_southeast_asia/etf_sudeste_asiatico, etf_africa, etf_middle_east/etf_medio_oriente,
      etf_frontier/etf_frontera.
    - UCITS (ETFs europeos LSE .L): ucits, ucits_equity, ucits_bonos, ucits_renta_fija,
      ucits_dividendos, ucits_oro, ucits_commodities, ucits_esg,
      ucits_small_cap/ucits_pequena_cap, ucits_emerging/ucits_emergentes,
      ucits_usa, ucits_europe/ucits_europa, ucits_japan/ucits_japon,
      ucits_pacific/ucits_pacifico.
    - SIC Mexico (ETFs extranjeros en el SIC): sic, sic_mexico.
    - Dividendos global: etf_dividendos_global.
    - Fondos de inversion Mexico: fondos_mexico, fondos_inversion, fondos_blackrock,
      blackrock_mexico, fondos_gbm. Incluye fondos BlackRock (BLKEM1, GOLD5+, BLKGUB1,
      BLKINT, BLKDOL, BLKLIQ, BLKRFA) y GBM (GBMCRE, GBMMOD, GBMF2, GBMINT)
      con sus tickers de serie Yahoo Finance (.MX).

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

# Subset for the NL Portfolio Builder — only selection/validation tools.
# Excludes optimization tools so the LLM never sees computed weights and
# is not tempted to modify them.  Weight calculation happens deterministically
# in the dashboard callback.
BUILDER_TOOLS = [validate_tickers, search_tickers_by_sector, fetch_and_analyze]
