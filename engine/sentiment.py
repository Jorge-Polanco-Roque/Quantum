"""Sentiment analysis via yfinance news + keyword scoring."""

import yfinance as yf


# ── Bilingual keyword dictionaries ───────────────────────────────────

_POSITIVE_KEYWORDS = {
    # English
    "surge", "soar", "rally", "beat", "strong", "growth", "upgrade",
    "bullish", "record", "high", "gain", "profit", "revenue", "outperform",
    "boost", "momentum", "breakout", "innovation", "expansion", "optimistic",
    "buy", "positive", "upside", "recovery", "milestone",
    # Spanish
    "sube", "alza", "crecimiento", "ganancias", "positivo", "fuerte",
    "supera", "record", "avanza", "impulso", "optimista", "mejora",
    "beneficio", "ingresos", "expansion",
}

_NEGATIVE_KEYWORDS = {
    # English
    "crash", "plunge", "drop", "fall", "decline", "loss", "weak",
    "bearish", "downgrade", "risk", "fear", "sell", "warning", "miss",
    "lawsuit", "investigation", "fine", "layoff", "cut", "recession",
    "bubble", "overvalued", "debt", "default", "negative",
    # Spanish
    "cae", "baja", "perdida", "negativo", "debil", "riesgo", "temor",
    "recorte", "demanda", "multa", "despidos", "recesion", "deuda",
    "crisis", "caida",
}


def _score_text(text: str) -> float:
    """Score a text string based on keyword presence.

    Returns a value between -1.0 and 1.0.
    """
    words = set(text.lower().split())
    pos_count = len(words & _POSITIVE_KEYWORDS)
    neg_count = len(words & _NEGATIVE_KEYWORDS)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def fetch_ticker_news(ticker: str, max_items: int = 5) -> list[dict]:
    """Fetch recent news for a ticker via yfinance and score headlines.

    Returns a list of dicts with: title, publisher, link, score.
    Handles both old yfinance format (flat dict) and new format
    (yfinance >= 1.x where data is nested under 'content').
    """
    try:
        t = yf.Ticker(ticker)
        news_items = t.news or []
    except Exception:
        return []

    results = []
    for item in news_items[:max_items]:
        # yfinance >= 1.x nests data under "content"
        content = item.get("content", item)
        title = content.get("title", "") or ""
        pub_raw = content.get("publisher", content.get("provider", ""))
        if isinstance(pub_raw, dict):
            publisher = pub_raw.get("displayName", pub_raw.get("name", ""))
        else:
            publisher = pub_raw or ""
        link = content.get("canonicalUrl", content.get("link", ""))
        # Also score the summary for better coverage
        summary = content.get("summary", "") or ""
        text_to_score = f"{title} {summary}"
        score = _score_text(text_to_score)
        results.append({
            "title": title,
            "publisher": publisher,
            "link": link,
            "score": score,
        })

    return results


def fetch_all_news(
    tickers: list[str],
    max_per_ticker: int = 3,
) -> dict:
    """Fetch and aggregate news sentiment for all tickers.

    Returns dict with:
      - 'por_ticker': {ticker: {'noticias': [...], 'score_promedio': float}}
      - 'score_general': float (average across all tickers)
      - 'resumen': str (formatted summary text)
    """
    por_ticker = {}
    all_scores = []

    for ticker in tickers:
        news = fetch_ticker_news(ticker, max_items=max_per_ticker)
        scores = [n["score"] for n in news]
        avg = sum(scores) / len(scores) if scores else 0.0
        por_ticker[ticker] = {
            "noticias": news,
            "score_promedio": round(avg, 3),
        }
        all_scores.append(avg)

    score_general = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Build formatted summary
    lines = []
    for ticker in tickers:
        info = por_ticker[ticker]
        emoji = "+" if info["score_promedio"] >= 0 else "-"
        lines.append(
            f"{ticker}: sentimiento {emoji}{abs(info['score_promedio']):.2f}"
        )
        for n in info["noticias"]:
            s = n["score"]
            indicator = "+" if s >= 0 else "-"
            lines.append(f"  [{indicator}{abs(s):.1f}] {n['title'][:80]}")

    resumen = "\n".join(lines)

    return {
        "por_ticker": por_ticker,
        "score_general": round(score_general, 3),
        "resumen": resumen,
    }
