"""Fetch historical price data from Yahoo Finance."""

from datetime import date

import pandas as pd
import yfinance as yf


def lookup_ticker(symbol: str) -> str | None:
    """Validate a ticker symbol and return its short name, or None if invalid."""
    symbol = symbol.strip().upper()
    if not symbol:
        return None
    try:
        info = yf.Ticker(symbol).info
        name = info.get("shortName") or info.get("longName")
        if name and info.get("regularMarketPrice") is not None:
            return name
    except Exception:
        pass
    return None


# Common asset keywords -> well-known ETF tickers to surface first.
_ASSET_HINTS: dict[str, list[tuple[str, str]]] = {
    "gold": [("GLD", "SPDR Gold Trust"), ("IAU", "iShares Gold Trust")],
    "silver": [("SLV", "iShares Silver Trust")],
    "oil": [("USO", "United States Oil Fund"), ("BNO", "United States Brent Oil Fund")],
    "natural gas": [("UNG", "United States Natural Gas Fund")],
    "gas": [("UNG", "United States Natural Gas Fund")],
    "platinum": [("PPLT", "abrdn Platinum ETF Trust")],
    "palladium": [("PALL", "abrdn Palladium ETF Trust")],
    "copper": [("CPER", "United States Copper Index Fund")],
    "bonds": [("BND", "Vanguard Total Bond Market ETF"), ("TLT", "iShares 20+ Year Treasury Bond ETF")],
    "treasury": [("TLT", "iShares 20+ Year Treasury Bond ETF"), ("SHV", "iShares Short Treasury Bond ETF")],
    "real estate": [("VNQ", "Vanguard Real Estate ETF")],
    "bitcoin": [("BTC-USD", "Bitcoin USD"), ("IBIT", "iShares Bitcoin Trust ETF")],
    "ethereum": [("ETH-USD", "Ethereum USD")],
    "crypto": [("BTC-USD", "Bitcoin USD"), ("ETH-USD", "Ethereum USD")],
    "s&p": [("SPY", "SPDR S&P 500 ETF"), ("VOO", "Vanguard S&P 500 ETF")],
    "s&p 500": [("SPY", "SPDR S&P 500 ETF"), ("VOO", "Vanguard S&P 500 ETF")],
    "nasdaq": [("QQQ", "Invesco QQQ Trust")],
}


def search_tickers(query: str, max_results: int = 8) -> list[dict]:
    """Search Yahoo Finance by company name or keyword.

    Returns a list of dicts with keys: symbol, name, type, exchange.
    Commodity / asset keywords automatically surface the most relevant ETFs first.
    """
    query = query.strip()
    if not query:
        return []

    # Prepend well-known ETFs for common asset keywords
    hints = _ASSET_HINTS.get(query.lower(), [])
    out: list[dict] = []
    seen: set[str] = set()
    for sym, name in hints:
        out.append({"symbol": sym, "name": name, "type": "ETF", "exchange": ""})
        seen.add(sym)

    try:
        results = yf.Search(query, max_results=max_results).quotes
        for q in results:
            symbol = q.get("symbol", "")
            if symbol and symbol not in seen:
                name = q.get("shortname") or q.get("longname") or symbol
                qtype = q.get("quoteType", "")
                exchange = q.get("exchange", "")
                out.append({"symbol": symbol, "name": name, "type": qtype, "exchange": exchange})
                seen.add(symbol)
    except Exception:
        pass
    return out


def fetch_prices(
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers and date range.

    Returns a DataFrame indexed by date with one column per ticker.
    Missing values (weekends, holidays) are forward-filled.
    """
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers=" ".join(tickers),
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame()

    # yf.download may return MultiIndex columns (newer yfinance) or flat columns.
    # Normalise to a DataFrame with one column per ticker.
    close = raw["Close"]
    if isinstance(close, pd.Series):
        # Single ticker with flat columns → Series; wrap in DataFrame
        prices = close.to_frame(name=tickers[0])
    elif isinstance(close, pd.DataFrame) and isinstance(close.columns, pd.MultiIndex):
        # Shouldn't normally happen after selecting "Close", but flatten just in case
        close.columns = close.columns.get_level_values(-1)
        prices = close
    else:
        prices = close
        # If single ticker returned a 1-column DF with wrong name, fix it
        if len(tickers) == 1 and list(prices.columns) != tickers:
            prices.columns = tickers

    prices = prices.ffill().dropna(how="all")
    return prices
