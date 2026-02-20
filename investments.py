"""Predefined investment catalog grouped by category."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Investment:
    ticker: str
    name: str
    category: str


INVESTMENTS: list[Investment] = [
    # US Stock Indices
    Investment("SPY", "S&P 500", "US Stocks (Index)"),
    Investment("QQQ", "NASDAQ 100", "US Stocks (Index)"),
    Investment("VTI", "Total US Market", "US Stocks (Index)"),
    # International
    Investment("VXUS", "International Developed", "International"),
    Investment("VWO", "Emerging Markets", "International"),
    # Bonds
    Investment("BND", "Total Bond Market", "Bonds"),
    Investment("TLT", "Long-Term Treasury", "Bonds"),
    Investment("SHV", "Short-Term Treasury", "Bonds"),
    # Commodities
    Investment("GLD", "Gold", "Commodities"),
    Investment("USO", "Oil", "Commodities"),
    # Real Estate
    Investment("VNQ", "US Real Estate", "Real Estate"),
    # Individual Stocks
    Investment("AAPL", "Apple", "Individual Stocks"),
    Investment("MSFT", "Microsoft", "Individual Stocks"),
    Investment("GOOGL", "Alphabet (Google)", "Individual Stocks"),
    Investment("AMZN", "Amazon", "Individual Stocks"),
    Investment("TSLA", "Tesla", "Individual Stocks"),
    Investment("NVDA", "NVIDIA", "Individual Stocks"),
]

INVESTMENTS_BY_CATEGORY: dict[str, list[Investment]] = {}
for inv in INVESTMENTS:
    INVESTMENTS_BY_CATEGORY.setdefault(inv.category, []).append(inv)

TICKER_TO_INVESTMENT: dict[str, Investment] = {inv.ticker: inv for inv in INVESTMENTS}
