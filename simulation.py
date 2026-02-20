"""Portfolio simulation and summary statistics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class InvestmentResult:
    """Summary statistics for a single investment."""

    ticker: str
    name: str
    initial_value: float
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float


def simulate(
    prices: pd.DataFrame,
    allocations: dict[str, float],
) -> pd.DataFrame:
    """Compute portfolio value over time for each ticker.

    Args:
        prices: DataFrame with date index and one column per ticker (close prices).
        allocations: Mapping of ticker -> dollar amount allocated to that ticker.

    Returns:
        DataFrame with the same shape, where each cell is the portfolio value on that date.
    """
    if prices.empty:
        return prices

    portfolio = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        ticker_str = str(ticker)
        alloc = allocations.get(ticker_str, 0.0)
        if alloc <= 0:
            continue
        first_price = prices[ticker].dropna().iloc[0]
        units = alloc / first_price
        portfolio[ticker] = prices[ticker] * units

    return portfolio


def compute_stats(
    portfolio: pd.DataFrame,
    amount: float,
    ticker_names: dict[str, str],
) -> list[InvestmentResult]:
    """Derive summary statistics for each investment in the portfolio.

    Args:
        portfolio: DataFrame of portfolio values over time (output of simulate()).
        amount: The original investment amount.
        ticker_names: Mapping of ticker -> display name.

    Returns:
        List of InvestmentResult, one per ticker.
    """
    results: list[InvestmentResult] = []

    for ticker in portfolio.columns:
        series = portfolio[ticker].dropna()
        if series.empty:
            continue

        initial = series.iloc[0]
        final = series.iloc[-1]
        total_return = (final - initial) / initial * 100

        # Annualized return
        days = (series.index[-1] - series.index[0]).days
        if days > 0:
            annualized = ((final / initial) ** (365.25 / days) - 1) * 100
        else:
            annualized = 0.0

        # Max drawdown
        cummax = series.cummax()
        drawdown = (series - cummax) / cummax
        max_dd = drawdown.min() * 100

        results.append(
            InvestmentResult(
                ticker=str(ticker),
                name=ticker_names.get(str(ticker), str(ticker)),
                initial_value=round(float(initial), 2),
                final_value=round(float(final), 2),
                total_return_pct=round(float(total_return), 2),
                annualized_return_pct=round(float(annualized), 2),
                max_drawdown_pct=round(float(max_dd), 2),
            )
        )

    return results
