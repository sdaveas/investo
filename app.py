"""Investo – Investment Simulation App."""

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data import fetch_prices, lookup_ticker, search_tickers
from investments import TICKER_TO_INVESTMENT
from simulation import compute_stats, simulate

st.set_page_config(page_title="Investo", page_icon="📈", layout="wide")
st.title("📈 Investo – Investment Simulator")

# ── Session state for custom tickers ─────────────────────────────────────────

if "custom_tickers" not in st.session_state:
    st.session_state.custom_tickers: dict[str, str] = {}  # ticker -> name

# ── Sidebar inputs ──────────────────────────────────────────────────────────

with st.sidebar:
    run = st.button("▶ Run Simulation", type="primary", width="stretch")

    st.header("Simulation Parameters")

    start_date = st.date_input(
        "Start date",
        value=date.today() - timedelta(days=5 * 365),
        max_value=date.today() - timedelta(days=1),
    )

    end_date = st.date_input(
        "End date",
        value=date.today(),
        min_value=start_date + timedelta(days=1),
        max_value=date.today(),
    )

    amount = st.number_input(
        "Initial investment ($)",
        min_value=100.0,
        max_value=10_000_000.0,
        value=10_000.0,
        step=1_000.0,
    )

    selected_tickers: list[str] = []

    # ── Dynamic ticker search ──────────────────────────────────────────────
    st.header("Add Custom Ticker")

    def _do_search() -> None:
        query = st.session_state.custom_ticker_input.strip()
        if not query:
            return
        results = search_tickers(query)
        # Also try exact ticker match and prepend if not already present
        exact_name = lookup_ticker(query)
        if exact_name:
            sym = query.upper()
            if not any(r["symbol"] == sym for r in results):
                results.insert(0, {"symbol": sym, "name": exact_name, "type": "", "exchange": ""})
        st.session_state.search_results = results

    st.text_input(
        "Search by name or ticker (e.g. Google, BRK-B, Bitcoin)",
        key="custom_ticker_input",
        on_change=_do_search,
    )

    # Show search results grouped by type – click a result to add it
    if "search_results" in st.session_state and st.session_state.search_results:
        results = st.session_state.search_results
        _TYPE_LABELS = {
            "EQUITY": "Stocks", "ETF": "ETFs", "MUTUALFUND": "Mutual Funds",
            "FUTURE": "Futures", "INDEX": "Indices", "CURRENCY": "Currencies",
            "CRYPTOCURRENCY": "Crypto", "OPTION": "Options",
        }
        grouped: dict[str, list[dict]] = {}
        for r in results:
            cat = _TYPE_LABELS.get(r.get("type", ""), r.get("type") or "Other")
            grouped.setdefault(cat, []).append(r)
        for cat, items in grouped.items():
            st.caption(f"**{cat}**")
            for r in items:
                sym, nm = r["symbol"], r["name"]
                label = f"{nm} ({sym})"
                if sym in st.session_state.custom_tickers:
                    st.caption(f"  ✓ {label}")
                else:
                    if st.button(label, key=f"add_{sym}"):
                        st.session_state.custom_tickers[sym] = nm
                        st.session_state.auto_run = True
                        del st.session_state.search_results
                        st.rerun()
    elif "search_results" in st.session_state:
        st.warning("No results found. Try a different search term.")

    # Show added tickers with remove buttons
    if st.session_state.custom_tickers:
        st.caption("**Selected assets:**")
        for ticker, name in list(st.session_state.custom_tickers.items()):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.write(f"{name} ({ticker})")
            with c2:
                if st.button("✕", key=f"rm_{ticker}"):
                    del st.session_state.custom_tickers[ticker]
                    st.session_state.auto_run = True
                    st.rerun()
        selected_tickers.extend(st.session_state.custom_tickers.keys())

    # ── Allocation ──
    alloc_dollars: dict[str, float] = {}
    if selected_tickers:
        st.header("Allocation")
        alloc_mode = st.radio("Mode", ["Percentage", "Amount ($)"], horizontal=True, key="alloc_mode")

        def _get_name(ticker: str) -> str:
            if ticker in TICKER_TO_INVESTMENT:
                return TICKER_TO_INVESTMENT[ticker].name
            return st.session_state.custom_tickers.get(ticker, ticker)

        # ── Shared priority order ────────────────────────────────────────────
        if "alloc_order" not in st.session_state:
            st.session_state.alloc_order = []

        prev = st.session_state.alloc_order
        order = [t for t in prev if t in selected_tickers]
        for t in selected_tickers:
            if t not in order:
                order.append(t)
        st.session_state.alloc_order = order
        n = len(order)

        # ── Reorder callbacks ────────────────────────────────────────────────
        def move_up(ticker: str) -> None:
            o = st.session_state.alloc_order
            i = o.index(ticker)
            if i > 0:
                o[i], o[i - 1] = o[i - 1], o[i]

        def move_down(ticker: str) -> None:
            o = st.session_state.alloc_order
            i = o.index(ticker)
            if i < len(o) - 1:
                o[i], o[i + 1] = o[i + 1], o[i]

        if n > 1:
            with st.expander("⇅ Reorder priority"):
                for i, ticker in enumerate(order):
                    c1, c2, c3 = st.columns([4, 1, 1])
                    with c1:
                        st.caption(f"**{i + 1}.** {_get_name(ticker)}")
                    with c2:
                        st.button("↑", key=f"up_{ticker}", on_click=move_up, args=(ticker,), disabled=(i == 0))
                    with c3:
                        st.button("↓", key=f"down_{ticker}", on_click=move_down, args=(ticker,), disabled=(i == n - 1))

        st.caption("Adjusting a slider auto-redistributes among assets below it.")

        # ── Percentage mode ──────────────────────────────────────────────────
        if alloc_mode == "Percentage":
            # Initialize new tickers with equal share of unallocated %
            new_tickers = [t for t in order if f"alloc_{t}" not in st.session_state]
            if new_tickers:
                existing_total = sum(
                    st.session_state.get(f"alloc_{t}", 0)
                    for t in order if t not in new_tickers
                )
                unallocated = max(0, 100 - existing_total)
                each = round(unallocated / len(new_tickers) / 5) * 5 if unallocated > 0 else 0
                for t in new_tickers:
                    st.session_state[f"alloc_{t}"] = each
                total_init = sum(st.session_state.get(f"alloc_{t}", 0) for t in order)
                if total_init != 100:
                    last_key = f"alloc_{new_tickers[-1]}"
                    st.session_state[last_key] = max(0, st.session_state[last_key] + (100 - total_init))

            def redistribute_pct(changed_ticker: str) -> None:
                order = st.session_state.alloc_order
                idx = order.index(changed_ticker)
                below = order[idx + 1:]
                if not below:
                    return
                used = sum(st.session_state[f"alloc_{t}"] for t in order[: idx + 1])
                remaining = max(0, 100 - used)
                old_below = sum(st.session_state[f"alloc_{t}"] for t in below)
                if old_below > 0 and remaining > 0:
                    for t in below:
                        share = st.session_state[f"alloc_{t}"] / old_below * remaining
                        st.session_state[f"alloc_{t}"] = max(0, round(share / 5) * 5)
                elif remaining > 0:
                    each = max(0, round(remaining / len(below) / 5) * 5)
                    for t in below:
                        st.session_state[f"alloc_{t}"] = each
                else:
                    for t in below:
                        st.session_state[f"alloc_{t}"] = 0
                total = sum(st.session_state[f"alloc_{t}"] for t in order)
                if total != 100 and below:
                    last = f"alloc_{below[-1]}"
                    st.session_state[last] = max(0, min(100, st.session_state[last] + (100 - total)))

            for ticker in order:
                st.slider(
                    f"{_get_name(ticker)} ({ticker})",
                    min_value=0, max_value=100, step=5, format="%d%%",
                    key=f"alloc_{ticker}",
                    on_change=redistribute_pct, args=(ticker,),
                )

            allocations_pct = {t: float(st.session_state[f"alloc_{t}"]) for t in order}
            total_pct = sum(allocations_pct.values())
            if abs(total_pct - 100.0) > 0.1:
                st.warning(f"Total: **{total_pct:.0f}%** — must be 100%")
            else:
                st.success(f"Total: **{total_pct:.0f}%** ✓")
            alloc_dollars = {t: amount * pct / 100.0 for t, pct in allocations_pct.items()}

        # ── Amount mode ──────────────────────────────────────────────────────
        else:
            budget = int(amount)
            step_amt = max(100, round(budget / 50 / 100) * 100)

            # Initialize new tickers with equal share of unallocated $
            new_tickers = [t for t in order if f"alloc_amt_{t}" not in st.session_state]
            if new_tickers:
                existing_total = sum(
                    st.session_state.get(f"alloc_amt_{t}", 0)
                    for t in order if t not in new_tickers
                )
                unallocated = max(0, budget - existing_total)
                each = round(unallocated / len(new_tickers) / step_amt) * step_amt if unallocated > 0 else 0
                for t in new_tickers:
                    st.session_state[f"alloc_amt_{t}"] = int(each)
                total_init = sum(st.session_state.get(f"alloc_amt_{t}", 0) for t in order)
                if total_init != budget:
                    last_key = f"alloc_amt_{new_tickers[-1]}"
                    st.session_state[last_key] = max(0, st.session_state[last_key] + (budget - total_init))

            def redistribute_amt(changed_ticker: str) -> None:
                order = st.session_state.alloc_order
                idx = order.index(changed_ticker)
                below = order[idx + 1:]
                if not below:
                    return
                used = sum(st.session_state[f"alloc_amt_{t}"] for t in order[: idx + 1])
                remaining = max(0, budget - used)
                old_below = sum(st.session_state[f"alloc_amt_{t}"] for t in below)
                if old_below > 0 and remaining > 0:
                    for t in below:
                        share = st.session_state[f"alloc_amt_{t}"] / old_below * remaining
                        st.session_state[f"alloc_amt_{t}"] = max(0, round(share / step_amt) * step_amt)
                elif remaining > 0:
                    each = max(0, round(remaining / len(below) / step_amt) * step_amt)
                    for t in below:
                        st.session_state[f"alloc_amt_{t}"] = int(each)
                else:
                    for t in below:
                        st.session_state[f"alloc_amt_{t}"] = 0
                total = sum(st.session_state[f"alloc_amt_{t}"] for t in order)
                if total != budget and below:
                    last = f"alloc_amt_{below[-1]}"
                    st.session_state[last] = max(0, min(budget, st.session_state[last] + (budget - total)))

            for ticker in order:
                st.slider(
                    f"{_get_name(ticker)} ({ticker})",
                    min_value=0, max_value=budget, step=step_amt, format="$%d",
                    key=f"alloc_amt_{ticker}",
                    on_change=redistribute_amt, args=(ticker,),
                )

            alloc_dollars = {t: float(st.session_state[f"alloc_amt_{t}"]) for t in order}
            total_amt = sum(alloc_dollars.values())
            if abs(total_amt - amount) > 1:
                st.warning(f"Total: **${total_amt:,.0f}** / ${amount:,.0f}")
            else:
                st.success(f"Total: **${total_amt:,.0f}** ✓")

    # ── Inflation adjustment ──────────────────────────────────────────────
    st.header("Inflation")
    adjust_inflation = st.toggle("Show inflation-adjusted baseline", value=False)
    inflation_rate = st.number_input(
        "Annual inflation rate (%)",
        min_value=0.0,
        max_value=30.0,
        value=3.0,
        step=0.5,
        help="Average Greek inflation ≈ 2–3% historically. Adjust as needed.",
        disabled=not adjust_inflation,
    )
# ── Main area
# ── Main area ───────────────────────────────────────────────────────────────

# Fetch prices on Run and cache them
auto_run = st.session_state.pop("auto_run", False)
if run or auto_run:
    if not selected_tickers:
        st.warning("Please select at least one investment.")
        st.stop()
    if not alloc_dollars or sum(alloc_dollars.values()) <= 0:
        st.error("Please allocate funds to at least one asset.")
        st.stop()

    with st.spinner("Fetching market data…"):
        prices = fetch_prices(selected_tickers, start_date, end_date)

    if prices.empty:
        st.error("No data returned. Try a different date range or investment.")
        st.stop()

    st.session_state.sim_prices = prices

# ── Render (recomputes from cached prices + current allocations) ───────────

if "sim_prices" in st.session_state and alloc_dollars:
    prices = st.session_state.sim_prices

    # Only use tickers that have both price data and a current allocation
    available = set(str(c) for c in prices.columns)
    active_alloc = {t: v for t, v in alloc_dollars.items() if t in available and v > 0}

    if active_alloc:
        portfolio = simulate(prices, active_alloc)
        total_allocated = sum(active_alloc.values())

        # Build ticker -> display name mapping
        ticker_names: dict[str, str] = {}
        for t in active_alloc:
            if t in TICKER_TO_INVESTMENT:
                ticker_names[t] = TICKER_TO_INVESTMENT[t].name
            elif t in st.session_state.custom_tickers:
                ticker_names[t] = st.session_state.custom_tickers[t]
            else:
                ticker_names[t] = t

        stats = compute_stats(portfolio, total_allocated, ticker_names)

        # ── Chart ───────────────────────────────────────────────────────────
        st.subheader("Portfolio Value Over Time")

        fig = go.Figure()
        for ticker in portfolio.columns:
            label = ticker_names.get(str(ticker), str(ticker))
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=portfolio[ticker],
                    mode="lines",
                    name=f"{label} ({ticker})",
                )
            )

        if len(portfolio.columns) > 1:
            total_value = portfolio.sum(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=total_value.index,
                    y=total_value,
                    mode="lines",
                    name="Total Portfolio",
                    line=dict(color="white", width=3, dash="dot"),
                )
            )

        if adjust_inflation:
            days_elapsed = np.array([(d - portfolio.index[0]).days for d in portfolio.index])
            years_elapsed = days_elapsed / 365.25
            inflation_line = total_allocated * (1 + inflation_rate / 100.0) ** years_elapsed
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=inflation_line,
                    mode="lines",
                    name=f"Initial + {inflation_rate:.1f}% inflation",
                    line=dict(color="red", width=2, dash="dash"),
                )
            )

        fig.add_hline(
            y=total_allocated,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Initial: ${total_allocated:,.0f}",
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Value ($)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=500,
        )

        st.plotly_chart(fig, width="stretch")

        # ── Summary table ───────────────────────────────────────────────────
        st.subheader("Summary")

        if stats:
            cols = st.columns(min(len(stats), 4))
            for i, s in enumerate(stats):
                with cols[i % len(cols)]:
                    st.metric(f"{s.name} ({s.ticker})", f"${s.final_value:,.2f}", f"{s.total_return_pct:+.1f}%")

        summary_data = {
            "Investment": [f"{s.name} ({s.ticker})" for s in stats],
            "Allocated": [f"${active_alloc[s.ticker]:,.2f}" for s in stats],
            "Final Value": [f"${s.final_value:,.2f}" for s in stats],
            "Total Return": [f"{s.total_return_pct:+.2f}%" for s in stats],
            "Annualized Return": [f"{s.annualized_return_pct:+.2f}%" for s in stats],
            "Max Drawdown": [f"{s.max_drawdown_pct:.2f}%" for s in stats],
        }
        st.table(summary_data)
    else:
        st.info("Allocate funds to at least one asset with available price data.")

else:
    st.info("👈 Configure your simulation in the sidebar and click **Run Simulation**.")
