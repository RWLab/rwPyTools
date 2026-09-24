"""StatArb: daily pair factors, pair-universe prices, and intraday spreads.

The daily spreads and prices have bulk exports with live companions; the
intraday (hourly) spreads exist only on the live API.

Run::

    RWPYTOOLS_API_KEY=... python examples/equities_statarb.py
"""

from __future__ import annotations

import datetime as dt

import rwpytools


def main() -> None:
    start = dt.date.today() - dt.timedelta(days=10)

    with rwpytools.Client() as client:
        spreads = client.get("statarb_spreads", start=start)
        print(spreads.describe_route())
        latest_day = spreads.df["date"].max()
        top = spreads.df[spreads.df["date"] == latest_day].nsmallest(5, "combo_rank")
        print(f"top-ranked pairs on {latest_day}:")
        print(top[["ticker", "stock2", "combo_rank", "zscore", "lsr_score"]])

        # Prices for the legs of the best-ranked pair.
        legs = [str(top.iloc[0]["ticker"]), str(top.iloc[0]["stock2"])]
        prices = client.get("statarb_prices", symbols=legs, start=start)
        print("\n" + prices.describe_route())
        print(prices.df.pivot_table(index="date", columns="ticker", values="close").tail())

        # Intraday spreads: live-only.
        hourly = client.live.statarb_hourly_top_spreads(limit=10)
        print("\nintraday top spreads:")
        print(hourly[["date", "ticker", "stock2", "zscore"]].head())

        # The liquid universe variant, via the rwRtools-style pod method.
        liquid = client.equity_factors.get_statarb_liquid_spreads(start=start)
        print(f"\nliquid-universe spreads: {len(liquid)} rows since {start}")


if __name__ == "__main__":
    main()
