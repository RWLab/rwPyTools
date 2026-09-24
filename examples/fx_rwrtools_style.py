"""FX the rwRtools way: daily OHLC plus a carry-adjusted total-return index.

Pod methods mirror rwRtools' ``fx_*`` functions and return rwRtools-shaped
columns (``Date, Open, High, Low, Close, Volume, Ticker``). Daily bars are
topped up to real time from the live API.

Run::

    RWPYTOOLS_API_KEY=... python examples/fx_rwrtools_style.py
"""

from __future__ import annotations

import pandas as pd

import rwpytools

PAIRS = ["EURUSD", "AUDUSD"]


def policy_rates(client: rwpytools.Client, currencies: list[str]) -> pd.DataFrame:
    """Policy-rate histories in the long shape ``total_return_index`` wants:
    ``Currency, Date, Rate`` (rates in percent)."""

    frames = []
    for currency in currencies:
        raw = client.fx.get_policy_rates(currency, start="2015-01-01")
        # Each export has a "Time Period" column and one value column named
        # for the central bank's country code (e.g. "US").
        value_column = next(c for c in raw.columns if c != "Time Period")
        frames.append(
            pd.DataFrame(
                {
                    "Currency": currency,
                    "Date": pd.to_datetime(raw["Time Period"]),
                    "Rate": raw[value_column],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    with rwpytools.Client() as client:
        print("available pairs:", client.fx.list_symbols()[:10], "...")

        prices = client.fx.get_daily_ohlc(PAIRS, start="2020-01-01")
        print(prices.groupby("Ticker")["Date"].agg(["min", "max", "count"]))

        currencies = client.fx.get_unique_currencies(prices)
        rates = policy_rates(client, currencies)

        tri = client.fx.total_return_index(prices, rates)
        latest = tri.groupby("Ticker").tail(1)
        print(latest[["Ticker", "Date", "Spot_Return_Index", "Total_Return_Index"]])


if __name__ == "__main__":
    main()
