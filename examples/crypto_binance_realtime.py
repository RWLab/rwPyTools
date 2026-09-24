"""Binance perpetuals, hourly bars and funding, up to the latest hour.

With nothing cached and a recent ``start``, the client serves the request
from the live API instead of downloading the multi-GB hourly export. Widen
``start`` (or leave it off) to pull the full history once; later calls then
read it from disk and only fetch the newest hours live.

Run::

    RWPYTOOLS_API_KEY=... python examples/crypto_binance_realtime.py
"""

from __future__ import annotations

import datetime as dt

import rwpytools

SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def main() -> None:
    start = dt.date.today() - dt.timedelta(days=5)

    with rwpytools.Client() as client:
        bars = client.get("binance_perps_1h", symbols=SYMBOLS, start=start)
        print(bars.describe_route())
        df = bars.df
        print(df.groupby("Ticker")["Datetime"].agg(["min", "max", "count"]))

        # Hourly close-to-close returns per symbol.
        df = df.sort_values(["Ticker", "Datetime"])
        df["return"] = df.groupby("Ticker")["Close"].pct_change()
        print(df.groupby("Ticker")["return"].describe())

        funding = client.get("binance_perps_funding", symbols=SYMBOLS, start=start)
        print("\n" + funding.describe_route())
        print(funding.df.sort_values("fundingTimeHR").tail(6))

        # The rwRtools-style pod method routes the same way.
        spot = client.crypto.get_binance_spot_1h(symbols=["BTCUSDT"], start=start)
        print(f"\nspot BTCUSDT: {len(spot)} hourly bars, last at {spot['Datetime'].max()}")


if __name__ == "__main__":
    main()
