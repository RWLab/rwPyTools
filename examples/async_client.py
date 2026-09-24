"""AsyncClient: the same surface with coroutines, for concurrent reads.

Run::

    RWPYTOOLS_API_KEY=... python examples/async_client.py
"""

from __future__ import annotations

import asyncio
import datetime as dt

import rwpytools


async def main() -> None:
    start = dt.date.today() - dt.timedelta(days=7)

    async with rwpytools.AsyncClient() as client:
        vix, eurusd, spreads, weights = await asyncio.gather(
            client.get("vix", start=start),
            client.fx.get_daily_ohlc_ticker("EURUSD", start=start),
            client.get("statarb_spreads", start=start),
            client.live.yolo_weights(),
        )

        print(vix.describe_route())
        print(f"EURUSD: {len(eurusd)} daily bars, last close {eurusd['Close'].iloc[-1]}")
        print(spreads.describe_route())
        print(f"Y.O.L.O: {len(weights)} weights as of {weights['date'].max()}")


if __name__ == "__main__":
    asyncio.run(main())
