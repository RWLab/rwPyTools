"""Strategy and event data served only by the live API.

Every endpoint in https://api.robotwealth.com/v1/docs has a method on
``client.live``; cursor-paginated routes are followed to the last page.

Run::

    RWPYTOOLS_API_KEY=... python examples/live_endpoints.py
"""

from __future__ import annotations

import rwpytools


def main() -> None:
    with rwpytools.Client() as client:
        weights = client.live.yolo_weights()
        print("Y.O.L.O weights:")
        print(weights[["date", "ticker", "combo_weight", "arrival_price"]])

        history = client.live.yolo_historical(days=30)
        print(f"\nY.O.L.O history: {len(history)} rows over {history['date'].nunique()} days")

        rp = client.live.rpschteroids_weights()
        print("\nRisk Premia on Schteroids weights:")
        print(rp)

        surprises = client.live.earnings_recent_surprises(lookback_days=7)
        print(f"\n{len(surprises)} earnings reports in the last 7 days; biggest EPS surprises:")
        print(
            surprises.nlargest(5, "eps_surprise_pct")[
                ["symbol", "report_date", "eps_estimate", "eps_actual", "eps_surprise_pct"]
            ]
        )

        shorts = client.live.ib_short_stocks(ticker="AAPL")
        print("\nIB short availability, AAPL:")
        print(shorts[["date", "ticker", "available", "fee_rate", "rebate_rate"]])

        tickers = client.live.equity_tickers(tickers=["AAPL", "MSFT", "NVDA"])
        print("\nticker metadata:")
        print(tickers[["ticker", "name", "sector", "industry", "scalemarketcap"]])


if __name__ == "__main__":
    main()
