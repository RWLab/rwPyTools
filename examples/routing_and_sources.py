"""Choosing where rows come from.

Every read takes ``source=``:

* ``"auto"`` (default) — cached bulk export for history, live API for
  anything newer than the export's watermark;
* ``"bulk"`` — the cached export only (downloaded if missing);
* ``"live"`` — the live API only.

``client.explain()`` shows the routing decision without fetching any rows.

Run::

    RWPYTOOLS_API_KEY=... python examples/routing_and_sources.py
"""

from __future__ import annotations

import datetime as dt

import rwpytools


def main() -> None:
    week_ago = dt.date.today() - dt.timedelta(days=7)

    with rwpytools.Client() as client:
        # Dry run: reads only local cache metadata.
        print("plan:", client.explain("fx_daily", symbols=["EURUSD"], start="2020-01-01"))

        for source in ("auto", "bulk", "live"):
            result = client.get("fx_daily", symbols=["EURUSD"], start=week_ago, source=source)
            print(f"\nsource={source!r}: {result.describe_route()}")
            print(result.df.tail(3))

        # Every dataset get() can route, with its bulk export and live endpoint.
        print("\nroutable datasets:")
        for spec in client.datasets():
            bulk = (
                f"{spec.bulk.namespace}/{spec.bulk.dataset}/{spec.bulk.schema}"
                if spec.bulk
                else "-"
            )
            live = spec.live.path if spec.live else "-"
            print(f"  {spec.name:<36} bulk={bulk:<40} live={live}")


if __name__ == "__main__":
    main()
