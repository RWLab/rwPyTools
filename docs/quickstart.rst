Quickstart
==========

Installation
------------

.. code-block:: bash

    pip install rwpytools

Authentication
--------------

Pass the key to :class:`rwpytools.Client`, or set ``RWPYTOOLS_API_KEY``:

.. code-block:: python

    import rwpytools

    client = rwpytools.Client("...")

In an IPython / Jupyter / Colab kernel or a terminal, omitting the key opens
a masked stdin prompt (disable with ``prompt_for_api_key=False`` or
``RWPYTOOLS_NO_PROMPT=1``).

First fetch
-----------

.. code-block:: python

    result = client.get("vix", start="2020-01-01")
    print(result.describe_route())
    df = result.df

The first call downloads the dataset's bulk export through a signed Cloud CDN
URL and caches it on disk. Every later call reads the cached export — without
calling rw-api at all, so no bandwidth is charged — and appends the rows newer
than the export's watermark from the dataset's live endpoint. ``result.route``
records exactly which rows came from where.

Choosing the source
-------------------

.. code-block:: python

    client.get("vix", source="bulk")               # the cached export only
    client.get("vix", source="live", start="2026-09-01")  # the live endpoint only
    client.get("vix", force_refresh=True)          # re-download the export first
    print(client.explain("vix", start="2020-01-01"))  # the plan, without fetching

The same ``source=``, ``start=``, ``end=``, ``symbols=`` and ``force_refresh=``
arguments work on every rwRtools-style pod method:

.. code-block:: python

    client.crypto.get_binance_perps_1h(symbols=["BTCUSDT"], start="2025-01-01")
    client.macro.get_vix_vix3m(source="bulk")

FX, rwRtools-shaped
-------------------

.. code-block:: python

    # One concatenated frame, grouped by ticker in input order.
    all_fx = client.fx.get_daily_ohlc(["EURUSD", "GBPUSD", "USDJPY"])

    # Per-ticker handles:
    frames = client.fx.get_daily_ohlc_frames(["EURUSD", "GBPUSD"])
    eur_df = frames["EURUSD"]

Each pair is its own export; missing ones download concurrently, bounded by
``ClientConfig.download_concurrency`` (default 8), and all pairs are topped up
from ``/v1/fx/daily``.

Live-only data
--------------

.. code-block:: python

    factors = client.live.yolo_factors()
    weights = client.live.rpschteroids_weights()
    spreads = client.live.statarb_hourly_top_spreads()

Async
-----

:class:`rwpytools.AsyncClient` has the same surface with coroutines:

.. code-block:: python

    import asyncio
    import rwpytools

    async def main() -> None:
        async with rwpytools.AsyncClient() as client:
            result = await client.get("statarb_spreads", start="2026-01-01")
            eurusd = await client.fx.get_daily_ohlc_ticker("EURUSD")

    asyncio.run(main())
