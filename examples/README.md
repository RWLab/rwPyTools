# Examples

Small, runnable scripts, one per use case. Each reads your API key from
`RWPYTOOLS_API_KEY` (or prompts for it in a terminal):

```bash
pip install rwpytools
export RWPYTOOLS_API_KEY=...
python examples/quickstart.py
```

| script | shows |
|---|---|
| [`quickstart.py`](quickstart.py) | `Client`, `get()`, `describe_route()`, `.df` |
| [`routing_and_sources.py`](routing_and_sources.py) | `explain()`; `source="auto"` / `"bulk"` / `"live"`; the dataset list |
| [`cache_and_bandwidth.py`](cache_and_bandwidth.py) | first call downloads, second makes no API call; `download()`, `cache`, `force_refresh` |
| [`fx_rwrtools_style.py`](fx_rwrtools_style.py) | `client.fx.get_daily_ohlc`, policy rates, `total_return_index` |
| [`crypto_binance_realtime.py`](crypto_binance_realtime.py) | hourly perps and funding up to the latest hour |
| [`equities_statarb.py`](equities_statarb.py) | StatArb spreads and prices; intraday spreads from the live-only route |
| [`live_endpoints.py`](live_endpoints.py) | Y.O.L.O, RP-Schteroids, earnings surprises, IB shorts, ticker metadata |
| [`async_client.py`](async_client.py) | `AsyncClient` reading several datasets concurrently |

Every bulk export you download is charged against your key's bandwidth cap
(20 GiB). The examples keep their date ranges short; the first run of
`quickstart.py`, `routing_and_sources.py`, `cache_and_bandwidth.py` and
`fx_rwrtools_style.py` downloads a few MB of exports, and the rest read
recent data from the live API.
