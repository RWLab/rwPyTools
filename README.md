# rwpytools

[![CI](https://github.com/Robot-Wealth/rwPyTools/actions/workflows/ci.yml/badge.svg)](https://github.com/Robot-Wealth/rwPyTools/actions/workflows/ci.yml)

The official Python client for the [Robot Wealth](https://robotwealth.com) research
data API — the Python equivalent of [rwRTools](https://github.com/RWLab/rwRtools).

rw-api publishes most datasets two ways: a **bulk export** (feather / parquet /
csv, downloaded through a short-lived signed Cloud CDN URL) and a **live
endpoint** backed by the PostgreSQL warehouse, which always holds the newest
rows. The export is refreshed on a schedule; anything after its last date — its
*watermark* — exists only behind the live endpoint.

This library makes that split disappear. `client.get(...)` downloads the export
once, caches it, and on every later call reads it from disk and **tops it up
from the live endpoint with only the rows newer than the watermark** — the
diff — joining the two into one DataFrame. And it tells you exactly what it did.

```python
import rwpytools

client = rwpytools.Client("...")

result = client.get("fx_daily", symbols=["EURUSD", "AUDUSD"])

print(result.describe_route())
# fx_daily [-inf .. now] -> bulk[-inf .. 2026-09-23] (cache) live[2026-09-23 .. now]
#   (split at 2026-09-23) — cached export ends 2026-09-23; bulk through the watermark,
#   live diff from 2026-09-23 | 15191 bulk + 0 live = 15191 rows; bulk served from cache

df = result.df
```

## Install

```bash
pip install rwpytools
```

Python 3.10+. Depends on `httpx`, `pandas`, `pyarrow`, `pydantic` and `platformdirs`.

## Authentication

```python
client = rwpytools.Client("...")          # explicit
client = rwpytools.Client()               # RWPYTOOLS_API_KEY
```

Resolution order: constructor argument → `RWPYTOOLS_API_KEY` → a masked stdin
prompt (only in a notebook or a TTY; `prompt_for_api_key=False` or
`RWPYTOOLS_NO_PROMPT=1` disables it). A key entered at the prompt is kept in
`RWPYTOOLS_API_KEY` for the life of the kernel, so re-running a cell does not
re-prompt. The endpoint defaults to `https://api.robotwealth.com` and can be
overridden with `base_url=` or `RWPYTOOLS_API_BASE_URL`.

rw-api reads the key from the `api_key` query parameter. The library redacts it
from every log record and masks it in every `repr`.

## The routing decision

`client.get()` is the recommended entry point.

| situation | route |
|---|---|
| nothing cached | download the export, then top up live |
| nothing cached, `start` within the last 30 days | all live — no multi-GB download for a few recent days |
| export cached | read it from disk (**no rw-api call**) + live diff from the watermark |
| requested range ends before the watermark | all bulk, from cache |
| requested range starts after the watermark | all live |
| cached watermark > 30 days old *and* the copy is past `bulk_ttl` | re-validate / re-download the export, then top up live |
| dataset has no live endpoint | bulk, re-validated once the copy is older than `bulk_ttl` (24h) |
| dataset has no bulk export | all live |

The live half starts **on** the watermark day (hourly exports can end mid-day).
Rows both sides return for that seam day are kept once, **with the live
values**: an export's newest day can be provisional (the StatArb price export
has carried half-day volumes for its latest date), while the live route holds
the settled numbers. Timestamps are matched to the second, absorbing the
millisecond jitter in Binance funding times. Live rows are renamed and typed to the export's schema
— the Binance export's `Ticker` / `Datetime` / `Open` and the live route's
`ticker` / `date` / `open` come back as one rectangular frame. Export-only
columns (e.g. `Number of trades`) are `NaN` on live rows.

```python
result = client.get("vix", start="2020-01-01")

result.route.sources        # ('bulk', 'live')
result.route.split_at       # datetime.date(2026, 9, 24)
result.watermark            # Timestamp('2026-09-24 00:00:00')
result.bulk_rows, result.live_rows
result.seam_updates         # seam rows whose values were refreshed from live
result.served_from_cache    # was the export already on disk?
result.warnings             # e.g. the live endpoint failed; export returned alone
```

### Choosing the source yourself

Every read — `client.get()` and every rwRtools-style pod method — takes
`source=`:

```python
client.get("vix", source="bulk")      # the cached export only (downloaded if missing)
client.get("vix", source="live")      # the live endpoint only
client.get("vix", source="auto")      # the default: export + live diff

client.macro.get_vix_vix3m(source="bulk")
client.crypto.get_binance_spot_1h(source="live", start="2026-09-20", symbols=["BTCUSDT"])
client.fx.get_daily_ohlc(["EURUSD", "GBPUSD"], source="bulk", start="2020-01-01")

client.get("vix", force_refresh=True)  # re-download the export first
```

Preview the decision without fetching anything (reads only local cache metadata):

```python
print(client.explain("binance_spot_1h", start="2024-01-01"))
```

### Why the cache matters — bandwidth

rw-api charges an export's **full size** against your key's bandwidth cap
(20 GiB) the moment it issues a signed URL — even if the bytes then come from a
local cache. So a cached export is used without asking rw-api at all, and kept
current from the live endpoint instead of being re-downloaded. Datasets without
a live endpoint are re-validated (one signed-URL issue, re-downloaded only if
the file changed) once their copy is older than `bulk_ttl`.

## Datasets

```python
for spec in client.datasets():
    print(spec.name, spec.bulk, spec.live and spec.live.path)
```

| dataset | bulk export | live endpoint |
|---|---|---|
| `binance_spot_1h` | crypto `binance.spot` / `ohlcv-1h` | `/v1/crypto/binance/spot` |
| `binance_spot_1d` | crypto `binance.spot` / `ohlcv-1d` | — |
| `binance_perps_1h` | crypto `binance.perps` / `ohlcv-1h` | `/v1/crypto/binance/perps` |
| `binance_perps_funding` | crypto `binance.perps` / `funding` | `/v1/crypto/binance/perps/funding` |
| `binance_coin_m_perps_1h` | crypto `binance.perps.coin_m` / `ohlcv-1h` | `/v1/crypto/binance/coin-m/perps` |
| `binance_coin_m_funding` | crypto `binance.perps.coin_m` / `funding` | `/v1/crypto/binance/coin-m/funding` |
| `coincodex` | crypto `coincodex.marketcap` / `snapshot` | — |
| `asset_classes` | macro `asset_classes` / `ohlcv-1d` | `/v1/macro/asset-classes` |
| `earnings_calendar` | macro `earnings` / `calendar` | — |
| `futures` | macro `futures` / `ohlcv-1d` | — |
| `rp_futures` | macro `rp.futures` / `ohlcv-1d` | `/v1/macro/futures` |
| `vix` | macro `vix` / `ohlcv-1d` | `/v1/macro/vix` |
| `vix_futures` | macro `vix.futures` / `ohlcv-1d` | — |
| `tbill_rates` | macro `tbill` / `rates` | `/v1/macro/rates` |
| `ib_shortstock` | macro `ib.shortstock` / `yearly` (one file per year) | `/v1/equities/ib-short-stocks` |
| `fx_daily` | fx `pairs` / `ohlcv-1d` (one file per pair) | `/v1/fx/daily` |
| `fx_hourly` | fx `pairs` / `ohlcv-1h` (one file per pair) | — |
| `fx_policy_rates` | fx `policy.rates` / `history` (one file per currency) | — |
| `earnings_surprises` | equities `earnings` / `surprises` | `/v1/equities/earnings/recent-surprises` |
| `liquid_universe_prices` | equities `equity.factors` / `prices` | — |
| `liquid_universe_sectors` | equities `equity.factors` / `sectors` | — |
| `statarb_spreads` | equities `statarb` / `spreads` | `/v1/equities/statarb/daily/topspreads` |
| `statarb_prices` | equities `statarb` / `prices` | `/v1/equities/statarb/prices` |
| `statarb_acquisitions` | equities `statarb` / `acquisitions` | `/v1/equities/acquisitions` |
| `statarb_liquid_spreads` | equities `statarb.liquid` / `spreads` | `/v1/equities/statarb/liquid/daily/topspreads` |
| `statarb_liquid_prices` | equities `statarb.liquid` / `prices` | — |
| `statarb_liquid_acquisitions` | equities `statarb.liquid` / `acquisitions` | `/v1/equities/acquisitions` |
| `streaks` | equities `streaks` / `factors` | — |
| `equity_tickers` | equities `tickers` / `snapshot` | `/v1/equities/tickers` (via `source="live"`) |
| `statarb_hourly_top_spreads` | — | `/v1/equities/statarb/hourly/topspreads` |
| `statarb_liquid_hourly_top_spreads` | — | `/v1/equities/statarb/liquid/hourly/topspreads` |
| `yolo_factors`, `yolo_weights`, `yolo_volatilities`, `yolo_historical` | — | `/v1/yolo/*` |
| `rpschteroids_weights` | — | `/v1/rpschteroids/weights` |

Per-symbol exports need `symbols=` (`client.get("fx_daily", symbols=["EURUSD"])`);
`ib_shortstock` picks its yearly files from `start` / `end` (default: the current year).

## Pods (rwRtools parity)

Pod method names mirror the rwRtools function with the `<pod>_get_` prefix
dropped, and return DataFrames routed exactly like `client.get()`:

```python
client.crypto.get_binance_spot_1h()
client.fx.get_daily_ohlc(["EURUSD", "GBPUSD"])       # Date, Open, High, Low, Close, Volume, Ticker
client.fx.get_policy_rates("USD")
client.macro.get_historical_short_sale(2025)
client.equity_factors.get_statarb_spreads(start="2026-01-01")
```

| Pod | Bucket | Attribute |
|---|---|---|
| Crypto | `crypto_research_pod` | `client.crypto` |
| FX | `fx_research_pod` | `client.fx` |
| Macro | `macro_research_pod` | `client.macro` |
| Equity factors | `equity_factors_research_pod` | `client.equity_factors` |

rwRtools functions whose data rw-api does not publish (the FTX series,
`coinmetrics`, Zorro asset lists, ...) raise `DatasetNotSeededError`.

## Live endpoints

Every endpoint in the [API docs](https://api.robotwealth.com/v1/docs) has a
method on `client.live`, returning a DataFrame. Cursor-paginated routes are
followed to the last page automatically.

```python
client.status()
client.live.yolo_factors()
client.live.yolo_historical(days=30)
client.live.rpschteroids_weights()
client.live.statarb_daily_top_spreads(gte="2026-09-01", limit=5000)
client.live.statarb_liquid_hourly_top_spreads()
client.live.earnings_recent_surprises(lookback_days=20, tickers=["AAPL", "MSFT"])
client.live.ib_short_stocks(ticker="AAPL", gte="2026-09-01")
client.live.forex_daily(ticker="EURUSD", gte="2026-01-01")
client.live.binance_perps_funding(ticker="BTCUSDT")
client.live.macro_rates(symbol="%IRX")
client.live.equity_tickers(tickers=["AAPL", "NVDA"])
```

## Async

`AsyncClient` mirrors `Client` exactly, with coroutines:

```python
async with rwpytools.AsyncClient() as client:
    result = await client.get("statarb_spreads", start="2026-01-01")
    eurusd = await client.fx.get_daily_ohlc_ticker("EURUSD")
```

## Cache

Exports live under the platform cache directory (`RWPYTOOLS_CACHE_DIR`), at
`<cache_dir>/<bucket>/<object>`, each with a `.meta.json` sidecar recording its
size, fetch time, the rw-api cell it serves, and its watermark once read.
Downloads commit via temp file + atomic rename, so an interrupted transfer is
never served as a cache hit. Eviction is LRU by fetch time against a 10 GiB
budget (`RWPYTOOLS_CACHE_MAX_BYTES`). The cache is thread-safe within a
process; it is not coordinated across processes.

```python
client.cache.total_bytes()
client.cache.entries()
client.cache.clear()
client.download("binance_spot_1h")                 # just make sure it is on disk
client.download("fx_daily", symbol="EURUSD")
```

## Errors

```python
from rwpytools.errors import (
    RwPyToolsError,           # base for everything
    ConfigError,              # bad client construction
    RoutingError,             # unknown dataset, bad source / range, missing symbols=
    AuthenticationError,      # 401 — bad API key
    AuthorizationError,       # 403 — valid key, no access
    BandwidthExceededError,   # 403 — bandwidth cap reached; .limit_gb
    NotFoundError,            # 404
    RateLimitError,           # 429 — .retry_after
    ServerError,              # 5xx
    SignedUrlExpiredError,    # CDN rejected the URL — re-minted internally once
    CacheError,               # unrecoverable on-disk cache state
    DataFormatError,          # a response shape the client does not understand
    DatasetNotSeededError,    # rwRtools function whose data rw-api does not publish
)
```

`GET`s are retried with exponential backoff and full jitter on transport
errors, 429 and 5xx. rw-api's rate limits are per minute and it sends no
`Retry-After`, so a 429 waits `rate_limit_wait` seconds per attempt. Every
`RwApiError` carries the `request_id` sent with the request.

In `source="auto"`, a live endpoint that fails with a 5xx, 429 or transport
error does not fail the read: the cached export is returned alone and the
failure is recorded in `result.warnings` (and logged). With `source="live"` it
raises.

## Configuration

Every setting takes a constructor argument or an environment variable
(argument wins).

| Environment variable | Default | Meaning |
|---|---|---|
| `RWPYTOOLS_API_KEY` | — | API key |
| `RWPYTOOLS_API_BASE_URL` | `https://api.robotwealth.com` | API endpoint |
| `RWPYTOOLS_CACHE_DIR` | platform cache dir | Export cache root |
| `RWPYTOOLS_CACHE_MAX_BYTES` | 10 GiB | Cache budget (0 disables eviction) |
| `RWPYTOOLS_BULK_TTL` | 86400 | Seconds a bulk-only export is trusted before re-validation |
| `RWPYTOOLS_LIVE_DIFF_MAX_DAYS` | 30 | Watermark age beyond which an expired export is refreshed rather than diffed |
| `RWPYTOOLS_LIVE_ONLY_WINDOW_DAYS` | 30 | With nothing cached, ranges starting this recently go live-only |
| `RWPYTOOLS_RATE_LIMIT_WAIT` | 20 | Seconds per retry after a 429 without `Retry-After` |
| `RWPYTOOLS_MAX_RETRIES` | 3 | Retry attempts |
| `RWPYTOOLS_DOWNLOAD_CONCURRENCY` | 8 | Parallel downloads / live requests |
| `RWPYTOOLS_REQUEST_TIMEOUT` | 30 | Umbrella HTTP timeout (per-phase overrides exist) |
| `RWPYTOOLS_NO_PROMPT` | — | Set to disable the interactive key prompt |

```python
client = rwpytools.Client(
    "...",
    cache_dir="/fast/disk/rw-cache",
    cache_max_bytes=100 * 1024**3,
    bulk_ttl=6 * 3600,
    max_retries=5,
)
```

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest          # mocked HTTP only; any real network call fails the test
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
.venv/bin/mypy                      # strict, with pandas-stubs
```

## Releasing

`.github/workflows/release.yml` publishes with PyPI trusted publishing (OIDC — no
tokens stored):

* every merge to `main` → TestPyPI as `<version>.dev<run_number>`;
* a `v*` tag → PyPI (a pre-release tag such as `v0.2.0rc1` goes to TestPyPI).

To release: bump `__version__` in `src/rwpytools/_version.py` via a PR, then
`git tag v0.1.0 && git push origin v0.1.0` and approve the `pypi` environment.
The one-time PyPI / TestPyPI setup is described at the top of `release.yml`.

## License

MIT. See [LICENSE](LICENSE).
