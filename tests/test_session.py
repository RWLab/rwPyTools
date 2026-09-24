"""End-to-end routing: cache → bulk download → live diff → one result.

Everything is mocked with respx; the clock is pinned so watermarks and
windows are deterministic.
"""

from __future__ import annotations

import datetime as dt
import io
from collections.abc import Iterator
from typing import Any

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.feather
import pyarrow.parquet
import pytest
import respx

from rwpytools import Client, ClientConfig
from rwpytools.errors import RoutingError, ServerError

FAKE_API = "https://fake-rw-api.test"
TODAY = dt.date(2026, 9, 24)


@pytest.fixture
def client(config: ClientConfig) -> Iterator[Client]:
    instance = Client(
        config=config.with_overrides(
            retry_backoff_base=0.001, retry_backoff_max=0.002, rate_limit_wait=0.0
        )
    )
    instance._stack.session._today = lambda: TODAY
    try:
        yield instance
    finally:
        instance.close()


def _signed(path: str) -> str:
    return f"https://cdn.test/{path}?Expires=1&KeyName=k&Signature=z"


def _mock_file(
    router: respx.MockRouter,
    *,
    namespace: str,
    dataset: str,
    schema: str,
    bucket: str,
    obj: str,
    payload: bytes,
    fmt: str,
) -> respx.Route:
    def _respond(request: httpx.Request, **_: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "namespace": namespace,
                    "dataset": dataset,
                    "schema": schema,
                    "symbol": request.url.params.get("symbol"),
                    "format": fmt,
                    "bucket": bucket,
                    "object": obj,
                    "size": len(payload),
                    "expires_in": 300,
                    "url": _signed(obj),
                },
            },
        )

    router.get(_signed(obj)).respond(content=payload)
    return router.get(f"/v1/{namespace}/file").mock(side_effect=_respond)


def _page(rows: list[dict[str, Any]], key: str = "data") -> dict[str, Any]:
    return {
        "success": True,
        key: rows,
        "count": len(rows),
        "pagination": {"has_more": False, "next_cursor": None, "limit": 1000},
    }


# ---- vix: CSV export, string dates, restricted symbol universe ------------------------

_VIX_CSV = (
    b"ticker,date,open,high,low,close\n"
    b"VIX,2026-09-19,15.0,16.0,14.0,15.5\n"
    b"VIX3M,2026-09-19,18.0,19.0,17.0,18.5\n"
    b"VIX,2026-09-20,15.5,16.5,15.0,16.0\n"
    b"VIX3M,2026-09-20,18.5,19.5,18.0,19.0\n"
)

_VIX_LIVE = [
    # seam day, already in the export
    {"ticker": "VIX", "date": "2026-09-20", "open": 15.5, "high": 16.5, "low": 15.0, "close": 16.0},
    {"ticker": "VIX", "date": "2026-09-21", "open": 16.0, "high": 17.0, "low": 15.5, "close": 16.5},
    {
        "ticker": "VIX3M",
        "date": "2026-09-21",
        "open": 19.0,
        "high": 20.0,
        "low": 18.5,
        "close": 19.5,
    },
    # a series the live table carries but the export does not
    {
        "ticker": "SPVIXSTR",
        "date": "2026-09-21",
        "open": None,
        "high": None,
        "low": None,
        "close": 5000.0,
    },
]


def _mock_vix(
    router: respx.MockRouter, live_rows: list[dict[str, Any]] | None = None
) -> tuple[respx.Route, respx.Route]:
    files = _mock_file(
        router,
        namespace="macro",
        dataset="vix",
        schema="ohlcv-1d",
        bucket="macro_research_pod",
        obj="vix_vix3m.csv",
        payload=_VIX_CSV,
        fmt="csv",
    )
    live = router.get("/v1/macro/vix").respond(
        json=_page(_VIX_LIVE if live_rows is None else live_rows)
    )
    return files, live


@respx.mock(base_url=FAKE_API)
def test_download_then_top_up_live(respx_mock: respx.MockRouter, client: Client) -> None:
    files, live = _mock_vix(respx_mock)

    result = client.get("vix")

    assert files.call_count == 1
    assert live.calls.last.request.url.params["gte"] == "2026-09-20"
    assert result.sources == ("bulk", "live")
    assert result.route.split_at == dt.date(2026, 9, 20)
    assert result.served_from_cache is False
    assert result.watermark == pd.Timestamp("2026-09-20")
    assert (result.bulk_rows, result.live_rows) == (4, 2)
    df = result.df
    # seam row de-duplicated, SPVIXSTR dropped, export's column order and
    # string date representation kept
    assert list(df.columns) == ["ticker", "date", "open", "high", "low", "close"]
    assert df["date"].tolist()[-2:] == ["2026-09-21", "2026-09-21"]
    assert set(df["ticker"]) == {"VIX", "VIX3M"}
    assert len(df) == 6
    assert "bulk downloaded" in result.describe_route()


@respx.mock(base_url=FAKE_API)
def test_cached_export_is_used_without_any_api_call(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    files, live = _mock_vix(respx_mock)
    client.get("vix")
    assert files.call_count == 1

    again = client.get("vix")

    # The signed-URL endpoint charges bandwidth even on a cache hit, so a
    # cached export must never touch it.
    assert files.call_count == 1
    assert live.call_count == 2
    assert again.served_from_cache is True
    assert [o.api_called for o in again.bulk_objects] == [False]
    assert len(again) == 6


@respx.mock(base_url=FAKE_API, assert_all_called=False)
def test_source_bulk_never_calls_live(respx_mock: respx.MockRouter, client: Client) -> None:
    _files, live = _mock_vix(respx_mock)
    result = client.get("vix", source="bulk")
    assert not live.called
    assert result.sources == ("bulk",)
    assert result.route.forced
    assert len(result) == 4


@respx.mock(base_url=FAKE_API, assert_all_called=False)
def test_source_live_never_downloads(respx_mock: respx.MockRouter, client: Client) -> None:
    files, live = _mock_vix(respx_mock)
    result = client.get("vix", start="2026-09-01", source="live")
    assert not files.called
    assert live.calls.last.request.url.params["gte"] == "2026-09-01"
    assert result.sources == ("live",)
    # Nothing cached to conform against: live columns, parsed dates, and
    # every live series (no export universe to restrict to).
    assert pd.api.types.is_datetime64_any_dtype(result.df["date"])
    assert "SPVIXSTR" in set(result.df["ticker"])


@respx.mock(base_url=FAKE_API, assert_all_called=False)
def test_source_live_conforms_to_a_cached_export(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    _mock_vix(respx_mock)
    client.get("vix", source="bulk")
    result = client.get("vix", start="2026-09-21", source="live")
    assert list(result.df.columns) == ["ticker", "date", "open", "high", "low", "close"]
    assert result.df["date"].iloc[0] == "2026-09-21"


@respx.mock(base_url=FAKE_API, assert_all_called=False)
def test_recent_range_with_nothing_cached_skips_the_download(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    files, live = _mock_vix(respx_mock)
    result = client.get("vix", start="2026-09-21")
    assert not files.called
    assert live.called
    assert result.sources == ("live",)


@respx.mock(base_url=FAKE_API)
def test_symbols_filter_both_halves(respx_mock: respx.MockRouter, client: Client) -> None:
    _files, live = _mock_vix(respx_mock)
    result = client.get("vix", symbols=["VIX"])
    assert live.calls.last.request.url.params["ticker"] == "VIX"
    assert set(result.df["ticker"]) == {"VIX"}
    assert len(result) == 3


@respx.mock(base_url=FAKE_API)
def test_date_range_filters_the_bulk_half(respx_mock: respx.MockRouter, client: Client) -> None:
    _mock_vix(respx_mock)
    client.get("vix", source="bulk")
    result = client.get("vix", start="2026-09-20", end="2026-09-20")
    # end == watermark: straddle with a one-day live window; the seam row
    # comes back from both sides and is kept once.
    assert result.df["date"].unique().tolist() == ["2026-09-20"]
    assert len(result) == 2


@respx.mock(base_url=FAKE_API)
def test_force_refresh_redownloads(respx_mock: respx.MockRouter, client: Client) -> None:
    files, _live = _mock_vix(respx_mock)
    client.get("vix")
    result = client.get("vix", force_refresh=True)
    assert files.call_count == 2
    assert result.served_from_cache is False


@respx.mock(base_url=FAKE_API)
def test_live_failure_degrades_to_the_cached_export(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    _mock_file(
        respx_mock,
        namespace="macro",
        dataset="vix",
        schema="ohlcv-1d",
        bucket="macro_research_pod",
        obj="vix_vix3m.csv",
        payload=_VIX_CSV,
        fmt="csv",
    )
    respx_mock.get("/v1/macro/vix").respond(
        500, json={"success": False, "message": "Failed to fetch data from the database"}
    )
    result = client.get("vix")
    assert len(result) == 4
    assert result.warnings and "Failed to fetch" in result.warnings[0]
    assert "WARNING" in result.describe_route()

    with pytest.raises(ServerError):
        client.get("vix", source="live")


@respx.mock(base_url=FAKE_API, assert_all_called=False)
def test_explain_reads_only_the_cache(respx_mock: respx.MockRouter, client: Client) -> None:
    files, live = _mock_vix(respx_mock)
    plan = client.explain("vix", start="2020-01-01")
    assert plan.bulk_action is not None and plan.bulk_action.value == "download"
    assert not files.called and not live.called

    client.get("vix", source="bulk")
    plan = client.explain("vix", start="2020-01-01")
    assert plan.sources == ("bulk", "live")
    assert plan.watermark == dt.date(2026, 9, 20)
    assert files.call_count == 1


def test_unknown_dataset_suggests_a_name(client: Client) -> None:
    with pytest.raises(RoutingError, match="did you mean 'vix'"):
        client.get("vixx")


def test_per_symbol_dataset_requires_symbols(client: Client) -> None:
    with pytest.raises(RoutingError, match="symbols="):
        client.get("fx_daily")


# ---- binance: feather export, renamed columns, intraday seam -----------------------


def _binance_feather() -> bytes:
    stamps = pd.to_datetime(["2026-09-24 06:00", "2026-09-24 07:00", "2026-09-24 06:00"]).as_unit(
        "us"
    )
    table = pa.table(
        {
            "Ticker": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
            "Datetime": pa.array(stamps.to_pydatetime(), pa.timestamp("us")),
            "Open": [1.0, 2.0, 3.0],
            "High": [1.0, 2.0, 3.0],
            "Low": [1.0, 2.0, 3.0],
            "Close": [1.0, 2.0, 3.0],
            "Volume": [1.0, 2.0, 3.0],
            "Number of trades": pa.array([10, 20, 30], pa.int64()),
        }
    )
    buf = io.BytesIO()
    pyarrow.feather.write_feather(table, buf)
    return buf.getvalue()


@respx.mock(base_url=FAKE_API)
def test_binance_live_rows_are_renamed_and_typed_like_the_export(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    _mock_file(
        respx_mock,
        namespace="crypto",
        dataset="binance.spot",
        schema="ohlcv-1h",
        bucket="crypto_research_pod",
        obj="binance_spot_1h.feather",
        payload=_binance_feather(),
        fmt="feather",
    )
    live = respx_mock.get("/v1/crypto/binance/spot").respond(
        json=_page(
            [
                # already in the export (same hour) — dropped
                {
                    "ticker": "BTCUSDT",
                    "date": "2026-09-24T07:00:00",
                    "open": 2.0,
                    "high": 2.0,
                    "low": 2.0,
                    "close": 2.0,
                    "volume": 2.0,
                },
                # ETHUSDT lags BTCUSDT in the export; its 07:00 bar is new
                {
                    "ticker": "ETHUSDT",
                    "date": "2026-09-24T07:00:00",
                    "open": 4.0,
                    "high": 4.0,
                    "low": 4.0,
                    "close": 4.0,
                    "volume": 4.0,
                },
                {
                    "ticker": "BTCUSDT",
                    "date": "2026-09-24T08:00:00",
                    "open": 5.0,
                    "high": 5.0,
                    "low": 5.0,
                    "close": 5.0,
                    "volume": 5.0,
                },
            ]
        )
    )

    result = client.get("binance_spot_1h")

    assert live.calls.last.request.url.params["gte"] == "2026-09-24"
    df = result.df
    assert list(df.columns) == [
        "Ticker",
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Number of trades",
    ]
    assert df["Datetime"].dtype == "datetime64[us]"
    assert result.live_rows == 2
    new = df.iloc[3:]
    assert new["Datetime"].tolist() == [
        pd.Timestamp("2026-09-24 07:00"),
        pd.Timestamp("2026-09-24 08:00"),
    ]
    assert new["Ticker"].tolist() == ["ETHUSDT", "BTCUSDT"]
    # export-only columns are missing on live rows
    assert new["Number of trades"].isna().all()


@respx.mock(base_url=FAKE_API)
def test_funding_rows_get_the_epoch_column_rebuilt(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    # The export carries Binance's raw millisecond jitter.
    stamps = pd.to_datetime(["2026-09-20 00:00:00.003"]).as_unit("us")
    table = pa.table(
        {
            "Ticker": ["BTCUSDT"],
            "fundingTime": pa.array([int(stamps[0].value // 1000)], pa.int64()),
            "fundingRate": [0.0001],
            "markPrice": [60000.0],
            "fundingTimeHR": pa.array(stamps.to_pydatetime(), pa.timestamp("us")),
        }
    )
    buf = io.BytesIO()
    pyarrow.feather.write_feather(table, buf)
    _mock_file(
        respx_mock,
        namespace="crypto",
        dataset="binance.perps",
        schema="funding",
        bucket="crypto_research_pod",
        obj="binance_perps_funding.feather",
        payload=buf.getvalue(),
        fmt="feather",
    )
    respx_mock.get("/v1/crypto/binance/perps/funding").respond(
        json=_page(
            [
                # the export's jittered 00:00:00.003 event, reported on the hour
                {
                    "ticker": "BTCUSDT",
                    "funding_time": "2026-09-20T00:00:00",
                    "funding_rate": 0.0001,
                    "mark_price": 60000.0,
                },
                {
                    "ticker": "BTCUSDT",
                    "funding_time": "2026-09-20T08:00:00",
                    "funding_rate": 0.0002,
                    "mark_price": 61000.0,
                },
            ]
        )
    )
    df = client.get("binance_perps_funding").df
    # the jittered seam event is not duplicated
    assert len(df) == 2
    assert df["fundingTimeHR"].iloc[0] == pd.Timestamp("2026-09-20 00:00:00.003")
    last = df.iloc[-1]
    assert last["fundingRate"] == 0.0002
    assert last["markPrice"] == 61000.0
    assert last["fundingTime"] == pd.Timestamp("2026-09-20 08:00").value // 1_000_000
    assert df["fundingTime"].dtype == "int64"


# ---- statarb spreads: top-spreads route returns "rows", multi-column key ------------


@respx.mock(base_url=FAKE_API)
def test_top_spreads_rows_merge_on_pair_key(respx_mock: respx.MockRouter, client: Client) -> None:
    table = pa.table(
        {
            "date": pa.array([dt.date(2026, 9, 22)] * 2, pa.date32()),
            "ticker": ["AEE", "COLB"],
            "stock2": ["OGE", "WTFC"],
            "zscore": [0.1, 0.2],
            "alpha": [0.01, 0.02],
        }
    )
    buf = io.BytesIO()
    pyarrow.feather.write_feather(table, buf)
    _mock_file(
        respx_mock,
        namespace="equities",
        dataset="statarb",
        schema="spreads",
        bucket="equity_factors_research_pod",
        obj="statarb/spreads.feather",
        payload=buf.getvalue(),
        fmt="feather",
    )
    route = respx_mock.get("/v1/equities/statarb/daily/topspreads").respond(
        json={
            "success": True,
            "count": 3,
            "rows": [
                {
                    "date": "2026-09-22",
                    "ticker": "AEE",
                    "stock2": "OGE",
                    "zscore": 0.1,
                    "close1": 1.0,
                },
                {
                    "date": "2026-09-23",
                    "ticker": "AEE",
                    "stock2": "OGE",
                    "zscore": 0.3,
                    "close1": 1.0,
                },
                {
                    "date": "2026-09-23",
                    "ticker": "AEE",
                    "stock2": "XEL",
                    "zscore": 0.4,
                    "close1": 1.0,
                },
            ],
        }
    )
    result = client.get("statarb_spreads")
    params = route.calls.last.request.url.params
    assert (params["gte"], params["lte"], params["limit"]) == ("2026-09-22", "2026-09-24", "50000")
    df = result.df
    assert list(df.columns) == ["date", "ticker", "stock2", "zscore", "alpha"]
    assert result.live_rows == 2
    assert df["date"].iloc[-1] == dt.date(2026, 9, 23)
    assert pd.isna(df["alpha"].iloc[-1])


# ---- earnings: trailing-window route ----------------------------------------------------


@respx.mock(base_url=FAKE_API)
def test_earnings_lookback_covers_the_gap(respx_mock: respx.MockRouter, client: Client) -> None:
    table = pa.table(
        {
            "event_key": ["A_2026_Q3"],
            "symbol": ["A"],
            "report_date": pa.array([dt.date(2026, 9, 14)], pa.date32()),
            "eps_actual": [1.0],
        }
    )
    buf = io.BytesIO()
    pyarrow.feather.write_feather(table, buf)
    _mock_file(
        respx_mock,
        namespace="equities",
        dataset="earnings",
        schema="surprises",
        bucket="equity_factors_research_pod",
        obj="earnings/surprises.feather",
        payload=buf.getvalue(),
        fmt="feather",
    )
    route = respx_mock.get("/v1/equities/earnings/recent-surprises").respond(
        json={
            "success": True,
            "data": [
                {
                    "event_key": "A_2026_Q3",
                    "symbol": "A",
                    "report_date": "2026-09-14",
                    "eps_actual": 1.0,
                },
                {
                    "event_key": "B_2026_Q3",
                    "symbol": "B",
                    "report_date": "2026-09-20",
                    "eps_actual": 2.0,
                },
                # inside the lookback window but before the watermark
                {
                    "event_key": "C_2026_Q3",
                    "symbol": "C",
                    "report_date": "2026-09-01",
                    "eps_actual": 3.0,
                },
            ],
        }
    )
    result = client.get("earnings_surprises")
    # 2026-09-14 .. 2026-09-24 inclusive
    assert route.calls.last.request.url.params["lookback_days"] == "11"
    assert result.df["event_key"].tolist() == ["A_2026_Q3", "B_2026_Q3"]


# ---- per-symbol and per-year exports --------------------------------------------------------


@respx.mock(base_url=FAKE_API, assert_all_called=False)
def test_download_selects_a_templated_file(respx_mock: respx.MockRouter, client: Client) -> None:
    files = _mock_file(
        respx_mock,
        namespace="macro",
        dataset="ib.shortstock",
        schema="yearly",
        bucket="macro_research_pod",
        obj="ib_shortstock_2025.csv.gzip",
        payload=b"",
        fmt="csv.gz",
    )
    with pytest.raises(RoutingError, match="per year"):
        client.download("ib_shortstock")
    fetched = client.download("ib_shortstock", symbol="2025")
    assert files.calls.last.request.url.params["symbol"] == "2025"
    assert fetched.symbol == "2025"


def test_year_template_cells(client: Client) -> None:
    session = client._stack.session
    from rwpytools.datasets import get_spec

    spec = get_spec("ib_shortstock")
    assert session._cells(spec, None, None, [], TODAY) == ["2026"]
    assert session._cells(spec, dt.date(2024, 5, 1), None, [], TODAY) == ["2024", "2025", "2026"]
    assert session._cells(spec, None, dt.date(2023, 12, 31), [], TODAY) == ["2023"]


# ---- seam refresh ----------------------------------------------------------------------


@respx.mock(base_url=FAKE_API)
def test_live_values_refresh_a_provisional_seam_day(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    """An export's newest day can be provisional (half-day volume); the live
    route's settled values win, and export-only columns are kept."""

    table = pa.table(
        {
            "ticker": ["HBM", "HBM"],
            "date": pa.array([dt.date(2026, 9, 21), dt.date(2026, 9, 22)], pa.date32()),
            "close": [27.0, 27.5],
            "volume": [4_000_000.0, 3_176_000.0],
            "closeunadj": [27.0, 27.5],
        }
    )
    buf = io.BytesIO()
    pyarrow.feather.write_feather(table, buf)
    _mock_file(
        respx_mock,
        namespace="equities",
        dataset="statarb",
        schema="prices",
        bucket="equity_factors_research_pod",
        obj="statarb/prices.feather",
        payload=buf.getvalue(),
        fmt="feather",
    )
    respx_mock.get("/v1/equities/statarb/prices").respond(
        json=_page(
            [
                {"ticker": "HBM", "date": "2026-09-22", "close": 27.74, "volume": 5_836_000.0},
                {"ticker": "HBM", "date": "2026-09-23", "close": 28.0, "volume": 1_000_000.0},
            ]
        )
    )
    result = client.get("statarb_prices")
    df = result.df
    assert (result.live_rows, result.seam_updates) == (1, 1)
    seam = df[df["date"] == dt.date(2026, 9, 22)].iloc[0]
    assert (seam["close"], seam["volume"], seam["closeunadj"]) == (27.74, 5_836_000.0, 27.5)
    assert df["close"].tolist() == [27.0, 27.74, 28.0]
    assert "1 seam rows refreshed from live" in result.describe_route()


@respx.mock(base_url=FAKE_API, assert_all_called=False)
def test_snapshot_export_is_served_from_cache_without_replanning_a_download(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    table = pa.table({"ticker": ["AAPL", "MSFT"], "sector": ["Technology", "Technology"]})
    buf = io.BytesIO()
    pyarrow.parquet.write_table(table, buf)
    files = _mock_file(
        respx_mock,
        namespace="equities",
        dataset="tickers",
        schema="snapshot",
        bucket="equity_factors_research_pod",
        obj="tickers/snapshot.parquet",
        payload=buf.getvalue(),
        fmt="parquet",
    )
    live = respx_mock.get("/v1/equities/tickers")
    client.get("equity_tickers")
    again = client.get("equity_tickers")
    assert files.call_count == 1
    assert not live.called
    assert again.route.bulk_action is not None and again.route.bulk_action.value == "cache"
    assert "no date watermark" in again.route.reason
