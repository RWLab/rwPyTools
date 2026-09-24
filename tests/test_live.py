"""Live JSON endpoints: envelopes, pagination, and full endpoint coverage."""

from __future__ import annotations

import datetime as dt
import inspect
from typing import Any

import httpx
import pytest
import respx

from rwpytools import Client
from rwpytools import live as live_module
from rwpytools.datasets import DATASETS, LiveSource, LiveStyle
from rwpytools.errors import DataFormatError
from rwpytools.live import AsyncLiveClient, LiveClient

FAKE_API = "https://fake-rw-api.test"

#: Every path published at https://api.robotwealth.com/v1/docs (OpenAPI
#: ``basePath: /v1``), mapped to the LiveClient method or registry entry
#: that serves it. A new endpoint in the spec must be added here — and to
#: the client — or this list goes stale on purpose.
SPEC_PATHS: dict[str, str] = {
    "/status": "status",
    "/yolo/factors": "yolo_factors",
    "/yolo/weights": "yolo_weights",
    "/yolo/volatilities": "yolo_volatilities",
    "/yolo/historical": "yolo_historical",
    "/rpschteroids/weights": "rpschteroids_weights",
    "/equities/ib-short-stocks": "ib_short_stocks",
    "/equities/acquisitions": "acquisitions",
    "/equities/statarb/daily/topspreads": "statarb_daily_top_spreads",
    "/equities/statarb/hourly/topspreads": "statarb_hourly_top_spreads",
    "/equities/statarb/liquid/daily/topspreads": "statarb_liquid_daily_top_spreads",
    "/equities/statarb/liquid/hourly/topspreads": "statarb_liquid_hourly_top_spreads",
    "/equities/statarb/prices": "statarb_prices",
    "/equities/earnings/recent-surprises": "earnings_recent_surprises",
    "/equities/tickers": "equity_tickers",
    "/fx/daily": "forex_daily",
    "/crypto/binance/spot": "binance_spot",
    "/crypto/binance/perps": "binance_perps",
    "/crypto/binance/perps/funding": "binance_perps_funding",
    "/crypto/binance/coin-m/perps": "binance_coin_m_perps",
    "/crypto/binance/coin-m/funding": "binance_coin_m_funding",
    "/macro/vix": "macro_vix",
    "/macro/asset-classes": "macro_asset_classes",
    "/macro/futures": "macro_futures",
    "/macro/rates": "macro_rates",
}
#: The bulk-file routes, served by BulkClient / the pods.
SPEC_BULK_PATHS = {
    "/crypto/datasets",
    "/crypto/file",
    "/macro/datasets",
    "/macro/file",
    "/fx/datasets",
    "/fx/symbols",
    "/fx/file",
    "/equities/datasets",
    "/equities/file",
}


def test_every_published_live_endpoint_has_a_method() -> None:
    for path, method in SPEC_PATHS.items():
        assert callable(getattr(LiveClient, method, None)), f"{path} -> LiveClient.{method}"
        assert inspect.iscoroutinefunction(getattr(AsyncLiveClient, method)), method
        source = inspect.getsource(getattr(AsyncLiveClient, method))
        assert f'"/v1{path}"' in source, f"LiveClient.{method} does not call /v1{path}"


def test_every_registry_live_path_is_published() -> None:
    for spec in DATASETS.values():
        if spec.live is not None:
            assert spec.live.path.removeprefix("/v1") in SPEC_PATHS, spec.name


def test_every_bulk_namespace_is_published() -> None:
    namespaces = {s.bulk.namespace for s in DATASETS.values() if s.bulk is not None}
    for ns in namespaces:
        assert f"/{ns}/file" in SPEC_BULK_PATHS
        assert f"/{ns}/datasets" in SPEC_BULK_PATHS


# ---- envelopes and pagination --------------------------------------------------------


def _page(rows: list[dict[str, Any]], cursor: str | None) -> dict[str, Any]:
    return {
        "success": True,
        "data": rows,
        "pagination": {"has_more": cursor is not None, "next_cursor": cursor, "limit": 2},
    }


@respx.mock(base_url=FAKE_API)
def test_range_routes_follow_every_page(respx_mock: respx.MockRouter, client: Client) -> None:
    pages = {
        None: _page(
            [
                {"ticker": "EURUSD", "date": "2024-01-01"},
                {"ticker": "EURUSD", "date": "2024-01-02"},
            ],
            "p2",
        ),
        "p2": _page(
            [
                {"ticker": "EURUSD", "date": "2024-01-03"},
                {"ticker": "EURUSD", "date": "2024-01-04"},
            ],
            "p3",
        ),
        "p3": _page([{"ticker": "EURUSD", "date": "2024-01-05"}], None),
    }

    def _respond(request: httpx.Request, **_: Any) -> httpx.Response:
        assert request.url.params["gte"] == "2024-01-01"
        return httpx.Response(200, json=pages[request.url.params.get("cursor")])

    route = respx_mock.get("/v1/fx/daily").mock(side_effect=_respond)
    df = client.live.forex_daily(ticker="EURUSD", gte=dt.date(2024, 1, 1))
    assert len(df) == 5
    assert route.call_count == 3


@respx.mock(base_url=FAKE_API)
def test_pagination_stops_on_a_repeated_cursor(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    route = respx_mock.get("/v1/macro/vix").respond(json=_page([{"ticker": "VIX"}], "same"))
    df = client.live.macro_vix()
    assert route.call_count == 2
    assert len(df) == 2


@respx.mock(base_url=FAKE_API)
def test_top_spreads_read_the_rows_envelope(respx_mock: respx.MockRouter, client: Client) -> None:
    """These routes put rows under ``rows``, not ``data`` — reading ``data``
    silently returned an empty frame."""

    for path in (
        "/v1/equities/statarb/daily/topspreads",
        "/v1/equities/statarb/hourly/topspreads",
        "/v1/equities/statarb/liquid/daily/topspreads",
        "/v1/equities/statarb/liquid/hourly/topspreads",
    ):
        respx_mock.get(path).respond(
            json={"success": True, "count": 1, "rows": [{"ticker": "AEE", "stock2": "OGE"}]}
        )
    assert len(client.live.statarb_daily_top_spreads(limit=5)) == 1
    assert len(client.live.statarb_hourly_top_spreads()) == 1
    assert len(client.live.statarb_liquid_daily_top_spreads()) == 1
    assert len(client.live.statarb_liquid_hourly_top_spreads()) == 1


@respx.mock(base_url=FAKE_API)
def test_earnings_sends_lookback_and_tickers(respx_mock: respx.MockRouter, client: Client) -> None:
    route = respx_mock.get("/v1/equities/earnings/recent-surprises").respond(
        json={"success": True, "count": 0, "data": [], "meta": {}}
    )
    client.live.earnings_recent_surprises(lookback_days=5, tickers=["AAPL", "MSFT"])
    params = route.calls.last.request.url.params
    assert params["lookback_days"] == "5"
    assert params["tickers"] == "AAPL,MSFT"


@respx.mock(base_url=FAKE_API)
def test_equity_tickers_joins_a_list(respx_mock: respx.MockRouter, client: Client) -> None:
    route = respx_mock.get("/v1/equities/tickers").respond(json=_page([{"ticker": "AAPL"}], None))
    client.live.equity_tickers(tickers=["AAPL", "MSFT"])
    assert route.calls.last.request.url.params["tickers"] == "AAPL,MSFT"


@respx.mock(base_url=FAKE_API)
def test_status(respx_mock: respx.MockRouter, client: Client) -> None:
    respx_mock.get("/v1/status").respond(json={"success": True, "time": "now"})
    assert client.status() == {"success": True, "time": "now"}


def test_bad_date_is_rejected_before_any_request(client: Client) -> None:
    from rwpytools.errors import RoutingError

    with pytest.raises(RoutingError):
        client.live.forex_daily(gte="last tuesday")


def test_rows_rejects_unknown_shapes() -> None:
    with pytest.raises(DataFormatError):
        live_module._rows([1, 2])
    with pytest.raises(DataFormatError):
        live_module._rows({"data": "nope"})
    assert live_module._rows({"data": {"a": 1}}) == [{"a": 1}]
    assert live_module._rows({"success": True}) == []


# ---- generic fetch_rows used by the router ----------------------------------------------


@respx.mock(base_url=FAKE_API)
async def test_topspreads_halves_the_range_when_the_limit_fills(
    respx_mock: respx.MockRouter, client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live_module, "TOPSPREADS_MAX_LIMIT", 2)

    def _respond(request: httpx.Request, **_: Any) -> httpx.Response:
        gte = dt.date.fromisoformat(request.url.params["gte"])
        lte = dt.date.fromisoformat(request.url.params["lte"])
        days = [gte + dt.timedelta(days=i) for i in range((lte - gte).days + 1)]
        rows = [{"date": d.isoformat(), "ticker": "A", "stock2": "B"} for d in days][:2]
        return httpx.Response(200, json={"success": True, "count": len(rows), "rows": rows})

    route = respx_mock.get("/v1/equities/statarb/daily/topspreads").mock(side_effect=_respond)
    source = LiveSource("/v1/equities/statarb/daily/topspreads", style=LiveStyle.TOPSPREADS)
    rows = await client._stack.live.fetch_rows(
        source, gte=dt.date(2026, 9, 1), lte=dt.date(2026, 9, 4), today=dt.date(2026, 9, 24)
    )
    assert sorted(r["date"] for r in rows) == [
        "2026-09-01",
        "2026-09-02",
        "2026-09-03",
        "2026-09-04",
    ]
    assert route.call_count == 7  # 4-day range -> 2+2 -> 1+1+1+1


@respx.mock(base_url=FAKE_API)
async def test_lookback_is_clamped_to_the_server_maximum(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    route = respx_mock.get("/v1/equities/earnings/recent-surprises").respond(
        json={"success": True, "data": []}
    )
    source = LiveSource(
        "/v1/equities/earnings/recent-surprises", style=LiveStyle.LOOKBACK, symbol_param="tickers"
    )
    await client._stack.live.fetch_rows(
        source, gte=dt.date(2025, 1, 1), today=dt.date(2026, 9, 24), symbols=["A"] * 150
    )
    assert [c.request.url.params["lookback_days"] for c in route.calls] == ["90", "90"]
    # 150 tickers -> chunks of 100 and 50
    assert [len(c.request.url.params["tickers"].split(",")) for c in route.calls] == [100, 50]


@respx.mock(base_url=FAKE_API)
async def test_range_with_few_symbols_fans_out_per_symbol(
    respx_mock: respx.MockRouter, client: Client
) -> None:
    route = respx_mock.get("/v1/crypto/binance/spot").respond(json=_page([], None))
    source = LiveSource("/v1/crypto/binance/spot", symbol_param="ticker")
    await client._stack.live.fetch_rows(
        source, gte=dt.date(2026, 9, 20), symbols=["BTCUSDT", "ETHUSDT"]
    )
    assert sorted(c.request.url.params["ticker"] for c in route.calls) == ["BTCUSDT", "ETHUSDT"]

    many = [f"T{i}USDT" for i in range(live_module.PER_SYMBOL_FANOUT_MAX + 1)]
    await client._stack.live.fetch_rows(source, gte=dt.date(2026, 9, 20), symbols=many)
    assert "ticker" not in route.calls.last.request.url.params
