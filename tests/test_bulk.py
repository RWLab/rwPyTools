"""End-to-end tests for the bulk download flow.

Mocks both rw-api (for the signed-URL endpoint) and the CDN (for the
download itself) with respx, and verifies the full path: HEAD → cache
miss → download → store → re-read serves from cache.
"""

from __future__ import annotations

import io

import httpx
import pyarrow as pa
import pyarrow.feather
import pytest
import respx

from rwpytools.bulk import BulkClient, BulkFetchBatch, FetchRequest
from rwpytools.cache import FilesystemCache
from rwpytools.errors import (
    BandwidthExceededError,
    NotFoundError,
    RwApiError,
    SignedUrlExpiredError,
)
from rwpytools.http import _Http

_CDN_HOST = "https://cdn.data-staging.robotwealth.test"
_FAKE_API = "https://fake-rw-api.test"


def _signed_url(pod: str, object_name: str) -> str:
    return f"{_CDN_HOST}/{pod}/{object_name}?Expires=1&KeyName=k&Signature=abcd"


def _file_response(
    *,
    namespace: str,
    dataset: str,
    schema: str,
    symbol: str | None,
    bucket: str,
    object_name: str,
    size: int,
    pod: str,
    fmt: str = "feather",
) -> dict:
    return {
        "success": True,
        "data": {
            "namespace": namespace,
            "dataset": dataset,
            "schema": schema,
            "symbol": symbol,
            "format": fmt,
            "bucket": bucket,
            "object": object_name,
            "size": size,
            "expires_in": 3600,
            "url": _signed_url(pod, object_name),
        },
    }


def _make_csv_bytes() -> bytes:
    return b"a,b\n1,2\n3,4\n"


def _make_feather_bytes() -> bytes:
    buf = io.BytesIO()
    pyarrow.feather.write_feather(pa.table({"x": [1.0, 2.0]}), buf)
    return buf.getvalue()


@pytest.fixture
def bulk(config) -> BulkClient:
    http = _Http(config)
    cache = FilesystemCache(config.cache_dir, max_bytes=config.cache_max_bytes)
    return BulkClient(config, http=http, cache=cache)


# ---------------------------------------------------------------------------
# fetch_one
# ---------------------------------------------------------------------------


@respx.mock(base_url=_FAKE_API)
async def test_fetch_one_downloads_and_caches(respx_mock, bulk: BulkClient) -> None:
    payload = _make_csv_bytes()
    object_name = "binance_spot_production_1d_ohlc.csv"
    respx_mock.get("/v1/crypto/file").respond(
        json=_file_response(
            namespace="crypto",
            dataset="binance.spot",
            schema="ohlcv-1d",
            symbol=None,
            bucket="crypto_research_pod_staging",
            object_name=object_name,
            size=len(payload),
            pod="crypto",
            fmt="csv",
        )
    )
    respx_mock.get(_signed_url("crypto", object_name)).respond(content=payload)

    result = await bulk.fetch_one("crypto", dataset="binance.spot", schema="ohlcv-1d")

    assert result.served_from_cache is False
    assert result.namespace == "crypto"
    assert result.dataset == "binance.spot"
    assert result.schema == "ohlcv-1d"
    assert result.symbol is None
    assert result.size == len(payload)
    assert result.path.read_bytes() == payload


@respx.mock(base_url=_FAKE_API, assert_all_called=False)
async def test_fetch_one_serves_from_cache_on_size_match(respx_mock, bulk: BulkClient) -> None:
    payload = _make_csv_bytes()
    object_name = "binance_spot_production_1d_ohlc.csv"

    obj_path = bulk._cache.path_for("crypto_research_pod_staging", object_name)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_bytes(payload)
    bulk._cache.store_meta(
        bucket="crypto_research_pod_staging",
        object_name=object_name,
        size=len(payload),
    )

    respx_mock.get("/v1/crypto/file").respond(
        json=_file_response(
            namespace="crypto",
            dataset="binance.spot",
            schema="ohlcv-1d",
            symbol=None,
            bucket="crypto_research_pod_staging",
            object_name=object_name,
            size=len(payload),
            pod="crypto",
            fmt="csv",
        )
    )
    cdn_route = respx_mock.get(_signed_url("crypto", object_name)).respond(
        content=b"should-not-be-fetched"
    )

    result = await bulk.fetch_one("crypto", dataset="binance.spot", schema="ohlcv-1d")

    assert result.served_from_cache is True
    assert not cdn_route.called, "Cache hit must not fetch the CDN"


@respx.mock(base_url=_FAKE_API)
async def test_fetch_one_force_refresh_redownloads(respx_mock, bulk: BulkClient) -> None:
    object_name = "binance_perps_funding.feather"
    payload_v1 = b"old"
    payload_v2 = b"new-bytes"

    obj_path = bulk._cache.path_for("crypto_research_pod_staging", object_name)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_bytes(payload_v1)
    bulk._cache.store_meta(
        bucket="crypto_research_pod_staging",
        object_name=object_name,
        size=len(payload_v1),
    )

    respx_mock.get("/v1/crypto/file").respond(
        json=_file_response(
            namespace="crypto",
            dataset="binance.perps",
            schema="funding",
            symbol=None,
            bucket="crypto_research_pod_staging",
            object_name=object_name,
            size=len(payload_v2),
            pod="crypto",
        )
    )
    respx_mock.get(_signed_url("crypto", object_name)).respond(content=payload_v2)

    result = await bulk.fetch_one(
        "crypto",
        dataset="binance.perps",
        schema="funding",
        force_refresh=True,
    )
    assert result.served_from_cache is False
    assert obj_path.read_bytes() == payload_v2


@respx.mock(base_url=_FAKE_API)
async def test_fetch_one_propagates_404_from_api(respx_mock, bulk: BulkClient) -> None:
    respx_mock.get("/v1/crypto/file").respond(
        status_code=404,
        json={
            "success": False,
            "message": "Unknown (dataset, schema) for namespace 'crypto'",
        },
    )

    with pytest.raises(NotFoundError):
        await bulk.fetch_one("crypto", dataset="nope", schema="ohlcv-1h")


@respx.mock(base_url=_FAKE_API)
async def test_fetch_one_propagates_403_bandwidth(respx_mock, bulk: BulkClient) -> None:
    respx_mock.get("/v1/crypto/file").respond(
        status_code=403,
        json={"success": False, "message": "Bandwidth limit of 20 GB reached."},
    )

    with pytest.raises(BandwidthExceededError):
        await bulk.fetch_one("crypto", dataset="binance.spot", schema="ohlcv-1h")


@respx.mock(base_url=_FAKE_API)
async def test_fetch_one_rejects_truncated_cdn_download(respx_mock, bulk: BulkClient) -> None:
    """Server says 100 bytes, CDN sends 50 — must not cache the truncated file."""

    object_name = "binance_spot_1h.feather"
    respx_mock.get("/v1/crypto/file").respond(
        json=_file_response(
            namespace="crypto",
            dataset="binance.spot",
            schema="ohlcv-1h",
            symbol=None,
            bucket="crypto_research_pod_staging",
            object_name=object_name,
            size=100,  # advertised
            pod="crypto",
        )
    )
    respx_mock.get(_signed_url("crypto", object_name)).respond(content=b"x" * 50)

    with pytest.raises(RwApiError):
        await bulk.fetch_one("crypto", dataset="binance.spot", schema="ohlcv-1h")

    cache_path = bulk._cache.path_for("crypto_research_pod_staging", object_name)
    assert not cache_path.exists(), "truncated file must not be left in the cache"
    assert bulk._cache.lookup("crypto_research_pod_staging", object_name) is None, (
        "no sidecar should exist for a rejected download"
    )


@respx.mock(base_url=_FAKE_API)
async def test_fetch_one_rejects_content_length_mismatch(respx_mock, bulk: BulkClient) -> None:
    """If the CDN sends a Content-Length that disagrees with rw-api, refuse."""

    object_name = "binance_spot_1h.feather"
    respx_mock.get("/v1/crypto/file").respond(
        json=_file_response(
            namespace="crypto",
            dataset="binance.spot",
            schema="ohlcv-1h",
            symbol=None,
            bucket="crypto_research_pod_staging",
            object_name=object_name,
            size=100,
            pod="crypto",
        )
    )
    respx_mock.get(_signed_url("crypto", object_name)).respond(
        content=b"x" * 100,
        headers={"Content-Length": "999"},  # CDN lies about size
    )

    with pytest.raises(RwApiError):
        await bulk.fetch_one("crypto", dataset="binance.spot", schema="ohlcv-1h")


@respx.mock(base_url=_FAKE_API)
async def test_fetch_one_signed_url_expired(respx_mock, bulk: BulkClient) -> None:
    """If the CDN keeps rejecting after one re-mint, the second 403 propagates."""

    respx_mock.get("/v1/crypto/file").respond(
        json=_file_response(
            namespace="crypto",
            dataset="binance.spot",
            schema="ohlcv-1h",
            symbol=None,
            bucket="crypto_research_pod_staging",
            object_name="binance_spot_1h.feather",
            size=10,
            pod="crypto",
        )
    )
    respx_mock.get(_signed_url("crypto", "binance_spot_1h.feather")).respond(status_code=403)

    with pytest.raises(SignedUrlExpiredError):
        await bulk.fetch_one("crypto", dataset="binance.spot", schema="ohlcv-1h")


@respx.mock(base_url=_FAKE_API)
async def test_fetch_one_remints_signed_url_on_first_expiry(respx_mock, bulk: BulkClient) -> None:
    """First CDN GET returns 403 (expired URL); we re-mint and succeed."""

    payload = b"a,b\n1,2\n"
    object_name = "fresh_object.csv"
    respx_mock.get("/v1/crypto/file").respond(
        json=_file_response(
            namespace="crypto",
            dataset="binance.spot",
            schema="ohlcv-1d",
            symbol=None,
            bucket="crypto_research_pod_staging",
            object_name=object_name,
            size=len(payload),
            pod="crypto",
            fmt="csv",
        )
    )
    cdn_route = respx_mock.get(_signed_url("crypto", object_name)).mock(
        side_effect=[
            httpx.Response(403),  # expired
            httpx.Response(200, content=payload),  # re-minted URL works
        ]
    )

    result = await bulk.fetch_one("crypto", dataset="binance.spot", schema="ohlcv-1d")
    assert result.served_from_cache is False
    assert result.path.read_bytes() == payload
    assert cdn_route.call_count == 2, "must retry exactly once after re-minting"


# ---------------------------------------------------------------------------
# fetch_many — fans out one /file call per request
# ---------------------------------------------------------------------------


@respx.mock(base_url=_FAKE_API)
async def test_fetch_many_partial_collects_successes_and_failures(
    respx_mock, bulk: BulkClient
) -> None:
    """fetch_many_partial returns both succeeded and failed slots aligned with input."""

    feather = _make_feather_bytes()
    requests = [
        FetchRequest(dataset="pairs", schema="ohlcv-1d", symbol="GOOD"),
        FetchRequest(dataset="pairs", schema="ohlcv-1d", symbol="BAD"),
    ]

    def _file_route(request, **_kwargs):
        symbol = request.url.params["symbol"]
        if symbol == "BAD":
            return httpx.Response(404, json={"success": False, "message": "no symbol"})
        return httpx.Response(
            200,
            json=_file_response(
                namespace="fx",
                dataset="pairs",
                schema="ohlcv-1d",
                symbol=symbol,
                bucket="fx_research_pod_staging",
                object_name=f"feather/Daily/{symbol}.feather",
                size=len(feather),
                pod="fx",
            ),
        )

    respx_mock.get("/v1/fx/file").mock(side_effect=_file_route)
    respx_mock.get(_signed_url("fx", "feather/Daily/GOOD.feather")).respond(content=feather)

    batch = await bulk.fetch_many_partial("fx", requests)
    assert isinstance(batch, BulkFetchBatch)
    assert not batch.all_succeeded()
    assert len(batch.succeeded_results()) == 1
    assert batch.succeeded_results()[0].symbol == "GOOD"
    failures = batch.failures()
    assert len(failures) == 1
    assert failures[0][0] == 1  # second request


@respx.mock(base_url=_FAKE_API)
async def test_fetch_many_issues_one_request_per_cell(respx_mock, bulk: BulkClient) -> None:
    """FX bulk: get_daily_ohlc(["EURUSD","GBPUSD"]) → two parallel /v1/fx/file
    calls. The bulk endpoint was removed; we issue N round-trips."""

    feather = _make_feather_bytes()
    requests = [
        FetchRequest(dataset="pairs", schema="ohlcv-1d", symbol="EURUSD"),
        FetchRequest(dataset="pairs", schema="ohlcv-1d", symbol="GBPUSD"),
    ]

    def _file_route(request, **_kwargs):  # respx side_effect signature
        symbol = request.url.params["symbol"]
        return httpx.Response(
            200,
            json=_file_response(
                namespace="fx",
                dataset="pairs",
                schema="ohlcv-1d",
                symbol=symbol,
                bucket="fx_research_pod_staging",
                object_name=f"feather/Daily/{symbol}.feather",
                size=len(feather),
                pod="fx",
            ),
        )

    respx_mock.get("/v1/fx/file").mock(side_effect=_file_route)
    respx_mock.get(_signed_url("fx", "feather/Daily/EURUSD.feather")).respond(content=feather)
    respx_mock.get(_signed_url("fx", "feather/Daily/GBPUSD.feather")).respond(content=feather)

    results = await bulk.fetch_many("fx", requests)
    assert {r.symbol for r in results} == {"EURUSD", "GBPUSD"}
    for r in results:
        assert r.served_from_cache is False
        assert r.path.exists()


# ---------------------------------------------------------------------------
# fetch_one_as_frame — feather round-trip
# ---------------------------------------------------------------------------


@respx.mock(base_url=_FAKE_API)
async def test_fetch_one_as_frame_returns_dataframe(respx_mock, bulk: BulkClient) -> None:
    payload = _make_feather_bytes()
    respx_mock.get("/v1/crypto/file").respond(
        json=_file_response(
            namespace="crypto",
            dataset="binance.spot",
            schema="ohlcv-1h",
            symbol=None,
            bucket="crypto_research_pod_staging",
            object_name="binance_spot_1h.feather",
            size=len(payload),
            pod="crypto",
        )
    )
    respx_mock.get(_signed_url("crypto", "binance_spot_1h.feather")).respond(content=payload)

    df = await bulk.fetch_one_as_frame("crypto", dataset="binance.spot", schema="ohlcv-1h")
    assert list(df.columns) == ["x"]
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


@respx.mock(base_url=_FAKE_API)
async def test_list_datasets_parses_response(respx_mock, bulk: BulkClient) -> None:
    respx_mock.get("/v1/crypto/datasets").respond(
        json={
            "success": True,
            "namespace": "crypto",
            "datasets": [
                {
                    "dataset": "binance.spot",
                    "requires_symbol": False,
                    "schemas": [
                        {"schema": "ohlcv-1h", "format": "feather"},
                        {"schema": "ohlcv-1d", "format": "csv"},
                    ],
                },
            ],
        }
    )
    catalog = await bulk.list_datasets("crypto")
    assert catalog.namespace == "crypto"
    assert len(catalog.datasets) == 1
    spot = catalog.datasets[0]
    assert spot.dataset == "binance.spot"
    assert {s.schema_name for s in spot.schemas} == {"ohlcv-1h", "ohlcv-1d"}


@respx.mock(base_url=_FAKE_API)
async def test_list_symbols_parses_response(respx_mock, bulk: BulkClient) -> None:
    respx_mock.get("/v1/fx/symbols").respond(
        json={
            "success": True,
            "namespace": "fx",
            "dataset": "pairs",
            "schema": "ohlcv-1d",
            "symbols": ["AUDUSD", "EURUSD", "GBPUSD"],
            "count": 3,
        }
    )
    resp = await bulk.list_symbols("fx", dataset="pairs")
    assert resp.symbols == ["AUDUSD", "EURUSD", "GBPUSD"]
    assert resp.count == 3
