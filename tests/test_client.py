"""High-level synchronous Client tests (mocked HTTP)."""

from __future__ import annotations

import datetime as dt
import io

import pyarrow as pa
import pyarrow.feather
import pytest
import respx

from rwpytools import Client

_FAKE_API = "https://fake-rw-api.test"


def _signed_url(host: str, path: str) -> str:
    return f"https://{host}/{path}?Expires=1&KeyName=k&Signature=z"


def _feather_bytes() -> bytes:
    buf = io.BytesIO()
    pyarrow.feather.write_feather(pa.table({"close": [100.0, 101.0]}), buf)
    return buf.getvalue()


@respx.mock(base_url=_FAKE_API, assert_all_called=False)
def test_client_crypto_get_binance_spot_1h_bulk_only(respx_mock, client: Client) -> None:
    payload = _feather_bytes()
    respx_mock.get("/v1/crypto/file").respond(
        json={
            "success": True,
            "data": {
                "namespace": "crypto",
                "dataset": "binance.spot",
                "schema": "ohlcv-1h",
                "symbol": None,
                "format": "feather",
                "bucket": "crypto_research_pod_staging",
                "object": "binance_spot_1h.feather",
                "size": len(payload),
                "expires_in": 3600,
                "url": _signed_url("cdn.test", "crypto/binance_spot_1h.feather"),
            },
        }
    )
    respx_mock.get(_signed_url("cdn.test", "crypto/binance_spot_1h.feather")).respond(
        content=payload
    )
    live = respx_mock.get("/v1/crypto/binance/spot")

    df = client.crypto.get_binance_spot_1h(source="bulk")
    assert df.shape == (2, 1)
    # source="bulk" never touches the live companion.
    assert not live.called


@respx.mock(base_url=_FAKE_API)
def test_client_fx_bulk_keys_by_ticker(respx_mock, client: Client) -> None:
    import httpx

    def _feather(ticker: str) -> bytes:
        buf = io.BytesIO()
        table = pa.table(
            {
                "Date": pa.array([dt.date(2024, 1, 2), dt.date(2024, 1, 3)], pa.date32()),
                "open": [1.0, 1.1],
                "high": [1.2, 1.3],
                "low": [0.9, 1.0],
                "close": [1.1, 1.2],
                "volume": pa.array([10, 20], pa.int32()),
                "ticker": [ticker, ticker],
            }
        )
        pyarrow.feather.write_feather(table, buf)
        return buf.getvalue()

    payloads = {t: _feather(t) for t in ("EURUSD", "GBPUSD")}

    def _file_route(request: httpx.Request, **_kwargs: object) -> httpx.Response:
        symbol = request.url.params["symbol"]
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "namespace": "fx",
                    "dataset": "pairs",
                    "schema": "ohlcv-1d",
                    "symbol": symbol,
                    "format": "feather",
                    "bucket": "fx_research_pod",
                    "object": f"feather/Daily/{symbol}.feather",
                    "size": len(payloads[symbol]),
                    "expires_in": 300,
                    "url": _signed_url("cdn.test", f"fx/feather/Daily/{symbol}.feather"),
                },
            },
        )

    files = respx_mock.get("/v1/fx/file").mock(side_effect=_file_route)
    for ticker, payload in payloads.items():
        respx_mock.get(_signed_url("cdn.test", f"fx/feather/Daily/{ticker}.feather")).respond(
            content=payload
        )
    # The live companion adds one new EURUSD bar after the watermark and
    # repeats the seam day (dropped as a duplicate).
    live_rows = [
        {
            "date": "2024-01-03",
            "ticker": "EURUSD",
            "open": 1.0,
            "high": 1.3,
            "low": 1.0,
            "close": 1.2,
            "volume": 20.0,
        },
        {
            "date": "2024-01-04",
            "ticker": "EURUSD",
            "open": 1.2,
            "high": 1.4,
            "low": 1.1,
            "close": 1.3,
            "volume": 30.0,
        },
    ]

    def _live_route(request: httpx.Request, **_kwargs: object) -> httpx.Response:
        ticker = request.url.params.get("ticker")
        rows = [r for r in live_rows if ticker in (None, r["ticker"])]
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": rows,
                "pagination": {"has_more": False, "next_cursor": None},
            },
        )

    live = respx_mock.get("/v1/fx/daily").mock(side_effect=_live_route)

    # rwRtools-1:1 shape: concatenated DataFrame, one row block per ticker.
    frame = client.fx.get_daily_ohlc(["EURUSD", "GBPUSD"])
    assert list(frame.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    assert list(frame["Ticker"]) == ["EURUSD"] * 3 + ["GBPUSD"] * 2
    assert frame["Date"].iloc[2] == dt.date(2024, 1, 4)
    assert files.call_count == 2
    assert live.called

    # Second call: both pairs are cached — no signed URL is issued again.
    frames = client.fx.get_daily_ohlc_frames(["EURUSD", "GBPUSD"])
    assert set(frames.keys()) == {"EURUSD", "GBPUSD"}
    assert frames["EURUSD"].shape[0] == 3
    assert files.call_count == 2


@respx.mock(base_url=_FAKE_API)
def test_client_live_yolo_factors(respx_mock, client: Client) -> None:
    respx_mock.get("/v1/yolo/factors").respond(
        json={
            "success": True,
            "data": [
                {"ticker": "BTC/USD", "factor_name": "X_MOM_30", "value": 0.2},
                {"ticker": "ETH/USD", "factor_name": "X_MOM_30", "value": 0.15},
            ],
        }
    )
    df = client.live.yolo_factors()
    assert df.shape == (2, 3)
    assert sorted(df.columns) == ["factor_name", "ticker", "value"]


@respx.mock(base_url=_FAKE_API)
def test_client_live_acquisitions(respx_mock, client: Client) -> None:
    route = respx_mock.get("/v1/equities/acquisitions").respond(
        json={
            "success": True,
            "data": [
                {
                    "date": "2024-01-15",
                    "action": "acquisitionby",
                    "ticker": "ALO1",
                    "name": "ALPHARMA INC",
                    "value": 1985.7,
                    "contraticker": "KG1",
                    "contraname": "KING PHARMACEUTICALS INC",
                    "record_uploaded": "2024-01-16",
                },
            ],
            "count": 1,
        }
    )
    df = client.live.acquisitions(ticker="ALO1", gte="2024-01-01", lte="2024-12-31")
    assert df.shape == (1, 8)
    assert df.iloc[0]["ticker"] == "ALO1"
    # filters forwarded as query params
    sent = str(route.calls.last.request.url)
    assert "ticker=ALO1" in sent
    assert "gte=2024-01-01" in sent
    assert "lte=2024-12-31" in sent


def test_client_live_forex_daily(respx_mock, client: Client) -> None:
    route = respx_mock.get("/v1/fx/daily").respond(
        json={
            "success": True,
            "data": [
                {
                    "date": "2024-01-15",
                    "ticker": "AUDUSD",
                    "open": 0.6712,
                    "high": 0.6745,
                    "low": 0.6698,
                    "close": 0.6731,
                    "volume": 0.0,
                },
            ],
            "count": 1,
        }
    )
    df = client.live.forex_daily(ticker="AUDUSD", gte="2024-01-01", lte="2024-12-31")
    assert df.shape == (1, 7)
    assert df.iloc[0]["ticker"] == "AUDUSD"
    sent = str(route.calls.last.request.url)
    assert "ticker=AUDUSD" in sent
    assert "gte=2024-01-01" in sent
    assert "lte=2024-12-31" in sent


def test_client_session_context_manager(client: Client) -> None:
    with client.session() as c:
        assert c is client
    client.close()


def test_client_close_idempotent(client: Client) -> None:
    client.close()
    client.close()


# ---------------------------------------------------------------------------
# Interactive API-key prompt (Colab / Jupyter / TTY entry point)
# ---------------------------------------------------------------------------


def test_prompt_when_missing_key_and_interactive(monkeypatch, tmp_path) -> None:
    """An IPython-style session with no key triggers the masked prompt."""

    from rwpytools import client as client_mod

    monkeypatch.delenv("RWPYTOOLS_API_KEY", raising=False)
    monkeypatch.delenv("RWPYTOOLS_NO_PROMPT", raising=False)
    monkeypatch.setattr(client_mod, "_is_interactive_session", lambda: True)

    prompts: list[str] = []

    def fake_prompt() -> str:
        prompts.append("called")
        return "secret-from-prompt"

    monkeypatch.setattr(client_mod, "_prompt_for_api_key", fake_prompt)
    monkeypatch.setenv("RWPYTOOLS_CACHE_DIR", str(tmp_path))

    c = Client()
    try:
        assert prompts == ["called"]
        assert c.config.api_key == "secret-from-prompt"
    finally:
        c.close()


def test_no_prompt_when_explicit_api_key_given(monkeypatch, tmp_path) -> None:
    """Explicit api_key kwarg short-circuits even when interactive."""

    from rwpytools import client as client_mod

    monkeypatch.delenv("RWPYTOOLS_API_KEY", raising=False)
    monkeypatch.setattr(client_mod, "_is_interactive_session", lambda: True)

    def boom() -> str:
        raise AssertionError("must not prompt when api_key is explicit")

    monkeypatch.setattr(client_mod, "_prompt_for_api_key", boom)
    monkeypatch.setenv("RWPYTOOLS_CACHE_DIR", str(tmp_path))

    c = Client(api_key="explicit")
    try:
        assert c.config.api_key == "explicit"
    finally:
        c.close()


def test_no_prompt_in_non_interactive_session(monkeypatch, tmp_path) -> None:
    """CI / piped stdin must raise ConfigError instead of hanging on prompt."""

    from rwpytools import client as client_mod
    from rwpytools.errors import ConfigError

    monkeypatch.delenv("RWPYTOOLS_API_KEY", raising=False)
    monkeypatch.setattr(client_mod, "_is_interactive_session", lambda: False)

    def boom() -> str:
        raise AssertionError("must not prompt in non-interactive session")

    monkeypatch.setattr(client_mod, "_prompt_for_api_key", boom)
    monkeypatch.setenv("RWPYTOOLS_CACHE_DIR", str(tmp_path))

    with pytest.raises(ConfigError):
        Client()


def test_rwpytools_no_prompt_env_suppresses_prompt(monkeypatch, tmp_path) -> None:
    """RWPYTOOLS_NO_PROMPT=1 forces non-interactive even inside a Jupyter kernel."""

    from rwpytools import client as client_mod

    monkeypatch.delenv("RWPYTOOLS_API_KEY", raising=False)
    monkeypatch.setenv("RWPYTOOLS_NO_PROMPT", "1")

    # If suppression works, _is_interactive_session must return False here.
    assert client_mod._is_interactive_session() is False


def test_prompt_stashes_key_in_env_for_session(monkeypatch, tmp_path) -> None:
    """First prompt populates RWPYTOOLS_API_KEY so re-instantiation skips it."""

    from rwpytools import client as client_mod

    monkeypatch.delenv("RWPYTOOLS_API_KEY", raising=False)
    monkeypatch.delenv("RWPYTOOLS_NO_PROMPT", raising=False)
    monkeypatch.setattr(client_mod, "_is_interactive_session", lambda: True)
    monkeypatch.setenv("RWPYTOOLS_CACHE_DIR", str(tmp_path))

    calls = {"n": 0}

    def fake_prompt() -> str:
        calls["n"] += 1
        return "session-key"

    monkeypatch.setattr(client_mod, "_prompt_for_api_key", fake_prompt)

    c1 = Client()
    c2 = Client()
    try:
        assert calls["n"] == 1, "second instantiation must reuse env var"
        assert c1.config.api_key == c2.config.api_key == "session-key"
    finally:
        c1.close()
        c2.close()
