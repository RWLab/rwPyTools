"""HTTP error mapping and run_sync helper."""

from __future__ import annotations

import httpx
import pytest

from rwpytools.errors import (
    AuthenticationError,
    AuthorizationError,
    BandwidthExceededError,
    NotFoundError,
    RateLimitError,
    RwApiError,
    ServerError,
)
from rwpytools.http import (
    _map_rw_api_error,
    _retry_delay,
    run_sync,
    sanitize_url,
)


def _resp(status: int, body: bytes = b'{"success": false, "message": "x"}') -> httpx.Response:
    return httpx.Response(
        status_code=status, content=body, request=httpx.Request("GET", "https://x")
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, AuthenticationError),
        (403, AuthorizationError),  # default 403 body — no "bandwidth"
        (404, NotFoundError),
        (429, RateLimitError),
        (500, ServerError),
        (503, ServerError),
        (418, RwApiError),
    ],
)
def test_status_codes_map_to_typed_errors(status: int, expected: type) -> None:
    err = _map_rw_api_error(_resp(status))
    assert isinstance(err, expected)
    assert err.status_code == status


def test_403_with_bandwidth_message_maps_to_bandwidth_error() -> None:
    body = b'{"message": "Bandwidth limit of 20 GB reached. Please contact support."}'
    err = _map_rw_api_error(_resp(403, body=body))
    assert isinstance(err, BandwidthExceededError)
    assert err.limit_gb == 20.0


def test_403_without_bandwidth_message_maps_to_authorization_error() -> None:
    body = b'{"message": "Forbidden: no access to this pod"}'
    err = _map_rw_api_error(_resp(403, body=body))
    assert isinstance(err, AuthorizationError)
    assert not isinstance(err, BandwidthExceededError)


def test_429_carries_retry_after_when_supplied() -> None:
    resp = httpx.Response(
        status_code=429,
        content=b'{"message": "slow down"}',
        request=httpx.Request("GET", "https://x"),
        headers={"Retry-After": "7"},
    )
    err = _map_rw_api_error(resp)
    assert isinstance(err, RateLimitError)
    assert err.retry_after == 7.0


def test_error_carries_request_id_when_server_echoes_one() -> None:
    resp = httpx.Response(
        status_code=500,
        content=b'{"message": "boom"}',
        request=httpx.Request("GET", "https://x"),
        headers={"X-Request-ID": "abc-123"},
    )
    err = _map_rw_api_error(resp)
    assert err.request_id == "abc-123"
    assert "abc-123" in str(err)


def test_sanitize_url_strips_api_key() -> None:
    raw = "https://rw-api.example/v1/crypto/file?api_key=SECRET&dataset=binance"
    cleaned = sanitize_url(raw)
    assert "SECRET" not in cleaned
    assert "dataset=binance" in cleaned


def test_retry_delay_caps_server_retry_after() -> None:
    # Server says wait 600s; our cap is 30s — we must clamp.
    delay = _retry_delay(attempt=0, base=0.5, cap=30.0, retry_after=600.0)
    assert delay == 30.0


def test_retry_delay_uses_jittered_backoff_when_no_retry_after() -> None:
    delays = [_retry_delay(attempt=2, base=0.5, cap=30.0, retry_after=None) for _ in range(10)]
    # base*2^2 = 2.0 → jitter in [0, 2.0]
    assert all(0.0 <= d <= 2.0 for d in delays)


def test_error_message_extracted_from_json() -> None:
    err = _map_rw_api_error(_resp(401, body=b'{"message": "Invalid token!"}'))
    assert "Invalid token" in str(err)


def test_error_falls_back_to_text_when_body_not_json() -> None:
    err = _map_rw_api_error(_resp(500, body=b"plain text upstream error"))
    assert "plain text upstream error" in str(err)


def test_run_sync_runs_simple_coro() -> None:
    async def add() -> int:
        return 7

    assert run_sync(add()) == 7


def test_run_sync_works_inside_running_loop() -> None:
    """If called from inside an async context, run_sync defers to a thread."""

    async def outer() -> int:
        async def inner() -> int:
            return 42

        return run_sync(inner())

    # Run `outer` itself with run_sync (so when it calls run_sync, a loop is
    # already running on this thread).
    assert run_sync(outer()) == 42


# ---------------------------------------------------------------------------
# Retry layer integration (uses respx to intercept httpx)
# ---------------------------------------------------------------------------


import respx  # noqa: E402

from rwpytools.config import ClientConfig  # noqa: E402
from rwpytools.http import _Http  # noqa: E402


def _retry_config(tmp_path, **overrides):
    return ClientConfig(
        api_base_url="https://fake-rw-api.test",
        api_key="k",
        cache_dir=tmp_path,
        max_retries=overrides.get("max_retries", 3),
        retry_backoff_base=0.001,  # keep tests fast
        retry_backoff_max=0.01,
    )


@respx.mock(base_url="https://fake-rw-api.test")
async def test_get_json_retries_on_5xx_then_succeeds(respx_mock, tmp_path) -> None:
    cfg = _retry_config(tmp_path)
    http = _Http(cfg)
    route = respx_mock.get("/v1/crypto/datasets").mock(
        side_effect=[
            httpx.Response(503, json={"message": "upstream down"}),
            httpx.Response(503, json={"message": "upstream down"}),
            httpx.Response(200, json={"success": True, "namespace": "crypto", "datasets": []}),
        ]
    )
    body = await http.get_json("/v1/crypto/datasets")
    assert body["success"] is True
    assert route.call_count == 3


@respx.mock(base_url="https://fake-rw-api.test")
async def test_get_json_gives_up_after_max_retries(respx_mock, tmp_path) -> None:
    cfg = _retry_config(tmp_path, max_retries=2)
    http = _Http(cfg)
    respx_mock.get("/v1/crypto/datasets").mock(
        return_value=httpx.Response(503, json={"message": "still broken"})
    )
    with pytest.raises(ServerError):
        await http.get_json("/v1/crypto/datasets")


@respx.mock(base_url="https://fake-rw-api.test")
async def test_get_json_does_not_retry_404(respx_mock, tmp_path) -> None:
    cfg = _retry_config(tmp_path)
    http = _Http(cfg)
    route = respx_mock.get("/v1/crypto/datasets").mock(
        return_value=httpx.Response(404, json={"message": "no such pod"})
    )
    with pytest.raises(NotFoundError):
        await http.get_json("/v1/crypto/datasets")
    assert route.call_count == 1


@respx.mock(base_url="https://fake-rw-api.test")
async def test_get_json_honours_retry_after_on_429(respx_mock, tmp_path) -> None:
    cfg = _retry_config(tmp_path)
    http = _Http(cfg)
    respx_mock.get("/v1/crypto/datasets").mock(
        side_effect=[
            httpx.Response(429, json={"message": "slow"}, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"success": True, "namespace": "crypto", "datasets": []}),
        ]
    )
    body = await http.get_json("/v1/crypto/datasets")
    assert body["success"] is True


# ---------------------------------------------------------------------------
# ClientConfig redaction
# ---------------------------------------------------------------------------


def test_clientconfig_repr_redacts_api_key(tmp_path) -> None:
    cfg = ClientConfig(
        api_base_url="https://x.test",
        api_key="secretSAUCE123",
        cache_dir=tmp_path,
    )
    rendered = repr(cfg)
    assert "secretSAUCE123" not in rendered
    # First / last two chars are kept for at-a-glance recognition.
    assert "se" in rendered
    assert "23" in rendered


def test_clientconfig_repr_handles_short_key(tmp_path) -> None:
    cfg = ClientConfig(api_base_url="https://x.test", api_key="abcd", cache_dir=tmp_path)
    rendered = repr(cfg)
    assert "abcd" not in rendered
    assert "***" in rendered


def test_real_network_access_is_blocked() -> None:
    """Guard for the autouse fixture in conftest.py: unit tests never hit a real API."""

    import socket

    with pytest.raises(RuntimeError, match="real network access"):
        socket.getaddrinfo("api.robotwealth.com", 443)
    with pytest.raises(RuntimeError, match="real network access"):
        socket.create_connection(("api.robotwealth.com", 443))
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(RuntimeError, match="real network access"),
    ):
        sock.connect(("93.184.216.34", 443))


def test_loopback_stays_open_for_the_windows_event_loop() -> None:
    """On Windows asyncio's self-pipe is a TCP pair on 127.0.0.1 (the
    fallback socketpair). The guard must allow it, or no event loop starts."""

    import asyncio
    import socket

    left, right = socket._fallback_socketpair()  # type: ignore[attr-defined]
    try:
        left.sendall(b"x")
        assert right.recv(1) == b"x"
    finally:
        left.close()
        right.close()
    assert asyncio.run(asyncio.sleep(0, result=1)) == 1
