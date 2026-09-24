"""Shared fixtures for rwpytools tests.

Most tests stub the network layer with respx and use a temp cache dir; the
``client`` fixture below returns a fully wired sync Client pointed at a
fake API base URL so respx can intercept its requests.
"""

from __future__ import annotations

import io
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import pyarrow as pa
import pyarrow.feather
import pytest

from rwpytools import Client, ClientConfig

_FAKE_API_BASE = "https://fake-rw-api.test"


class RealNetworkAccessError(RuntimeError):
    """A test tried to open a real network connection."""


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard stop: no test may reach a real host, the rw-api included.

    respx intercepts at the httpx transport layer, above sockets, so mocked
    requests never get here. Anything that does — an unmocked route, a
    forgotten ``respx.mock`` — fails loudly instead of calling the network.
    Local sockets (asyncio's self-pipe) are still allowed.
    """

    def _blocked(*args: object, **kwargs: object) -> NoReturn:
        raise RealNetworkAccessError(f"test attempted real network access: {args!r}")

    real_connect = socket.socket.connect

    def _guarded_connect(self: socket.socket, address: object) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _blocked(address)
        real_connect(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    cache = tmp_path / "rwpytools-cache"
    cache.mkdir()
    return cache


@pytest.fixture
def fake_api_base() -> str:
    return _FAKE_API_BASE


@pytest.fixture
def config(tmp_cache_dir: Path, fake_api_base: str) -> ClientConfig:
    return ClientConfig(
        api_base_url=fake_api_base,
        api_key="test-key",
        cache_dir=tmp_cache_dir,
        cache_max_bytes=10 * 1024 * 1024,
        default_expires_in=3600,
        request_timeout=5.0,
        download_concurrency=4,
    )


@pytest.fixture
def client(config: ClientConfig) -> Iterator[Client]:
    c = Client(config=config)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def make_feather_bytes() -> callable:
    """Return a helper that builds a tiny feather payload from a table."""

    def _make(table: pa.Table) -> bytes:
        buf = io.BytesIO()
        pyarrow.feather.write_feather(table, buf)
        return buf.getvalue()

    return _make


@pytest.fixture
def csv_bytes_eurusd() -> bytes:
    return b"date,open,close\n2024-01-02,1.10,1.11\n2024-01-03,1.11,1.12\n"
