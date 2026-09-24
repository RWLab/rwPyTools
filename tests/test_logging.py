"""Logging-level API-key redaction."""

from __future__ import annotations

import logging

import pytest

from rwpytools._logging import RedactingFilter


def _format(record: logging.LogRecord) -> str:
    fmt = logging.Formatter("%(message)s")
    return fmt.format(record)


def test_redacting_filter_strips_api_key_in_url() -> None:
    record = logging.LogRecord(
        name="rwpytools.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="GET https://rw-api.test/v1/crypto/file?api_key=SECRET-KEY&dataset=binance.spot",
        args=(),
        exc_info=None,
    )
    RedactingFilter().filter(record)
    rendered = _format(record)
    assert "SECRET-KEY" not in rendered
    assert "api_key=***REDACTED***" in rendered
    assert "dataset=binance.spot" in rendered


def test_redacting_filter_passes_through_non_secret_messages() -> None:
    record = logging.LogRecord(
        name="rwpytools.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="cache hit: %s/%s",
        args=("crypto_research_pod_staging", "binance_spot_1h.feather"),
        exc_info=None,
    )
    assert RedactingFilter().filter(record) is True
    rendered = _format(record)
    assert "binance_spot_1h.feather" in rendered


def test_filter_handles_lazy_args_with_api_key() -> None:
    record = logging.LogRecord(
        name="rwpytools.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="downloaded from %s",
        args=("https://cdn.test/x?api_key=ANOTHER",),
        exc_info=None,
    )
    RedactingFilter().filter(record)
    rendered = _format(record)
    assert "ANOTHER" not in rendered
    assert "api_key=***REDACTED***" in rendered


def test_httpx_request_urls_are_redacted(caplog: pytest.LogCaptureFixture) -> None:
    """rw-api takes the key as a query parameter; httpx logs request URLs."""

    import rwpytools  # noqa: F401 — installs the filters

    with caplog.at_level(logging.INFO, logger="httpx"):
        logging.getLogger("httpx").info(
            'HTTP Request: GET https://api.robotwealth.com/v1/status?api_key=%s "HTTP/1.1 200 OK"',
            "sekrit-key-123",
        )
    assert "sekrit-key-123" not in caplog.text
    assert "api_key=***REDACTED***" in caplog.text
