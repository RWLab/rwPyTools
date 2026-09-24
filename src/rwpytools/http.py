"""HTTP transport — a thin wrapper around httpx with rw-api error mapping.

The low-level surface is async. Synchronous callers go through
:class:`rwpytools.client.Client`, which runs each coroutine with
:func:`run_sync` (one fresh ``asyncio`` loop per call). Async callers
keep their own long-lived loop and reuse the same :class:`AsyncClient`.

**Why each call opens its own** ``httpx.AsyncClient``:

``httpx.AsyncClient`` binds its underlying transport (and the TCP
connections in its pool) to the event loop that created it. The sync
``Client`` runs every call inside a *new* loop via ``asyncio.run``,
which closes when the call returns. A long-lived AsyncClient would end
up holding sockets attached to a closed loop, and the next call —
which runs in another fresh loop — would crash with
``RuntimeError: Event loop is closed`` the moment it touched those
sockets. (We hit this exact bug repeatedly; the fix is to scope the
AsyncClient to a single coroutine and let ``async with`` clean it up.)

For pure-async callers this means a tiny per-call TCP handshake cost
instead of pooled keep-alive. The rwPyTools workload (a few sparse API
calls + one streaming CDN download per pod method) doesn't benefit
from pooling enough to be worth the cross-loop bug class.

Retries
-------

Retries are owned by this module so callers don't have to. Idempotent
requests (``GET`` / ``HEAD`` / catalog and ``/file`` URL issuance) are
retried up to :attr:`ClientConfig.max_retries` times on:

* connection errors / timeouts (``httpx.TransportError``);
* HTTP 429 — backoff honours ``Retry-After`` when the server sends it,
  and otherwise waits :attr:`ClientConfig.rate_limit_wait` seconds per
  attempt (rw-api limits per minute and omits the header);
* HTTP 5xx;

with exponential backoff and jitter so concurrent callers do not
synchronize their retries. Non-idempotent ``POST`` requests are never
auto-retried.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import ClientConfig
from .errors import (
    AuthenticationError,
    AuthorizationError,
    BandwidthExceededError,
    NotFoundError,
    RateLimitError,
    RwApiError,
    ServerError,
    SignedUrlExpiredError,
)

log = logging.getLogger(__name__)


_REQUEST_ID_HEADER = "X-Request-ID"
_BANDWIDTH_MESSAGE_NEEDLE = "bandwidth"  # lower-case fragment in rw-api's 403 body


def sanitize_url(url: str) -> str:
    """Strip the ``api_key`` query parameter from a URL for safe logging."""

    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.query:
        return url
    # Keep order so cache-key-like comparisons remain stable.
    kept = [kv for kv in parts.query.split("&") if not kv.lower().startswith("api_key=")]
    new_query = "&".join(kept)
    return urlunsplit(parts._replace(query=new_query))


def _is_bandwidth_message(message: str) -> bool:
    return _BANDWIDTH_MESSAGE_NEEDLE in (message or "").lower()


def _map_rw_api_error(response: httpx.Response) -> RwApiError:
    """Convert an rw-api JSON error response into the appropriate exception.

    The 403 path inspects the body to distinguish "bandwidth cap" from
    "you don't have permission to this resource". When the message
    mentions ``bandwidth`` we raise :class:`BandwidthExceededError`;
    other 403s become :class:`AuthorizationError`.
    """

    try:
        body = response.json()
        message = body.get("message") or body.get("error") or response.text
    except Exception:
        body = None
        message = response.text or response.reason_phrase

    status = response.status_code
    request_id = response.headers.get(_REQUEST_ID_HEADER)
    if status == 401:
        return AuthenticationError(message, status_code=status, request_id=request_id)
    if status == 403:
        if _is_bandwidth_message(str(message)):
            return BandwidthExceededError(
                message,
                status_code=status,
                request_id=request_id,
                server_body=body if isinstance(body, dict) else None,
            )
        return AuthorizationError(message, status_code=status, request_id=request_id)
    if status == 404:
        return NotFoundError(message, status_code=status, request_id=request_id)
    if status == 429:
        retry_after = response.headers.get("Retry-After")
        return RateLimitError(
            message,
            status_code=status,
            request_id=request_id,
            retry_after=_parse_retry_after(retry_after),
        )
    if 500 <= status < 600:
        return ServerError(message, status_code=status, request_id=request_id)
    return RwApiError(message, status_code=status, request_id=request_id)


def _parse_retry_after(raw: str | None) -> float | None:
    """Parse a ``Retry-After`` value; supports the int-seconds form.

    HTTP-date form is allowed by RFC 7231 but rw-api emits seconds.
    Returning ``None`` on a bad/missing header lets the retry layer
    fall back to exponential backoff.
    """

    if raw is None:
        return None
    try:
        return float(raw.strip())
    except (ValueError, AttributeError):
        return None


class _Http:
    """Async HTTP transport.

    Each public coroutine opens its own ``httpx.AsyncClient`` inside
    ``async with``. There is intentionally no long-lived client cached
    on the instance — see the module docstring for why.
    """

    def __init__(self, config: ClientConfig) -> None:
        self._config = config

    # ---- client construction ---------------------------------------------

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self._config.connect_timeout,
            read=self._config.read_timeout,
            write=self._config.write_timeout,
            pool=self._config.pool_timeout,
        )

    def _limits(self) -> httpx.Limits:
        return httpx.Limits(
            max_connections=self._config.max_connections,
            max_keepalive_connections=self._config.max_keepalive_connections,
        )

    def _base_headers(self, request_id: str) -> dict[str, str]:
        headers = {
            "User-Agent": self._config.user_agent,
            _REQUEST_ID_HEADER: request_id,
        }
        if self._config.send_auth_header:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _new_client(self, request_id: str) -> httpx.AsyncClient:
        """Construct a fresh httpx client. Callers wrap in ``async with``."""

        return httpx.AsyncClient(
            base_url=self._config.api_base_url,
            timeout=self._timeout(),
            limits=self._limits(),
            headers=self._base_headers(request_id),
            follow_redirects=False,
        )

    def _new_download_client(self, request_id: str) -> httpx.AsyncClient:
        """Like ``_new_client`` but absolute-URL friendly (no ``base_url``)."""

        return httpx.AsyncClient(
            timeout=self._timeout(),
            limits=self._limits(),
            headers={
                "User-Agent": self._config.user_agent,
                _REQUEST_ID_HEADER: request_id,
            },
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        """No-op. Retained for API compatibility with callers (the
        ``Client`` façade and ``AsyncClient.__aexit__``) that want a
        sentinel "I'm done" call. Per-operation clients clean themselves
        up via ``async with`` inside each method."""

    # ---- JSON requests with retry ----------------------------------------

    async def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        # rw-api currently only reads ?api_key=; the header is sent in
        # addition when send_auth_header is True so callers can dual-
        # stack against a server that supports both.
        params.setdefault("api_key", self._config.api_key)
        return await self._with_retries(method="GET", path=path, params=params)

    async def post_json(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        # POSTs are not auto-retried — they may have already had side
        # effects on the server before the failure surfaced.
        request_id = _new_request_id()
        async with self._new_client(request_id) as client:
            response = await client.post(path, params={"api_key": self._config.api_key}, json=json)
        return self._unwrap_json(response)

    async def _with_retries(
        self,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        cfg = self._config
        last_exc: BaseException | None = None
        attempts = cfg.max_retries + 1  # initial + retries
        for attempt in range(attempts):
            request_id = _new_request_id()
            try:
                async with self._new_client(request_id) as client:
                    response = await client.request(method, path, params=params)
                if response.status_code < 400:
                    return _decode_json(response)
                err = _map_rw_api_error(response)
                if not _is_retryable_status(response.status_code):
                    raise err
                last_exc = err
                if attempt == attempts - 1:
                    raise err
                retry_after = getattr(err, "retry_after", None)
                if retry_after is None and response.status_code == 429:
                    # rw-api rate limits per minute and sends no Retry-After;
                    # sub-second exponential backoff would just burn the
                    # remaining attempts inside the same window.
                    retry_after = cfg.rate_limit_wait * (attempt + 1)
                    log.info(
                        "rate limited on %s; waiting %.0fs before retry %d/%d",
                        path,
                        min(retry_after, max(cfg.retry_backoff_max, cfg.rate_limit_wait)),
                        attempt + 1,
                        attempts - 1,
                    )
                await asyncio.sleep(
                    _retry_delay(
                        attempt=attempt,
                        base=cfg.retry_backoff_base,
                        cap=max(cfg.retry_backoff_max, cfg.rate_limit_wait),
                        retry_after=retry_after,
                    )
                )
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt == attempts - 1:
                    raise RwApiError(f"transport error after {attempts} attempts: {exc}") from exc
                await asyncio.sleep(
                    _retry_delay(
                        attempt=attempt,
                        base=cfg.retry_backoff_base,
                        cap=cfg.retry_backoff_max,
                        retry_after=None,
                    )
                )
        # Defensive: loop exits via raise above on every path. If we
        # somehow fall through, re-raise the last seen exception.
        assert last_exc is not None
        raise last_exc  # pragma: no cover

    @staticmethod
    def _unwrap_json(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise _map_rw_api_error(response)
        return _decode_json(response)

    # ---- signed-URL downloads --------------------------------------------

    async def stream_signed_url(
        self,
        url: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        """Download a CDN signed URL into memory.

        For very large files prefer :meth:`download_signed_url_to_path` —
        this method materialises the full response in RAM.
        """

        request_id = _new_request_id()
        async with self._new_download_client(request_id) as client:
            response = await client.get(url)
            if response.status_code in (401, 403, 410):
                raise SignedUrlExpiredError(
                    f"CDN rejected the signed URL ({sanitize_url(url)}) — "
                    f"likely expired or revoked.",
                    status_code=response.status_code,
                    request_id=request_id,
                )
            if response.status_code >= 400:
                raise RwApiError(
                    f"CDN download failed: {response.status_code} "
                    f"{response.reason_phrase} ({sanitize_url(url)})",
                    status_code=response.status_code,
                    request_id=request_id,
                )
            content = response.content
            if max_bytes is not None and len(content) > max_bytes:
                raise RwApiError(
                    f"Object exceeded max_bytes={max_bytes}; refusing to load into memory."
                )
            return content

    async def download_signed_url_to_path(
        self,
        url: str,
        target_path: Path,
        *,
        chunk_size: int = 1024 * 1024,
        expected_size: int | None = None,
    ) -> int:
        """Stream a signed URL into ``target_path``. Returns bytes written.

        ``target_path`` is written in place (caller is responsible for
        any atomic-rename dance; in practice this is invoked from
        :meth:`rwpytools.cache.FilesystemCache.reserve`, which yields a
        ``.part`` tempfile and renames on success).

        If ``expected_size`` is supplied it is checked against the
        upstream ``Content-Length`` header (when present) and against
        the number of bytes actually streamed; a mismatch raises
        :class:`RwApiError` instead of silently writing a truncated
        file.
        """

        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        request_id = _new_request_id()
        bytes_written = 0
        async with (
            self._new_download_client(request_id) as client,
            client.stream("GET", url) as response,
        ):
            if response.status_code in (401, 403, 410):
                raise SignedUrlExpiredError(
                    f"CDN rejected the signed URL ({sanitize_url(url)}) — "
                    f"likely expired or revoked.",
                    status_code=response.status_code,
                    request_id=request_id,
                )
            if response.status_code >= 400:
                raise RwApiError(
                    f"CDN download failed: {response.status_code} "
                    f"{response.reason_phrase} ({sanitize_url(url)})",
                    status_code=response.status_code,
                    request_id=request_id,
                )
            if expected_size is not None:
                cl_header = response.headers.get("Content-Length")
                if cl_header is not None:
                    try:
                        cl_value: int | None = int(cl_header)
                    except ValueError:
                        cl_value = None  # malformed header; just skip the check
                    if cl_value is not None and cl_value != expected_size:
                        raise RwApiError(
                            f"CDN Content-Length {cl_value} does not match "
                            f"rw-api advertised size {expected_size}; "
                            f"aborting download to avoid caching a corrupt "
                            f"object."
                        )
            with target.open("wb") as fh:
                async for chunk in response.aiter_bytes(chunk_size):
                    fh.write(chunk)
                    bytes_written += len(chunk)

        if expected_size is not None and bytes_written != expected_size:
            raise RwApiError(
                f"CDN stream ended after {bytes_written} bytes but "
                f"rw-api advertised {expected_size}; download was "
                f"truncated or oversized."
            )
        return bytes_written


# ---- helpers --------------------------------------------------------------


_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUSES


def _retry_delay(*, attempt: int, base: float, cap: float, retry_after: float | None) -> float:
    """Return seconds to sleep before the next retry attempt.

    ``attempt`` is 0-indexed (0 means "before the second attempt").
    Honours an explicit ``Retry-After`` when present; otherwise uses
    capped exponential backoff with full jitter so concurrent callers
    do not retry in lockstep.
    """

    if retry_after is not None and retry_after >= 0:
        # Cap server-supplied Retry-After at our configured ceiling so a
        # buggy server can't park us for hours.
        return min(retry_after, cap)
    raw = min(cap, base * (2**attempt))
    return random.uniform(0, raw)


def _new_request_id() -> str:
    return f"rwpy-{uuid.uuid4().hex[:12]}"


def _decode_json(response: httpx.Response) -> dict[str, Any]:
    try:
        decoded = response.json()
    except ValueError as exc:
        raise RwApiError(
            f"rw-api returned non-JSON body (status={response.status_code}): "
            f"{response.text[:200]!r}"
        ) from exc
    if not isinstance(decoded, dict):
        raise RwApiError(f"rw-api returned JSON of unexpected shape: {type(decoded).__name__}")
    return decoded


_T = TypeVar("_T")


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an awaitable to completion, even from inside another loop.

    Generic in the coroutine's return type so sync façades that wrap
    typed async methods do not collapse to ``Any`` under mypy.

    The simple ``asyncio.run`` errors when called from inside a running
    loop; in that case we defer to a one-shot thread so notebook /
    Jupyter callers still work. Each invocation opens and closes its
    own loop — :class:`_Http` is designed to scope every connection
    inside the current coroutine so this loop-per-call pattern is safe.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Inside a running loop — defer to a thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


__all__ = ["_Http", "run_sync", "sanitize_url"]
