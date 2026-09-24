"""Public exception hierarchy.

The hierarchy is intentionally shallow. Callers typically catch
:class:`RwApiError` to handle anything coming from the network round
trip and let configuration issues (:class:`ConfigError`) surface as
programmer errors.

Every :class:`RwApiError` carries:

* ``status_code`` — the HTTP status code returned by rw-api / the CDN,
  when one is available (``None`` for purely client-side failures
  like a transport timeout that exhausted retries).
* ``request_id`` — the ``X-Request-ID`` echoed by the server. Surface
  this in support tickets so the server-side log can be located.
* ``__cause__`` / ``__context__`` — standard Python exception chaining;
  the upstream ``httpx`` exception (when present) is preserved.

Specific subclasses add extra fields where they're actionable —
e.g. :class:`BandwidthExceededError` parses the limit out of the
server message and exposes it as ``limit_gb`` so a UI layer can
render "you've hit your cap, contact support to upgrade".
"""

from __future__ import annotations

import re
from typing import Any


class RwPyToolsError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(RwPyToolsError):
    """Raised for invalid client construction (missing API key, bad URL, etc.).

    These are programmer errors — fix at the call site, do not retry.
    """


class CacheError(RwPyToolsError):
    """Raised when the on-disk cache is in a state we can't recover from."""


class RoutingError(RwPyToolsError, ValueError):
    """The request cannot be routed: unknown dataset, a ``source`` the
    dataset does not support, an empty date range, or a symbol-templated
    dataset called without symbols.

    Subclasses :class:`ValueError` because every case is a bad argument.
    """


class DataFormatError(RwPyToolsError):
    """rw-api returned a body whose shape the client does not understand."""


class DatasetNotSeededError(NotImplementedError, RwPyToolsError):
    """Method exists 1:1 with rwRTools but the backing dataset is not yet
    registered on rw-api.

    The class hierarchy intentionally bridges :class:`NotImplementedError`
    and :class:`RwPyToolsError` so callers can:

    * filter narrowly with ``except DatasetNotSeededError`` for a feature
      detection path that doesn't care about other RwPyTools errors;
    * catch the family with ``except NotImplementedError`` for the
      "this is a stub" semantics used by R's ``rwRTools`` parity layer.

    The message names the expected ``(namespace, dataset, schema)``
    triple so reporting tools and bug reports identify which rw-api
    fixture is missing.
    """

    def __init__(
        self,
        *,
        namespace: str,
        dataset: str,
        schema: str | None = None,
        method: str | None = None,
    ) -> None:
        target = f"{namespace}/{dataset}"
        if schema:
            target += f"/{schema}"
        suffix = f" (rwRTools name: {method})" if method else ""
        super().__init__(
            f"{target} is not yet seeded on rw-api{suffix}. "
            f"Track rw-api / rw-terraform for the bucket fixture; "
            f"this method's signature is final."
        )
        self.namespace = namespace
        self.dataset = dataset
        self.schema = schema
        self.method = method


class RwApiError(RwPyToolsError):
    """Base class for HTTP errors returned by rw-api or the CDN.

    Subclasses carry HTTP status and decoded message when available.
    """

    status_code: int | None = None

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        decorated = message
        if request_id:
            decorated = f"{message} [request_id={request_id}]"
        super().__init__(decorated)
        # The bare message (no request_id decoration) is occasionally
        # useful for callers that want to render the server's own
        # words; keep it accessible.
        self.message = message
        self.status_code = status_code
        self.request_id = request_id


class AuthenticationError(RwApiError):
    """401 from rw-api — bad / missing API key."""

    status_code = 401


class AuthorizationError(RwApiError):
    """403 from rw-api — the key is valid but lacks access to the resource.

    Distinct from :class:`BandwidthExceededError`, which is also 403
    but means "you've hit your cap" rather than "you have no permission
    to this dataset / pod".
    """

    status_code = 403


_BANDWIDTH_RE = re.compile(r"bandwidth limit of\s*([0-9.]+)\s*([kmg]i?b)", re.IGNORECASE)


class BandwidthExceededError(RwApiError):
    """403 returned because the per-user bandwidth cap would be exceeded.

    The server message contains the cap in GB; we parse it best-effort
    into :attr:`limit_gb` so a UI can render it without re-parsing.
    """

    status_code = 403

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        server_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, request_id=request_id)
        self.server_body = server_body or {}
        self.limit_gb: float | None = None
        match = _BANDWIDTH_RE.search(message or "")
        if match:
            try:
                self.limit_gb = float(match.group(1))
            except ValueError:
                self.limit_gb = None


class NotFoundError(RwApiError):
    """404 — pod or object not found."""

    status_code = 404


class RateLimitError(RwApiError):
    """429 — rate-limited. Caller should back off and retry with jitter.

    :attr:`retry_after` carries the parsed ``Retry-After`` header in
    seconds, when the server supplies one.
    """

    status_code = 429

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, request_id=request_id)
        self.retry_after = retry_after


class ServerError(RwApiError):
    """5xx — transient or upstream issue. Retried automatically up to
    :attr:`ClientConfig.max_retries`; reaches the caller only when
    retries are exhausted."""


class SignedUrlExpiredError(RwApiError):
    """The signed URL was rejected by Cloud CDN as expired or invalid.

    :class:`rwpytools.bulk.BulkClient` catches this internally and
    re-mints a fresh URL once before propagating; advanced callers
    using ``stream_signed_url`` directly should do the same.
    """


__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "BandwidthExceededError",
    "CacheError",
    "ConfigError",
    "DataFormatError",
    "DatasetNotSeededError",
    "NotFoundError",
    "RateLimitError",
    "RoutingError",
    "RwApiError",
    "RwPyToolsError",
    "ServerError",
    "SignedUrlExpiredError",
]
