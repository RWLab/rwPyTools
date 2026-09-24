"""Client configuration: API endpoints, cache locations, defaults.

Everything is overridable via constructor argument or environment variable.
The defaults target the production API at ``https://api.robotwealth.com``.

Precedence: constructor argument > environment variable > default.

API key handling
----------------

The ``api_key`` field is treated as a secret. Its value is **never**
echoed by :meth:`ClientConfig.__repr__` or by any of the package's own
logging — only the first/last two characters appear, with the middle
masked. Avoid printing ``ClientConfig`` instances yourself; if you do,
the masked representation is what shows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from platformdirs import user_cache_dir

from .errors import ConfigError

_ENV_API_BASE_URL = "RWPYTOOLS_API_BASE_URL"
_ENV_API_KEY = "RWPYTOOLS_API_KEY"
_ENV_CACHE_DIR = "RWPYTOOLS_CACHE_DIR"
_ENV_CACHE_MAX_BYTES = "RWPYTOOLS_CACHE_MAX_BYTES"
_ENV_DEFAULT_EXPIRES_IN = "RWPYTOOLS_DEFAULT_EXPIRES_IN"
_ENV_REQUEST_TIMEOUT = "RWPYTOOLS_REQUEST_TIMEOUT"
_ENV_CONNECT_TIMEOUT = "RWPYTOOLS_CONNECT_TIMEOUT"
_ENV_READ_TIMEOUT = "RWPYTOOLS_READ_TIMEOUT"
_ENV_WRITE_TIMEOUT = "RWPYTOOLS_WRITE_TIMEOUT"
_ENV_POOL_TIMEOUT = "RWPYTOOLS_POOL_TIMEOUT"
_ENV_DOWNLOAD_CONCURRENCY = "RWPYTOOLS_DOWNLOAD_CONCURRENCY"
_ENV_MAX_RETRIES = "RWPYTOOLS_MAX_RETRIES"
_ENV_RETRY_BACKOFF_BASE = "RWPYTOOLS_RETRY_BACKOFF_BASE"
_ENV_RETRY_BACKOFF_MAX = "RWPYTOOLS_RETRY_BACKOFF_MAX"
_ENV_MAX_CONNECTIONS = "RWPYTOOLS_MAX_CONNECTIONS"
_ENV_MAX_KEEPALIVE = "RWPYTOOLS_MAX_KEEPALIVE_CONNECTIONS"
_ENV_SEND_AUTH_HEADER = "RWPYTOOLS_SEND_AUTH_HEADER"
_ENV_BULK_TTL = "RWPYTOOLS_BULK_TTL"
_ENV_LIVE_DIFF_MAX_DAYS = "RWPYTOOLS_LIVE_DIFF_MAX_DAYS"
_ENV_LIVE_ONLY_WINDOW_DAYS = "RWPYTOOLS_LIVE_ONLY_WINDOW_DAYS"
_ENV_RATE_LIMIT_WAIT = "RWPYTOOLS_RATE_LIMIT_WAIT"

DEFAULT_API_BASE_URL = "https://api.robotwealth.com"
DEFAULT_CACHE_MAX_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB on-disk cache cap.
DEFAULT_PRESIGNED_EXPIRES_IN = 3600
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 60.0
DEFAULT_WRITE_TIMEOUT = 30.0
DEFAULT_POOL_TIMEOUT = 5.0
DEFAULT_DOWNLOAD_CONCURRENCY = 8
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE = 0.5  # seconds
DEFAULT_RETRY_BACKOFF_MAX = 30.0  # seconds — cap per attempt before jitter
DEFAULT_MAX_CONNECTIONS = 32
DEFAULT_MAX_KEEPALIVE = 16
#: Seconds a cached bulk export of a dataset WITHOUT a live companion is
#: trusted before it is re-validated against rw-api. Exports refresh daily.
DEFAULT_BULK_TTL = 24 * 3600.0
#: For datasets WITH a live companion, the cached export is topped up from
#: the live API instead of being re-downloaded — until its watermark is more
#: than this many days old, at which point one fresh bulk download is
#: cheaper than paging the gap through the live endpoint.
DEFAULT_LIVE_DIFF_MAX_DAYS = 30
#: With nothing cached, a request whose ``start`` is within this many days
#: of today is served entirely live rather than downloading the full export.
DEFAULT_LIVE_ONLY_WINDOW_DAYS = 30
#: Seconds to wait before retrying a 429 that carries no ``Retry-After``.
#: rw-api's limits are per minute and it does not send the header.
DEFAULT_RATE_LIMIT_WAIT = 20.0


def _mask_api_key(api_key: str) -> str:
    """Return a printable, low-information stand-in for ``api_key``.

    Preserves the first and last two characters so users recognise
    which key they configured, masks the middle. Empty / very short
    keys are reported as ``***``.
    """

    if not api_key:
        return "***"
    if len(api_key) <= 4:
        return "***"
    return f"{api_key[:2]}...{api_key[-2:]}"


@dataclass(frozen=True)
class ClientConfig:
    """Resolved configuration for a :class:`rwpytools.Client`.

    Construct via :meth:`from_env` to honour environment-variable overrides,
    or pass an instance directly to the client for full control.

    The ``request_timeout`` field is retained as a single-knob shortcut
    that fans out to ``connect_timeout`` / ``read_timeout`` /
    ``write_timeout`` / ``pool_timeout`` when those aren't explicitly
    set. For finer control set each phase explicitly.
    """

    api_base_url: str
    api_key: str
    cache_dir: Path
    cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES
    default_expires_in: int = DEFAULT_PRESIGNED_EXPIRES_IN
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    read_timeout: float = DEFAULT_READ_TIMEOUT
    write_timeout: float = DEFAULT_WRITE_TIMEOUT
    pool_timeout: float = DEFAULT_POOL_TIMEOUT
    download_concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff_base: float = DEFAULT_RETRY_BACKOFF_BASE
    retry_backoff_max: float = DEFAULT_RETRY_BACKOFF_MAX
    max_connections: int = DEFAULT_MAX_CONNECTIONS
    max_keepalive_connections: int = DEFAULT_MAX_KEEPALIVE
    send_auth_header: bool = False  # rw-api today only reads ?api_key=
    bulk_ttl: float = DEFAULT_BULK_TTL
    live_diff_max_days: int = DEFAULT_LIVE_DIFF_MAX_DAYS
    live_only_window_days: int = DEFAULT_LIVE_ONLY_WINDOW_DAYS
    rate_limit_wait: float = DEFAULT_RATE_LIMIT_WAIT
    user_agent: str = field(default_factory=lambda: _default_user_agent())

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ConfigError("api_key must be a non-empty string")
        if not self.api_base_url.startswith(("http://", "https://")):
            raise ConfigError("api_base_url must be absolute http(s)://")
        if self.cache_max_bytes < 0:
            raise ConfigError("cache_max_bytes must be >= 0 (0 disables eviction)")
        if not 0 < self.default_expires_in <= 24 * 3600:
            raise ConfigError("default_expires_in must be in (0, 86400]")
        if self.download_concurrency < 1:
            raise ConfigError("download_concurrency must be >= 1")
        if self.max_retries < 0:
            raise ConfigError("max_retries must be >= 0")
        if self.retry_backoff_base <= 0 or self.retry_backoff_max <= 0:
            raise ConfigError("retry_backoff_* values must be positive")
        if self.max_connections < 1 or self.max_keepalive_connections < 0:
            raise ConfigError("max_connections must be >= 1, max_keepalive >= 0")
        if self.bulk_ttl < 0:
            raise ConfigError("bulk_ttl must be >= 0 (0 re-validates on every call)")
        if self.live_diff_max_days < 0 or self.live_only_window_days < 0:
            raise ConfigError("live_diff_max_days / live_only_window_days must be >= 0")
        if self.rate_limit_wait < 0:
            raise ConfigError("rate_limit_wait must be >= 0")
        for phase, value in (
            ("connect_timeout", self.connect_timeout),
            ("read_timeout", self.read_timeout),
            ("write_timeout", self.write_timeout),
            ("pool_timeout", self.pool_timeout),
        ):
            if value <= 0:
                raise ConfigError(f"{phase} must be > 0")

    # ---- safe representation ----------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ClientConfig(api_base_url={self.api_base_url!r}, "
            f"api_key={_mask_api_key(self.api_key)!r}, "
            f"cache_dir={str(self.cache_dir)!r}, "
            f"cache_max_bytes={self.cache_max_bytes}, "
            f"download_concurrency={self.download_concurrency}, "
            f"max_retries={self.max_retries})"
        )

    def with_overrides(self, **changes: object) -> ClientConfig:
        """Return a copy of this config with the given fields replaced."""

        return replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        api_base_url: str | None = None,
        cache_dir: Path | None = None,
        cache_max_bytes: int | None = None,
        default_expires_in: int | None = None,
        request_timeout: float | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
        pool_timeout: float | None = None,
        download_concurrency: int | None = None,
        max_retries: int | None = None,
        retry_backoff_base: float | None = None,
        retry_backoff_max: float | None = None,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        send_auth_header: bool | None = None,
        bulk_ttl: float | None = None,
        live_diff_max_days: int | None = None,
        live_only_window_days: int | None = None,
        rate_limit_wait: float | None = None,
    ) -> ClientConfig:
        resolved_key = api_key or os.environ.get(_ENV_API_KEY) or ""
        if not resolved_key:
            raise ConfigError(f"API key not provided. Set {_ENV_API_KEY} or pass api_key=...")

        # Phase-specific timeouts fall back to the umbrella ``request_timeout``
        # so legacy callers that only set one knob still get coherent behaviour.
        umbrella_timeout = _float_env(
            _ENV_REQUEST_TIMEOUT, request_timeout, DEFAULT_REQUEST_TIMEOUT
        )

        return cls(
            api_base_url=api_base_url or os.environ.get(_ENV_API_BASE_URL) or DEFAULT_API_BASE_URL,
            api_key=resolved_key,
            cache_dir=cache_dir or _default_cache_dir(),
            cache_max_bytes=_int_env(
                _ENV_CACHE_MAX_BYTES, cache_max_bytes, DEFAULT_CACHE_MAX_BYTES
            ),
            default_expires_in=_int_env(
                _ENV_DEFAULT_EXPIRES_IN,
                default_expires_in,
                DEFAULT_PRESIGNED_EXPIRES_IN,
            ),
            request_timeout=umbrella_timeout,
            connect_timeout=_float_env(_ENV_CONNECT_TIMEOUT, connect_timeout, umbrella_timeout),
            read_timeout=_float_env(
                _ENV_READ_TIMEOUT, read_timeout, max(umbrella_timeout, DEFAULT_READ_TIMEOUT)
            ),
            write_timeout=_float_env(_ENV_WRITE_TIMEOUT, write_timeout, umbrella_timeout),
            pool_timeout=_float_env(_ENV_POOL_TIMEOUT, pool_timeout, DEFAULT_POOL_TIMEOUT),
            download_concurrency=_int_env(
                _ENV_DOWNLOAD_CONCURRENCY,
                download_concurrency,
                DEFAULT_DOWNLOAD_CONCURRENCY,
            ),
            max_retries=_int_env(_ENV_MAX_RETRIES, max_retries, DEFAULT_MAX_RETRIES),
            retry_backoff_base=_float_env(
                _ENV_RETRY_BACKOFF_BASE,
                retry_backoff_base,
                DEFAULT_RETRY_BACKOFF_BASE,
            ),
            retry_backoff_max=_float_env(
                _ENV_RETRY_BACKOFF_MAX,
                retry_backoff_max,
                DEFAULT_RETRY_BACKOFF_MAX,
            ),
            max_connections=_int_env(
                _ENV_MAX_CONNECTIONS,
                max_connections,
                DEFAULT_MAX_CONNECTIONS,
            ),
            max_keepalive_connections=_int_env(
                _ENV_MAX_KEEPALIVE,
                max_keepalive_connections,
                DEFAULT_MAX_KEEPALIVE,
            ),
            send_auth_header=_bool_env(_ENV_SEND_AUTH_HEADER, send_auth_header, False),
            bulk_ttl=_float_env(_ENV_BULK_TTL, bulk_ttl, DEFAULT_BULK_TTL),
            live_diff_max_days=_int_env(
                _ENV_LIVE_DIFF_MAX_DAYS, live_diff_max_days, DEFAULT_LIVE_DIFF_MAX_DAYS
            ),
            live_only_window_days=_int_env(
                _ENV_LIVE_ONLY_WINDOW_DAYS, live_only_window_days, DEFAULT_LIVE_ONLY_WINDOW_DAYS
            ),
            rate_limit_wait=_float_env(
                _ENV_RATE_LIMIT_WAIT, rate_limit_wait, DEFAULT_RATE_LIMIT_WAIT
            ),
        )


def _default_cache_dir() -> Path:
    override = os.environ.get(_ENV_CACHE_DIR)
    if override:
        return Path(override).expanduser().resolve()
    return Path(user_cache_dir("rwpytools", appauthor="robotwealth"))


def _default_user_agent() -> str:
    from ._version import __version__

    return f"rwpytools/{__version__} (+https://github.com/Robot-Wealth/rwPyTools)"


def _int_env(env_var: str, override: int | None, default: int) -> int:
    if override is not None:
        return override
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{env_var} must be an integer, got {raw!r}") from exc


def _float_env(env_var: str, override: float | None, default: float) -> float:
    if override is not None:
        return float(override)
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{env_var} must be a float, got {raw!r}") from exc


def _bool_env(env_var: str, override: bool | None, default: bool) -> bool:
    if override is not None:
        return bool(override)
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    raise ConfigError(f"{env_var} must be a boolean-like string, got {raw!r}")


__all__ = [
    "DEFAULT_API_BASE_URL",
    "DEFAULT_BULK_TTL",
    "DEFAULT_CACHE_MAX_BYTES",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_DOWNLOAD_CONCURRENCY",
    "DEFAULT_LIVE_DIFF_MAX_DAYS",
    "DEFAULT_LIVE_ONLY_WINDOW_DAYS",
    "DEFAULT_MAX_CONNECTIONS",
    "DEFAULT_MAX_KEEPALIVE",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_POOL_TIMEOUT",
    "DEFAULT_PRESIGNED_EXPIRES_IN",
    "DEFAULT_RATE_LIMIT_WAIT",
    "DEFAULT_READ_TIMEOUT",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_RETRY_BACKOFF_BASE",
    "DEFAULT_RETRY_BACKOFF_MAX",
    "DEFAULT_WRITE_TIMEOUT",
    "ClientConfig",
]
