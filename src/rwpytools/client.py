"""Public client classes.

:class:`Client` is the synchronous, primary entry point;
:class:`AsyncClient` is its async-native sibling with the same surface and
coroutine methods. Both accept the API key positionally, as ``api_key=``,
via the :envvar:`RWPYTOOLS_API_KEY` environment variable, or — in a
notebook or a terminal — from a masked stdin prompt.

    import rwpytools

    client = rwpytools.Client("...")

    # Auto-routed: the cached bulk export for history, topped up to real
    # time from the live API, joined into one result.
    result = client.get("binance_spot_1h", symbols=["BTCUSDT"], start="2024-01-01")
    print(result.describe_route())
    df = result.df

    # rwRtools-style pod methods route the same way and return DataFrames.
    eurusd = client.fx.get_daily_ohlc_ticker("EURUSD")
    vix = client.macro.get_vix_vix3m(source="bulk")   # force one side

The key is a secret: it is never printed by ``repr`` and is redacted from
every log record.
"""

from __future__ import annotations

import getpass
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from .bulk import BulkClient, FetchRequest, FetchResult
from .cache import FilesystemCache
from .config import _ENV_API_KEY, ClientConfig, _mask_api_key
from .datasets import DATASETS, DatasetSpec
from .http import _Http, run_sync
from .live import AsyncLiveClient, LiveClient
from .pods import (
    AsyncCryptoPod,
    AsyncEquityFactorsPod,
    AsyncFxPod,
    AsyncMacroPod,
    CryptoPod,
    EquityFactorsPod,
    FxPod,
    MacroPod,
)
from .result import Result
from .routing import RoutePlan, Source
from .schemas import DatasetCatalogResponse, SymbolsResponse
from .session import AsyncSession

# ---------------------------------------------------------------------------
# API-key prompt plumbing (Colab / Jupyter / TTY)
# ---------------------------------------------------------------------------


def _running_in_ipython() -> bool:
    """Return True if we're inside IPython/Jupyter/Colab (where stdin is the kernel)."""

    if "google.colab" in sys.modules:
        return True
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    return get_ipython() is not None


def _is_interactive_session() -> bool:
    """Whether it's safe to prompt the user for input.

    True in: IPython/Jupyter/Colab kernels; a TTY stdin (running in a
    terminal). False in: piped/redirected stdin, CI, daemon processes,
    when ``RWPYTOOLS_NO_PROMPT`` is set in the environment.
    """

    if os.environ.get("RWPYTOOLS_NO_PROMPT"):
        return False
    if _running_in_ipython():
        return True
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except (AttributeError, ValueError):
        return False


def _prompt_for_api_key() -> str:
    """Ask for the API key on stdin, masking input. Returns the trimmed key."""

    print(
        "rwpytools: no API key found (RWPYTOOLS_API_KEY env var unset).\n"
        "Paste your key below (input is hidden):"
    )
    try:
        return getpass.getpass("RWPYTOOLS_API_KEY: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _resolve_api_key(explicit: str | None, *, prompt_if_missing: bool | None) -> str | None:
    """Return the api_key to hand to :meth:`ClientConfig.from_env`.

    If ``explicit`` is given or the env var is set, returns
    ``explicit`` (the env var lookup happens inside ``from_env``).
    Otherwise, if ``prompt_if_missing`` allows it, opens an interactive
    prompt; on successful entry the key is also stashed in
    ``RWPYTOOLS_API_KEY`` for the lifetime of the process so
    re-imports inside the same notebook session don't re-prompt.
    """

    if explicit:
        return explicit
    if os.environ.get(_ENV_API_KEY):
        return None  # let from_env() pick it up

    if prompt_if_missing is None:
        prompt_if_missing = _is_interactive_session()
    if not prompt_if_missing:
        return None  # let from_env() raise ConfigError with the standard message

    key = _prompt_for_api_key()
    if key:
        os.environ[_ENV_API_KEY] = key
    return key or None


def _build_config(
    api_key: str | None,
    *,
    config: ClientConfig | None,
    prompt_for_api_key: bool | None,
    base_url: str | None,
    cache_dir: Path | str | None,
    cache_max_bytes: int | None,
    overrides: dict[str, Any],
) -> ClientConfig:
    if config is not None:
        return config
    resolved = _resolve_api_key(api_key, prompt_if_missing=prompt_for_api_key)
    return ClientConfig.from_env(
        api_key=resolved,
        api_base_url=base_url,
        cache_dir=None if cache_dir is None else Path(cache_dir).expanduser(),
        cache_max_bytes=cache_max_bytes,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Synchronous façades
# ---------------------------------------------------------------------------


class _SyncBulk:
    """Synchronous façade over :class:`BulkClient`.

    Method signatures mirror the async ones exactly; every call runs
    on a fresh one-shot event loop via :func:`rwpytools.http.run_sync`.
    """

    def __init__(self, async_bulk: BulkClient) -> None:
        self._async = async_bulk

    @property
    def cache(self) -> FilesystemCache:
        return self._async.cache

    def list_datasets(self, namespace: str) -> DatasetCatalogResponse:
        return run_sync(self._async.list_datasets(namespace))

    def list_symbols(
        self,
        namespace: str,
        *,
        dataset: str,
        schema: str | None = None,
    ) -> SymbolsResponse:
        return run_sync(self._async.list_symbols(namespace, dataset=dataset, schema=schema))

    def ensure(
        self,
        namespace: str,
        *,
        dataset: str,
        schema: str,
        symbol: str | None = None,
        force_refresh: bool = False,
        max_age: float | None = None,
    ) -> FetchResult:
        return run_sync(
            self._async.ensure(
                namespace,
                dataset=dataset,
                schema=schema,
                symbol=symbol,
                force_refresh=force_refresh,
                max_age=max_age,
            )
        )

    def fetch_one(
        self,
        namespace: str,
        *,
        dataset: str,
        schema: str,
        symbol: str | None = None,
        force_refresh: bool = False,
        expires_in: int | None = None,
    ) -> FetchResult:
        return run_sync(
            self._async.fetch_one(
                namespace,
                dataset=dataset,
                schema=schema,
                symbol=symbol,
                force_refresh=force_refresh,
                expires_in=expires_in,
            )
        )

    def fetch_many(
        self,
        namespace: str,
        requests: Sequence[FetchRequest],
        *,
        force_refresh: bool = False,
        expires_in: int | None = None,
    ) -> list[FetchResult]:
        return run_sync(
            self._async.fetch_many(
                namespace,
                requests,
                force_refresh=force_refresh,
                expires_in=expires_in,
            )
        )

    def fetch_one_as_frame(
        self,
        namespace: str,
        *,
        dataset: str,
        schema: str,
        symbol: str | None = None,
        force_refresh: bool = False,
        expires_in: int | None = None,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.fetch_one_as_frame(
                namespace,
                dataset=dataset,
                schema=schema,
                symbol=symbol,
                force_refresh=force_refresh,
                expires_in=expires_in,
            )
        )

    def fetch_many_as_frames(
        self,
        namespace: str,
        requests: Sequence[FetchRequest],
        *,
        force_refresh: bool = False,
        expires_in: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        return run_sync(
            self._async.fetch_many_as_frames(
                namespace,
                requests,
                force_refresh=force_refresh,
                expires_in=expires_in,
            )
        )


# ---------------------------------------------------------------------------
# Shared wiring
# ---------------------------------------------------------------------------


class _Stack:
    """The object graph both clients are built from."""

    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self.http = _Http(config)
        self.cache = FilesystemCache(config.cache_dir, max_bytes=config.cache_max_bytes)
        self.bulk = BulkClient(config, http=self.http, cache=self.cache)
        self.live = AsyncLiveClient(self.http, concurrency=config.download_concurrency)
        self.session = AsyncSession(config, bulk=self.bulk, live=self.live)
        self.crypto = AsyncCryptoPod(self.bulk, self.session)
        self.fx = AsyncFxPod(self.bulk, self.session)
        self.macro = AsyncMacroPod(self.bulk, self.session)
        self.equity_factors = AsyncEquityFactorsPod(self.bulk, self.session)


_CLIENT_PARAMS_DOC = """
    Parameters
    ----------
    api_key:
        The key, positionally or by keyword. Falls back to
        :envvar:`RWPYTOOLS_API_KEY`, then to a masked prompt when running
        interactively (Colab / Jupyter / TTY).
    base_url:
        Override the API endpoint (default :envvar:`RWPYTOOLS_API_BASE_URL`
        or ``https://api.robotwealth.com``).
    cache_dir / cache_max_bytes:
        Where bulk exports are cached and the byte budget for them.
    config:
        A fully-built :class:`ClientConfig`; when given, every other
        construction argument is ignored.
    prompt_for_api_key:
        ``True`` to force the stdin prompt, ``False`` to forbid it,
        ``None`` (default) to prompt only in an interactive session.
    **config_overrides:
        Any other :meth:`ClientConfig.from_env` field, e.g.
        ``bulk_ttl=3600`` or ``max_retries=5``.
"""


# ---------------------------------------------------------------------------
# Sync Client (primary entry point)
# ---------------------------------------------------------------------------


class Client:
    """Synchronous Robot Wealth research data client.

    Construct once per process; it holds an HTTP transport and a
    filesystem cache. Close it (or use it as a context manager) when done.

    Example::

        import rwpytools

        with rwpytools.Client("...") as client:
            result = client.get("binance_perps_1h", symbols=["BTCUSDT"], start="2025-01-01")
            df = client.crypto.get_binance_spot_1h(source="bulk")
            eurusd = client.fx.get_daily_ohlc_ticker("EURUSD")
    """

    __doc__ = (__doc__ or "") + _CLIENT_PARAMS_DOC

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        cache_dir: Path | str | None = None,
        cache_max_bytes: int | None = None,
        config: ClientConfig | None = None,
        prompt_for_api_key: bool | None = None,
        **config_overrides: Any,
    ) -> None:
        self._stack = _Stack(
            _build_config(
                api_key,
                config=config,
                prompt_for_api_key=prompt_for_api_key,
                base_url=base_url,
                cache_dir=cache_dir,
                cache_max_bytes=cache_max_bytes,
                overrides=config_overrides,
            )
        )
        self._config = self._stack.config
        self._http = self._stack.http
        self._cache = self._stack.cache
        self._async_bulk = self._stack.bulk

        self.bulk = _SyncBulk(self._stack.bulk)
        self.live = LiveClient(self._stack.http, concurrency=self._config.download_concurrency)

        self.crypto = CryptoPod(self._stack.crypto)
        self.fx = FxPod(self._stack.fx)
        self.macro = MacroPod(self._stack.macro)
        self.equity_factors = EquityFactorsPod(self._stack.equity_factors)

    # ---- introspection -----------------------------------------------------

    @property
    def config(self) -> ClientConfig:
        return self._config

    @property
    def cache(self) -> FilesystemCache:
        """The bulk export cache — inspect it, or ``.clear()`` it."""

        return self._cache

    def __repr__(self) -> str:
        return (
            f"<rwpytools.Client api_base_url={self._config.api_base_url!r} "
            f"api_key={_mask_api_key(self._config.api_key)!r}>"
        )

    __str__ = __repr__

    # ---- primary surface ---------------------------------------------------

    def get(
        self,
        dataset: str,
        *,
        start: Any = None,
        end: Any = None,
        symbols: Sequence[str] | str | None = None,
        source: Source | str = Source.AUTO,
        force_refresh: bool = False,
    ) -> Result:
        """Fetch a date range, choosing the source(s) automatically.

        With nothing cached, the dataset's bulk export is downloaded once
        (or, for a recent-only range, the live API answers alone). With the
        export cached, it is read from disk — no rw-api call, no bandwidth
        charge — and topped up from the live endpoint with every row newer
        than the export's watermark. ``source="bulk"`` or ``"live"`` forces
        one path; ``result.route`` always says what happened.

        ``dataset`` is a name from :data:`rwpytools.DATASETS` (see
        :meth:`datasets`). ``symbols`` narrows to tickers — and is required
        for per-symbol datasets such as ``fx_daily``.
        """

        return run_sync(
            self._stack.session.get(
                dataset,
                start=start,
                end=end,
                symbols=symbols,
                source=source,
                force_refresh=force_refresh,
            )
        )

    def explain(
        self,
        dataset: str,
        *,
        start: Any = None,
        end: Any = None,
        symbols: Sequence[str] | str | None = None,
        source: Source | str = Source.AUTO,
        force_refresh: bool = False,
    ) -> RoutePlan:
        """Show how a request *would* be routed, without fetching rows."""

        return run_sync(
            self._stack.session.plan(
                dataset,
                start=start,
                end=end,
                symbols=symbols,
                source=source,
                force_refresh=force_refresh,
            )
        )

    def download(
        self, dataset: str, *, symbol: str | None = None, force_refresh: bool = False
    ) -> FetchResult:
        """Ensure a dataset's bulk export is cached; return its path.

        Downloads only when the export is missing, older than ``bulk_ttl``,
        or ``force_refresh`` is set. ``symbol`` selects the file of a
        per-symbol (``fx_daily``) or per-year (``ib_shortstock``) dataset.
        """

        return run_sync(
            self._stack.session.download(dataset, symbol=symbol, force_refresh=force_refresh)
        )

    @staticmethod
    def datasets() -> list[DatasetSpec]:
        """Every dataset :meth:`get` can route, with its bulk / live sources."""

        return list(DATASETS.values())

    def catalog(self, namespace: str) -> DatasetCatalogResponse:
        """rw-api's own bulk catalog for a namespace (``crypto``, ``fx``,
        ``macro``, ``equities``)."""

        return self.bulk.list_datasets(namespace)

    def status(self) -> dict[str, Any]:
        """``GET /v1/status`` — whether the API is up."""

        return self.live.status()

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Release any HTTP / cache resources. Idempotent."""

        run_sync(self._http.aclose())

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def session(self) -> Iterator[Client]:
        """Context manager that closes the client on exit."""

        try:
            yield self
        finally:
            self.close()


# ---------------------------------------------------------------------------
# Async Client
# ---------------------------------------------------------------------------


class AsyncClient:
    """Async-native Robot Wealth research data client — same surface as
    :class:`Client`, coroutine methods.

    Example::

        async with rwpytools.AsyncClient() as client:
            result = await client.get("vix", start="2024-01-01")
            spreads = await client.live.statarb_daily_top_spreads(gte="2024-01-01")
    """

    __doc__ = (__doc__ or "") + _CLIENT_PARAMS_DOC

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        cache_dir: Path | str | None = None,
        cache_max_bytes: int | None = None,
        config: ClientConfig | None = None,
        prompt_for_api_key: bool | None = None,
        **config_overrides: Any,
    ) -> None:
        self._stack = _Stack(
            _build_config(
                api_key,
                config=config,
                prompt_for_api_key=prompt_for_api_key,
                base_url=base_url,
                cache_dir=cache_dir,
                cache_max_bytes=cache_max_bytes,
                overrides=config_overrides,
            )
        )
        self._config = self._stack.config
        self._http = self._stack.http
        self._cache = self._stack.cache

        self.bulk = self._stack.bulk
        self.live = self._stack.live

        self.crypto = self._stack.crypto
        self.fx = self._stack.fx
        self.macro = self._stack.macro
        self.equity_factors = self._stack.equity_factors

    @property
    def config(self) -> ClientConfig:
        return self._config

    @property
    def cache(self) -> FilesystemCache:
        return self._cache

    def __repr__(self) -> str:
        return (
            f"<rwpytools.AsyncClient api_base_url={self._config.api_base_url!r} "
            f"api_key={_mask_api_key(self._config.api_key)!r}>"
        )

    __str__ = __repr__

    async def get(
        self,
        dataset: str,
        *,
        start: Any = None,
        end: Any = None,
        symbols: Sequence[str] | str | None = None,
        source: Source | str = Source.AUTO,
        force_refresh: bool = False,
    ) -> Result:
        """Auto-routed range fetch. See :meth:`Client.get`."""

        return await self._stack.session.get(
            dataset,
            start=start,
            end=end,
            symbols=symbols,
            source=source,
            force_refresh=force_refresh,
        )

    async def explain(
        self,
        dataset: str,
        *,
        start: Any = None,
        end: Any = None,
        symbols: Sequence[str] | str | None = None,
        source: Source | str = Source.AUTO,
        force_refresh: bool = False,
    ) -> RoutePlan:
        """Routing decision only, no rows fetched."""

        return await self._stack.session.plan(
            dataset,
            start=start,
            end=end,
            symbols=symbols,
            source=source,
            force_refresh=force_refresh,
        )

    async def download(
        self, dataset: str, *, symbol: str | None = None, force_refresh: bool = False
    ) -> FetchResult:
        """Cache a dataset's bulk export; return its path. See :meth:`Client.download`."""

        return await self._stack.session.download(
            dataset, symbol=symbol, force_refresh=force_refresh
        )

    @staticmethod
    def datasets() -> list[DatasetSpec]:
        return list(DATASETS.values())

    async def catalog(self, namespace: str) -> DatasetCatalogResponse:
        return await self._stack.bulk.list_datasets(namespace)

    async def status(self) -> dict[str, Any]:
        return await self._stack.live.status()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


__all__ = ["AsyncClient", "Client"]
