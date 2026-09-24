"""Bulk-download flow: rw-api signed URL → CDN → filesystem cache → DataFrame.

Two entry points:

* :meth:`BulkClient.ensure` — **cache first**. If the cell is already on
  disk (and, when a ``max_age`` is given, young enough) it is returned
  without any network call. This matters: rw-api charges an object's full
  size against the caller's bandwidth cap the moment it *issues* a signed
  URL, even if the bytes end up coming from the local cache. The routed
  :meth:`rwpytools.Client.get` path always goes through here, and tops the
  cached export up from the live API instead of re-downloading it.
* :meth:`BulkClient.fetch_one` — **validate against rw-api**:

  1. Ask rw-api for a signed URL on ``/v1/{namespace}/file`` for the
     given (dataset, schema [, symbol]) — this is the metered step.
  2. If our cache holds an entry whose size matches the server's,
     serve it (and mark it re-validated).
  3. Otherwise, make room in the cache (evicting LRU entries), stream
     the CDN URL into a temp file, atomically rename, write the meta
     sidecar.
  4. Return the cached path (or parsed DataFrame) to the caller.

Multi-fetch (``fetch_many``) issues each request independently —
there is no bulk endpoint on rw-api today. Concurrency is bounded by
``ClientConfig.download_concurrency``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import formats
from .cache import CacheEntryMeta, Cell, FilesystemCache
from .config import ClientConfig
from .errors import CacheError, RwApiError, SignedUrlExpiredError
from .http import _Http
from .schemas import (
    DatasetCatalogResponse,
    PresignedUrlResponse,
    SymbolsResponse,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    """Outcome of fetching a single (namespace, dataset, schema [, symbol]) cell."""

    namespace: str
    dataset: str
    schema: str
    symbol: str | None
    bucket: str
    object_name: str
    path: Path
    size: int
    served_from_cache: bool
    #: Whether rw-api was called (a metered signed-URL issue). ``False``
    #: for a pure cache hit from :meth:`BulkClient.ensure`.
    api_called: bool = True
    #: Unix time the cached object was fetched or last re-validated.
    fetched_at: float | None = None

    @property
    def cell(self) -> Cell:
        return (self.namespace, self.dataset, self.schema, self.symbol)


@dataclass(frozen=True)
class BulkFetchBatch:
    """Aggregate result of a :meth:`BulkClient.fetch_many_partial` call.

    Both ``succeeded`` and ``failed`` are aligned with the input
    request order: ``succeeded[i]`` is the result for ``requests[i]``
    when ``failed[i]`` is ``None``, and vice versa. ``succeeded`` and
    ``failed`` together cover every request exactly once.
    """

    succeeded: list[FetchResult | None]
    failed: list[BaseException | None]

    def all_succeeded(self) -> bool:
        return all(f is None for f in self.failed)

    def succeeded_results(self) -> list[FetchResult]:
        """Return only the successful results, in input order."""

        return [r for r in self.succeeded if r is not None]

    def failures(
        self,
    ) -> list[tuple[int, BaseException]]:
        """Return ``(index, exception)`` pairs for failed requests."""

        return [(i, f) for i, f in enumerate(self.failed) if f is not None]


@dataclass(frozen=True)
class FetchRequest:
    """One fetch within a :meth:`BulkClient.fetch_many` call.

    ``namespace`` is provided once on the call, so it is not part of the
    request struct itself.
    """

    dataset: str
    schema: str
    symbol: str | None = None

    #: Bumped when the cache_key shape changes so old keys never collide
    #: with new ones in caller-side dictionaries.
    CACHE_KEY_VERSION: str = "v1"

    def cache_key(self) -> str:
        """Stable key for this request, suitable as a dict key.

        The ``v1`` prefix lets us evolve the format (e.g. add a tier
        field) without silently colliding with caller-side caches.
        """

        if self.symbol is not None:
            return f"{self.CACHE_KEY_VERSION}:{self.dataset}/{self.schema}/{self.symbol}"
        return f"{self.CACHE_KEY_VERSION}:{self.dataset}/{self.schema}"


class BulkClient:
    """Async core of the bulk-download flow.

    ``Client`` (the sync façade) wraps this and runs each call to
    completion via :func:`rwpytools.http.run_sync`. ``AsyncClient``
    exposes it directly.
    """

    def __init__(self, config: ClientConfig, *, http: _Http, cache: FilesystemCache) -> None:
        self._config = config
        self._http = http
        self._cache = cache

    @property
    def cache(self) -> FilesystemCache:
        return self._cache

    async def aclose(self) -> None:
        await self._http.aclose()

    # ----- cache-first -----------------------------------------------------

    def cached(
        self, namespace: str, *, dataset: str, schema: str, symbol: str | None = None
    ) -> CacheEntryMeta | None:
        """The cached export for a cell, if any. Never touches the network."""

        return self._cache.lookup_cell((namespace, dataset, schema, symbol))

    async def ensure(
        self,
        namespace: str,
        *,
        dataset: str,
        schema: str,
        symbol: str | None = None,
        force_refresh: bool = False,
        max_age: float | None = None,
        expires_in: int | None = None,
    ) -> FetchResult:
        """Return the cell's export from cache, downloading only if needed.

        A cached copy is used as-is — no rw-api call, no bandwidth charge —
        unless ``force_refresh`` is set or it is older than ``max_age``
        seconds. Otherwise this is :meth:`fetch_one`, which re-validates the
        cached copy by size and downloads only when it changed.
        """

        if not force_refresh:
            meta = self.cached(namespace, dataset=dataset, schema=schema, symbol=symbol)
            if meta is not None and (max_age is None or meta.age() <= max_age):
                log.debug("cache hit (no API call): %s/%s", meta.bucket, meta.object_name)
                return FetchResult(
                    namespace=namespace,
                    dataset=dataset,
                    schema=schema,
                    symbol=symbol,
                    bucket=meta.bucket,
                    object_name=meta.object_name,
                    path=self._cache.path_for(meta.bucket, meta.object_name),
                    size=meta.size,
                    served_from_cache=True,
                    api_called=False,
                    fetched_at=meta.fetched_at,
                )
        return await self.fetch_one(
            namespace,
            dataset=dataset,
            schema=schema,
            symbol=symbol,
            force_refresh=force_refresh,
            expires_in=expires_in,
        )

    def watermark(self, bucket: str, object_name: str, column: str) -> pd.Timestamp | None:
        """The last value of ``column`` in a cached export.

        Computed once by scanning just that column, then remembered in the
        cache sidecar so later calls (and :meth:`rwpytools.Client.explain`)
        cost nothing. A re-download writes a fresh sidecar, which drops the
        remembered value.
        """

        meta = self._cache.lookup(bucket, object_name)
        if meta is not None and meta.watermark_column == column:
            return None if meta.watermark is None else pd.Timestamp(meta.watermark)
        path = self._cache.path_for(bucket, object_name)
        latest = formats.max_timestamp(path, object_name=object_name, column=column)
        if meta is not None:
            self._cache.update_meta(
                meta,
                watermark_column=column,
                watermark=None if latest is None else latest.isoformat(),
            )
        return latest

    # ----- catalog ---------------------------------------------------------

    async def list_datasets(self, namespace: str) -> DatasetCatalogResponse:
        body = await self._http.get_json(f"/v1/{namespace}/datasets")
        return DatasetCatalogResponse.model_validate(body)

    async def list_symbols(
        self,
        namespace: str,
        *,
        dataset: str,
        schema: str | None = None,
    ) -> SymbolsResponse:
        params: dict[str, str] = {"dataset": dataset}
        if schema is not None:
            params["schema"] = schema
        body = await self._http.get_json(f"/v1/{namespace}/symbols", params=params)
        return SymbolsResponse.model_validate(body)

    # ----- single fetch ----------------------------------------------------

    async def fetch_one(
        self,
        namespace: str,
        *,
        dataset: str,
        schema: str,
        symbol: str | None = None,
        force_refresh: bool = False,
        expires_in: int | None = None,
    ) -> FetchResult:
        params: dict[str, Any] = {
            "dataset": dataset,
            "schema": schema,
            "expires_in": expires_in or self._config.default_expires_in,
        }
        if symbol is not None:
            params["symbol"] = symbol

        async def _request_signed_url() -> Any:
            body = await self._http.get_json(f"/v1/{namespace}/file", params=params)
            return PresignedUrlResponse.model_validate(body).data

        signed = await _request_signed_url()
        cell: Cell = (namespace, dataset, schema, symbol)

        cache_path = self._cache.path_for(signed.bucket, signed.object)
        cached = self._cache.lookup(signed.bucket, signed.object)
        if not force_refresh and cached is not None and cached.size == signed.size:
            log.debug("cache hit (re-validated): %s/%s", signed.bucket, signed.object)
            # rw-api vouched for the cached copy: restart its TTL clock and
            # record the cell so the next lookup needs no API call at all.
            now = time.time()
            self._cache.update_meta(cached, fetched_at=now, cell=cell)
            return FetchResult(
                namespace=namespace,
                dataset=dataset,
                schema=schema,
                symbol=symbol,
                bucket=signed.bucket,
                object_name=signed.object,
                path=cache_path,
                size=signed.size,
                served_from_cache=True,
                fetched_at=now,
            )

        # reserve() makes room, streams into a .part tempfile, then on
        # successful exit atomically renames into place AND writes the
        # sidecar. If the download or rename fails the .part is
        # cleaned up and the cache is unchanged.
        async def _download(presigned: Any) -> int:
            with self._cache.reserve(
                bucket=presigned.bucket,
                object_name=presigned.object,
                size=presigned.size,
                cell=cell,
            ) as tmp_path:
                return await self._http.download_signed_url_to_path(
                    presigned.url, tmp_path, expected_size=presigned.size
                )

        try:
            bytes_written = await _download(signed)
        except SignedUrlExpiredError:
            # The URL expired between when rw-api minted it and when the
            # CDN saw it. Mint a fresh one and try exactly once more —
            # repeated expiry implies a clock skew bug we should let
            # surface rather than retry into.
            log.info(
                "signed URL expired for %s/%s; re-minting once",
                signed.bucket,
                signed.object,
            )
            signed = await _request_signed_url()
            bytes_written = await _download(signed)
        if bytes_written != signed.size:
            # The cache.reserve() commit succeeded with the API-reported
            # size; if the actual download diverged that's a server /
            # CDN bug we should not silently cache.
            with contextlib.suppress(CacheError):
                self._cache.evict(signed.bucket, signed.object)
            raise RwApiError(
                f"CDN returned {bytes_written} bytes but rw-api advertised "
                f"{signed.size} for {signed.bucket}/{signed.object}; refusing "
                f"to cache a truncated/oversized object."
            )
        log.debug(
            "downloaded %d bytes for %s/%s",
            bytes_written,
            signed.bucket,
            signed.object,
        )
        return FetchResult(
            namespace=namespace,
            dataset=dataset,
            schema=schema,
            symbol=symbol,
            bucket=signed.bucket,
            object_name=signed.object,
            path=cache_path,
            size=bytes_written,
            served_from_cache=False,
            fetched_at=time.time(),
        )

    # ----- bulk fetch ------------------------------------------------------

    async def fetch_many(
        self,
        namespace: str,
        requests: Sequence[FetchRequest],
        *,
        force_refresh: bool = False,
        expires_in: int | None = None,
    ) -> list[FetchResult]:
        """Fetch many (dataset, schema [, symbol]) cells from the same
        namespace, parallel-bounded by ``config.download_concurrency``.

        rw-api has no batch-presigned endpoint today — each request is
        an independent ``/v1/{ns}/file`` round-trip. We don't try to
        deduplicate identical requests; if the caller passes the same
        cell twice they get two FetchResults.

        On the first failure, in-flight downloads are cancelled and the
        exception propagates. Use :meth:`fetch_many_partial` to collect
        partial successes instead.
        """

        if not requests:
            return []

        sem = asyncio.Semaphore(self._config.download_concurrency)

        async def _one(req: FetchRequest) -> FetchResult:
            async with sem:
                return await self.fetch_one(
                    namespace,
                    dataset=req.dataset,
                    schema=req.schema,
                    symbol=req.symbol,
                    force_refresh=force_refresh,
                    expires_in=expires_in,
                )

        return list(await asyncio.gather(*(_one(r) for r in requests)))

    async def fetch_many_partial(
        self,
        namespace: str,
        requests: Sequence[FetchRequest],
        *,
        force_refresh: bool = False,
        expires_in: int | None = None,
    ) -> BulkFetchBatch:
        """Fetch many cells, surfacing per-request successes and failures.

        Unlike :meth:`fetch_many`, one failing request does not abort
        the rest of the batch. The returned :class:`BulkFetchBatch`
        carries both arrays aligned with the input ``requests``
        ordering; callers can re-issue only the failed entries.
        """

        if not requests:
            return BulkFetchBatch(succeeded=[], failed=[])

        sem = asyncio.Semaphore(self._config.download_concurrency)

        async def _one(req: FetchRequest) -> FetchResult:
            async with sem:
                return await self.fetch_one(
                    namespace,
                    dataset=req.dataset,
                    schema=req.schema,
                    symbol=req.symbol,
                    force_refresh=force_refresh,
                    expires_in=expires_in,
                )

        raw = await asyncio.gather(*(_one(r) for r in requests), return_exceptions=True)
        succeeded: list[FetchResult | None] = []
        failed: list[BaseException | None] = []
        for entry in raw:
            if isinstance(entry, BaseException):
                succeeded.append(None)
                failed.append(entry)
            else:
                succeeded.append(entry)
                failed.append(None)
        return BulkFetchBatch(succeeded=succeeded, failed=failed)

    # ----- frame loaders ---------------------------------------------------

    async def fetch_one_as_frame(
        self,
        namespace: str,
        *,
        dataset: str,
        schema: str,
        symbol: str | None = None,
        force_refresh: bool = False,
        expires_in: int | None = None,
    ) -> pd.DataFrame:
        result = await self.fetch_one(
            namespace,
            dataset=dataset,
            schema=schema,
            symbol=symbol,
            force_refresh=force_refresh,
            expires_in=expires_in,
        )
        return formats.read_auto(result.path, object_name=result.object_name)

    async def fetch_many_as_frames(
        self,
        namespace: str,
        requests: Sequence[FetchRequest],
        *,
        force_refresh: bool = False,
        expires_in: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return a dict keyed by ``FetchRequest.cache_key()``.

        For symbol-templated requests this gives stable keys like
        ``"pairs/ohlcv-1d/EURUSD"``. For symbol-less requests it's
        ``"binance.spot/ohlcv-1h"``.
        """

        results = await self.fetch_many(
            namespace,
            requests,
            force_refresh=force_refresh,
            expires_in=expires_in,
        )
        out: dict[str, pd.DataFrame] = {}
        for req, res in zip(requests, results, strict=False):
            out[req.cache_key()] = formats.read_auto(res.path, object_name=res.object_name)
        return out


__all__ = ["BulkClient", "BulkFetchBatch", "FetchRequest", "FetchResult"]
