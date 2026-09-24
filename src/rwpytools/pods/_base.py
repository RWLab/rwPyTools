"""Base classes shared by pod accessors.

Each concrete pod lives twice: once as ``Async{Name}Pod`` (async
methods, used by :class:`rwpytools.AsyncClient`) and once as the
matching sync-named class (e.g. :class:`CryptoPod`) that wraps the
async one via :func:`rwpytools.http.run_sync` for use from
:class:`rwpytools.Client`.

The two-class design exists because:

1. ``Client`` (sync) and ``AsyncClient`` (async) have first-class
   pod accessors; users on either side get full method parity.
2. mypy ``--strict`` insists on explicit signatures — auto-generating
   sync wrappers via decorators / metaclasses hides them from the
   type checker. The boilerplate is the cost of strict typing.
3. The async class owns the actual API contract; the sync class is
   trivially derivable but written out so IDE auto-complete and ``help()``
   work the same on both.

Every data method routes through :class:`rwpytools.session.AsyncSession`,
so it accepts the same overrides as :meth:`rwpytools.Client.get`:

* ``source`` — ``"auto"`` (default: cached export topped up live),
  ``"bulk"`` (cached export only) or ``"live"`` (live API only);
* ``start`` / ``end`` — inclusive date bounds;
* ``symbols`` — restrict to these tickers;
* ``force_refresh`` — re-download the export first.

When adding a method, add it to BOTH classes.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from typing import Generic, TypeVar

import pandas as pd

from ..bulk import BulkClient
from ..http import run_sync
from ..routing import Source
from ..schemas import DatasetCatalogResponse
from ..session import AsyncSession

#: ``source=`` argument: a :class:`~rwpytools.routing.Source` or its string.
SourceLike = Source | str
#: ``start=`` / ``end=`` argument: ISO string, date or datetime.
DateLike = str | _dt.date | None
#: ``symbols=`` argument: a list of tickers or a comma-separated string.
SymbolsLike = Sequence[str] | str | None


class AsyncPodAccessor:
    """Common helpers for per-namespace async accessors."""

    #: API namespace this accessor talks to (``crypto``, ``fx``, ...).
    namespace: str = ""

    def __init__(self, bulk: BulkClient, session: AsyncSession) -> None:
        if not self.namespace:
            raise RuntimeError(f"{type(self).__name__} must set the `namespace` class attribute")
        self._bulk = bulk
        self._session = session

    async def _get(
        self,
        dataset: str,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        result = await self._session.get(
            dataset,
            start=start,
            end=end,
            symbols=symbols,
            source=source,
            force_refresh=force_refresh,
        )
        return result.frame

    # ---- catalog -----------------------------------------------------------

    async def list_datasets(self) -> DatasetCatalogResponse:
        """Return rw-api's (dataset, schema) catalog for this namespace."""

        return await self._bulk.list_datasets(self.namespace)


A = TypeVar("A", bound=AsyncPodAccessor)


class SyncPodAccessor(Generic[A]):
    """Synchronous wrapper around an :class:`AsyncPodAccessor` instance.

    Subclasses re-declare every public method with explicit type hints
    and delegate to ``run_sync(self._async.method(...))``. They MUST
    NOT call into ``self._async`` from outside ``run_sync`` (would
    block on awaiting a coroutine), and the ``run_sync`` wrapper
    handles loop creation / teardown.
    """

    namespace: str = ""

    def __init__(self, async_pod: A) -> None:
        self._async: A = async_pod
        self.namespace = async_pod.namespace

    def list_datasets(self) -> DatasetCatalogResponse:
        """Return rw-api's (dataset, schema) catalog for this namespace."""

        return run_sync(self._async.list_datasets())


__all__ = [
    "A",
    "AsyncPodAccessor",
    "DateLike",
    "SourceLike",
    "SymbolsLike",
    "SyncPodAccessor",
]
