"""The value type every routed read returns.

A :class:`Result` is a DataFrame plus the provenance of its rows. The
provenance half is the point of :attr:`Result.route`: for a request that
straddled the export watermark you can see exactly which days came from the
cached bulk export and which came from the live API, whether the export was
downloaded or served from disk, and whether anything degraded.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .bulk import FetchResult
from .routing import RoutePlan


@dataclass(frozen=True, eq=False)
class Result:
    """Rows plus the routing decision that produced them."""

    dataset: str
    frame: pd.DataFrame
    #: How the request was split between the cached export and the live API.
    route: RoutePlan
    #: Date and symbol columns for this dataset (bulk naming).
    date_column: str | None = None
    symbol_column: str | None = None
    #: Rows contributed by each source — the seam, in numbers. ``live_rows``
    #: counts only rows that were not already in the bulk half.
    bulk_rows: int = 0
    live_rows: int = 0
    #: Bulk rows at the seam whose values were refreshed from the live
    #: route (an export's newest day can be provisional).
    seam_updates: int = 0
    #: Whether the bulk half was answered from the local cache. ``None``
    #: when no bulk read happened.
    served_from_cache: bool | None = None
    #: Last timestamp in the cached export (naive UTC), when known.
    watermark: pd.Timestamp | None = None
    #: The bulk objects that were read, one per file.
    bulk_objects: tuple[FetchResult, ...] = ()
    #: Non-fatal problems, e.g. a live endpoint that failed while the
    #: cached export could still answer.
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # ---- sequence-ish surface ------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate rows as plain dicts."""

        for record in self.frame.to_dict("records"):
            yield {str(k): v for k, v in record.items()}

    def __bool__(self) -> bool:
        return not self.frame.empty

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    @property
    def sources(self) -> tuple[str, ...]:
        """Which sources served this result, e.g. ``('bulk', 'live')``."""

        return self.route.sources

    # ---- conversions -----------------------------------------------------------

    def to_pandas(self) -> pd.DataFrame:
        """The rows as a DataFrame (the same object as :attr:`frame`)."""

        return self.frame

    @property
    def df(self) -> pd.DataFrame:
        """Shorthand for :meth:`to_pandas`."""

        return self.frame

    def to_records(self) -> list[dict[str, Any]]:
        """The rows as a list of plain dicts."""

        return list(self)

    # ---- provenance --------------------------------------------------------------

    def describe_route(self) -> str:
        """One line saying where these rows came from."""

        counts = f"{self.bulk_rows} bulk + {self.live_rows} live = {len(self)} rows"
        if self.seam_updates:
            counts += f" ({self.seam_updates} seam rows refreshed from live)"
        cache = (
            ""
            if self.served_from_cache is None
            else ("; bulk served from cache" if self.served_from_cache else "; bulk downloaded")
        )
        warned = f"; WARNING: {' | '.join(self.warnings)}" if self.warnings else ""
        return f"{self.route.describe()} | {counts}{cache}{warned}"

    def __repr__(self) -> str:
        sources = "+".join(self.sources) or "none"
        return (
            f"<rwpytools.Result dataset={self.dataset!r} rows={len(self)} "
            f"sources={sources} columns={len(self.columns)}>"
        )


__all__ = ["Result"]
