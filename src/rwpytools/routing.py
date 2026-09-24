"""The routing decision: cached bulk export, live API, or both.

rw-api publishes most research datasets twice:

* a **bulk export** — one file per dataset (feather / parquet / csv),
  downloaded through a signed Cloud CDN URL and cached on disk; and
* a **live endpoint** — a cursor-paginated JSON route backed by the
  PostgreSQL warehouse, which always holds the newest rows.

rw-api does not publish a coverage catalog, so the export's watermark —
the last date it contains — is read from the cached file itself. With that
watermark ``W`` a request for ``[start, end]`` is split:

=============================================  ===============================
situation                                      route
=============================================  ===============================
dataset has no live companion                  all bulk
dataset has no bulk export                     all live
nothing cached, ``start`` within the live      all live (no multi-GB download
window (``live_only_window_days``)             for a few recent days)
nothing cached otherwise                       download bulk, then top up live
cached, ``end`` before ``W``                   all bulk (from cache)
cached, ``start`` after ``W``                  all live
cached, range straddles ``W``                  bulk through ``W`` + live diff
                                               from ``W``, joined
cached, ``W`` older than                       refresh bulk, then top up live
``live_diff_max_days`` and past ``bulk_ttl``
=============================================  ===============================

The live half starts **on** ``W`` rather than the day after: intraday
exports can end part-way through ``W``, so the seam day is fetched from
both sides and :mod:`rwpytools.session` drops live rows whose key is
already in the bulk half.

This module is pure: no HTTP, no filesystem, no clock (``today`` is an
argument). Every rule above is a function of its inputs, which is what
makes the routing matrix cheap to test exhaustively.
"""

from __future__ import annotations

import datetime as _dt
import enum
from dataclasses import dataclass
from typing import Any

from .errors import RoutingError

ONE_DAY = _dt.timedelta(days=1)


class Source(str, enum.Enum):
    """Where rows may come from. ``AUTO`` is the routing decision itself.

    Subclasses ``str`` so callers can pass the plain strings the public
    API advertises: ``client.get(..., source="bulk")``.
    """

    AUTO = "auto"
    BULK = "bulk"
    LIVE = "live"

    @classmethod
    def parse(cls, value: Source | str | None) -> Source:
        if value is None:
            return cls.AUTO
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(repr(m.value) for m in cls)
            raise RoutingError(f"source must be one of {allowed}; got {value!r}") from exc


class BulkAction(str, enum.Enum):
    """What the bulk half of a plan has to do before it can be read."""

    CACHE = "cache"  # serve the cached export as-is
    DOWNLOAD = "download"  # nothing cached: fetch the export
    REFRESH = "refresh"  # cached, but re-validate / re-download it first


def as_date(value: Any, *, field: str = "date") -> _dt.date | None:
    """Coerce a user-supplied bound to a ``datetime.date``.

    Accepts ``None`` (unbounded), ``datetime.date``, ``datetime.datetime``
    (date part taken), ``pandas.Timestamp`` and ISO-8601 strings —
    including a full timestamp, whose date part is used.
    """

    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        head = text.replace("Z", "").split("T")[0].split(" ")[0]
        try:
            return _dt.date.fromisoformat(head)
        except ValueError as exc:
            raise RoutingError(
                f"{field} must be an ISO date (YYYY-MM-DD) or datetime; got {value!r}"
            ) from exc
    raise RoutingError(f"{field} must be a date, datetime or ISO string; got {value!r}")


@dataclass(frozen=True)
class DateRange:
    """A closed, inclusive date interval. ``None`` on either side is open."""

    start: _dt.date | None = None
    end: _dt.date | None = None

    def is_empty(self) -> bool:
        return self.start is not None and self.end is not None and self.start > self.end

    def contains(self, value: _dt.date) -> bool:
        after_start = self.start is None or value >= self.start
        before_end = self.end is None or value <= self.end
        return after_start and before_end

    def describe(self) -> str:
        left = self.start.isoformat() if self.start else "-inf"
        right = self.end.isoformat() if self.end else "now"
        return f"[{left} .. {right}]"

    def to_json(self) -> dict[str, str | None]:
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
        }


@dataclass(frozen=True)
class BulkState:
    """What the local cache knows about a dataset's bulk export.

    ``watermark`` is the last date present in the cached export, or
    ``None`` when nothing is cached (or the watermark is not yet known).
    ``expired`` means the cached copy is past ``bulk_ttl`` and must be
    re-validated before it is trusted on its own.
    """

    cached: bool = False
    watermark: _dt.date | None = None
    expired: bool = False


@dataclass(frozen=True)
class RoutePlan:
    """The routing decision for one request — the thing callers can inspect.

    A data library must never be silently magic about where its numbers
    came from, so this is attached to every :class:`rwpytools.Result` and
    logged at INFO on each ``get()``.
    """

    dataset: str
    requested: DateRange
    #: Last date in the cached export; ``None`` when unknown / not cached.
    watermark: _dt.date | None
    bulk_range: DateRange | None
    live_range: DateRange | None
    #: First date served by the live endpoint when the request straddles
    #: the watermark; ``None`` when a single source covers everything.
    split_at: _dt.date | None
    #: What the bulk half does first. ``None`` when bulk is not used.
    bulk_action: BulkAction | None
    reason: str
    forced: bool

    @property
    def sources(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.bulk_range is not None:
            out.append(Source.BULK.value)
        if self.live_range is not None:
            out.append(Source.LIVE.value)
        return tuple(out)

    @property
    def uses_bulk(self) -> bool:
        return self.bulk_range is not None

    @property
    def uses_live(self) -> bool:
        return self.live_range is not None

    @property
    def needs_download(self) -> bool:
        """Whether the bulk half must hit rw-api before it can be read."""

        return self.bulk_action in (BulkAction.DOWNLOAD, BulkAction.REFRESH)

    def describe(self) -> str:
        parts = [f"{self.dataset} {self.requested.describe()} ->"]
        if self.bulk_range is not None:
            action = f" ({self.bulk_action.value})" if self.bulk_action else ""
            parts.append(f"bulk{self.bulk_range.describe()}{action}")
        if self.live_range is not None:
            parts.append(f"live{self.live_range.describe()}")
        if not self.sources:
            parts.append("nothing")
        if self.split_at is not None:
            parts.append(f"(split at {self.split_at.isoformat()})")
        parts.append(f"— {self.reason}")
        return " ".join(parts)

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "requested": self.requested.to_json(),
            "watermark": self.watermark.isoformat() if self.watermark else None,
            "bulk_range": self.bulk_range.to_json() if self.bulk_range else None,
            "live_range": self.live_range.to_json() if self.live_range else None,
            "split_at": self.split_at.isoformat() if self.split_at else None,
            "bulk_action": self.bulk_action.value if self.bulk_action else None,
            "sources": list(self.sources),
            "reason": self.reason,
            "forced": self.forced,
        }

    def __str__(self) -> str:
        return self.describe()


def plan_route(
    *,
    dataset: str,
    start: _dt.date | None = None,
    end: _dt.date | None = None,
    has_bulk: bool,
    has_live: bool,
    bulk: BulkState | None = None,
    today: _dt.date,
    source: Source | str | None = Source.AUTO,
    force_refresh: bool = False,
    live_only_window_days: int = 30,
    live_diff_max_days: int = 30,
) -> RoutePlan:
    """Decide which source(s) serve ``[start, end]`` for ``dataset``.

    ``bulk`` describes the cached export (see :class:`BulkState`); when a
    plan says :attr:`BulkAction.DOWNLOAD` or :attr:`BulkAction.REFRESH`,
    the session performs it and plans again with the new watermark, so the
    final plan always carries the real split.
    """

    chosen = Source.parse(source)
    state = bulk or BulkState()
    requested = DateRange(start, end)

    if requested.is_empty():
        raise RoutingError(f"start ({start}) is after end ({end}) — the requested range is empty")

    def _plan(
        *,
        bulk_range: DateRange | None,
        live_range: DateRange | None,
        reason: str,
        action: BulkAction | None = None,
        split_at: _dt.date | None = None,
        forced: bool = False,
    ) -> RoutePlan:
        return RoutePlan(
            dataset=dataset,
            requested=requested,
            watermark=state.watermark if state.cached else None,
            bulk_range=bulk_range,
            live_range=live_range,
            split_at=split_at,
            bulk_action=action if bulk_range is not None else None,
            reason=reason,
            forced=forced,
        )

    def _cached_action() -> BulkAction:
        if force_refresh:
            return BulkAction.REFRESH
        if not state.cached:
            return BulkAction.DOWNLOAD
        return BulkAction.CACHE

    # --- forced -------------------------------------------------------------

    if chosen is Source.BULK:
        if not has_bulk:
            raise RoutingError(f"{dataset} has no bulk export; use source='live' or 'auto'")
        return _plan(
            bulk_range=requested,
            live_range=None,
            action=_cached_action(),
            reason="source='bulk' forced; rows newer than the export watermark are not included",
            forced=True,
        )

    if chosen is Source.LIVE:
        if not has_live:
            raise RoutingError(f"{dataset} has no live endpoint; use source='bulk' or 'auto'")
        return _plan(
            bulk_range=None,
            live_range=requested,
            reason="source='live' forced; every row comes from the live API",
            forced=True,
        )

    # --- auto: single-source datasets -----------------------------------------

    if not has_bulk and not has_live:
        raise RoutingError(f"{dataset} has neither a bulk export nor a live endpoint")

    if not has_live:
        action = _cached_action()
        if action is BulkAction.CACHE and state.expired:
            action = BulkAction.REFRESH
        reason = {
            BulkAction.CACHE: "bulk-only dataset; serving the cached export",
            BulkAction.DOWNLOAD: "bulk-only dataset; no cached export, fetching it",
            BulkAction.REFRESH: "bulk-only dataset; re-validating the cached export",
        }[action]
        return _plan(bulk_range=requested, live_range=None, action=action, reason=reason)

    if not has_bulk:
        return _plan(
            bulk_range=None,
            live_range=requested,
            reason="live-only dataset; no bulk export exists",
        )

    # --- auto: bulk + live ----------------------------------------------------

    if force_refresh:
        return _plan(
            bulk_range=requested,
            live_range=None,
            action=BulkAction.REFRESH,
            reason="force_refresh=True; re-downloading the export before topping up live",
        )

    if state.cached and state.watermark is None:
        # Nothing to diff against: a snapshot export (no date column) or one
        # whose date column held no values. Serve it like a bulk-only dataset.
        action = BulkAction.REFRESH if state.expired else BulkAction.CACHE
        return _plan(
            bulk_range=requested,
            live_range=None,
            action=action,
            reason=(
                "export has no date watermark to diff from; serving the cached export "
                "(pass source='live' for the live snapshot)"
                if action is BulkAction.CACHE
                else "export has no date watermark; re-validating the cached export"
            ),
        )

    if not state.cached:
        recent = start is not None and (today - start).days <= live_only_window_days
        if recent:
            return _plan(
                bulk_range=None,
                live_range=requested,
                reason=(
                    f"nothing cached and the range starts within {live_only_window_days} "
                    f"days; serving everything live instead of downloading the export"
                ),
            )
        return _plan(
            bulk_range=requested,
            live_range=None,
            action=BulkAction.DOWNLOAD,
            reason="no cached export; fetching it, then topping up live",
        )

    watermark = state.watermark
    assert watermark is not None  # both None cases returned above
    lag_days = (today - watermark).days
    stale = lag_days > live_diff_max_days and (end is None or end > watermark)
    # Only re-validate once the cached copy is past its TTL: an export that
    # is itself behind (the upstream job stopped) would otherwise be
    # re-issued — and re-charged against the bandwidth cap — on every call.
    if stale and state.expired:
        return _plan(
            bulk_range=requested,
            live_range=None,
            action=BulkAction.REFRESH,
            reason=(
                f"cached export ends {watermark.isoformat()} ({lag_days} days ago, more than "
                f"live_diff_max_days={live_diff_max_days}); refreshing it before topping up live"
            ),
        )
    lag_note = (
        f" (the export itself is {lag_days} days behind; the live diff covers the gap)"
        if stale
        else ""
    )

    # The request ends before the watermark: the export has it all.
    if end is not None and end < watermark:
        return _plan(
            bulk_range=requested,
            live_range=None,
            action=BulkAction.CACHE,
            reason=(
                f"requested range ends before the export watermark "
                f"({watermark.isoformat()}); serving everything from the cached export"
            ),
        )

    # The request begins after the watermark: nothing in the export helps.
    if start is not None and start > watermark:
        return _plan(
            bulk_range=None,
            live_range=requested,
            reason=(
                f"requested range starts after the export watermark "
                f"({watermark.isoformat()}); serving everything live"
            ),
        )

    # Straddle: bulk through the watermark, live diff from the watermark day.
    return _plan(
        bulk_range=DateRange(start, watermark),
        live_range=DateRange(watermark, end),
        split_at=watermark,
        action=BulkAction.CACHE,
        reason=(
            f"cached export ends {watermark.isoformat()}; bulk through the watermark, "
            f"live diff from {watermark.isoformat()}{lag_note}"
        ),
    )


__all__ = [
    "ONE_DAY",
    "BulkAction",
    "BulkState",
    "DateRange",
    "RoutePlan",
    "Source",
    "as_date",
    "plan_route",
]
