"""Executes a :class:`~rwpytools.routing.RoutePlan` against both sources.

:mod:`rwpytools.routing` decides *what* to do; this module does it:

1. look the dataset up in :mod:`rwpytools.datasets` and inspect the local
   cache — which exports are on disk, how old, and their watermark — with
   **no network call**;
2. plan the split; if the plan needs the export downloaded or refreshed,
   do that and plan again against the real watermark;
3. read the bulk half from disk and fetch the live half **concurrently**;
4. conform the live rows to the export's schema, refresh the seam rows both
   sides returned with the live values, append the rest, and return one
   :class:`~rwpytools.result.Result` carrying the plan with it.

The routing decision is logged at INFO on every call. A library that
silently chooses your data source is a library you cannot debug.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime as _dt
import logging
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
import pyarrow as pa

from . import formats
from .bulk import BulkClient, FetchResult
from .cache import CacheEntryMeta
from .config import ClientConfig
from .datasets import DatasetSpec, Template, get_spec
from .errors import CacheError, RateLimitError, RoutingError, RwApiError, ServerError
from .live import AsyncLiveClient
from .result import Result
from .routing import (
    BulkAction,
    BulkState,
    DateRange,
    RoutePlan,
    Source,
    as_date,
    plan_route,
)

log = logging.getLogger(__name__)

#: Passed as ``live_diff_max_days`` when re-planning right after a refresh:
#: the export was just re-validated, so an old watermark means the export
#: itself is behind, and refreshing again would change nothing.
_NO_REFRESH = 10**6


class AsyncSession:
    """The auto-routing engine shared by ``Client`` and ``AsyncClient``."""

    def __init__(
        self,
        config: ClientConfig,
        *,
        bulk: BulkClient,
        live: AsyncLiveClient,
        today: Callable[[], _dt.date] | None = None,
    ) -> None:
        self._config = config
        self._bulk = bulk
        self._live = live
        self._today = today or _dt.date.today

    # ---- planning -----------------------------------------------------------

    async def plan(
        self,
        dataset: str,
        *,
        start: Any = None,
        end: Any = None,
        symbols: Sequence[str] | str | None = None,
        source: Source | str | None = Source.AUTO,
        force_refresh: bool = False,
    ) -> RoutePlan:
        """Resolve the routing decision without fetching any rows.

        Reads only local cache metadata (and, once per export, its date
        column to find the watermark). Useful as a dry run:
        ``print(client.explain("binance_spot_1h", start="2024-01-01"))``.
        """

        spec = get_spec(dataset)
        today = self._today()
        start_d, end_d = as_date(start, field="start"), as_date(end, field="end")
        cells = self._cells(spec, start_d, end_d, _symbol_list(symbols), today)
        state, _ = self._bulk_state(spec, cells)
        return self._plan(spec, start_d, end_d, state, today, source, force_refresh)

    def _plan(
        self,
        spec: DatasetSpec,
        start: _dt.date | None,
        end: _dt.date | None,
        state: BulkState,
        today: _dt.date,
        source: Source | str | None,
        force_refresh: bool,
        *,
        live_diff_max_days: int | None = None,
    ) -> RoutePlan:
        return plan_route(
            dataset=spec.name,
            start=start,
            end=end,
            has_bulk=spec.has_bulk,
            has_live=spec.has_live,
            bulk=state,
            today=today,
            source=source,
            force_refresh=force_refresh,
            live_only_window_days=self._config.live_only_window_days,
            live_diff_max_days=(
                self._config.live_diff_max_days
                if live_diff_max_days is None
                else live_diff_max_days
            ),
        )

    # ---- the routed read ----------------------------------------------------

    async def get(
        self,
        dataset: str,
        *,
        start: Any = None,
        end: Any = None,
        symbols: Sequence[str] | str | None = None,
        source: Source | str | None = Source.AUTO,
        force_refresh: bool = False,
    ) -> Result:
        """Fetch a range, routing each part of it to the right source."""

        spec = get_spec(dataset)
        chosen = Source.parse(source)
        today = self._today()
        start_d, end_d = as_date(start, field="start"), as_date(end, field="end")
        syms = _symbol_list(symbols)
        cells = self._cells(spec, start_d, end_d, syms, today)

        state, watermark = self._bulk_state(spec, cells)
        route = self._plan(spec, start_d, end_d, state, today, chosen, force_refresh)
        fetched: dict[str | None, FetchResult] = {}

        if route.needs_download:
            log.info("route: %s", route.describe())
            fetched = await self._ensure_cells(
                spec, cells, hard=force_refresh, revalidate=route.bulk_action is BulkAction.REFRESH
            )
            state, watermark = self._bulk_state(spec, cells)
            final = self._plan(
                spec,
                start_d,
                end_d,
                dataclasses.replace(state, expired=False),
                today,
                chosen,
                False,
                live_diff_max_days=_NO_REFRESH,
            )
            route = dataclasses.replace(
                final,
                bulk_action=route.bulk_action if final.uses_bulk else None,
                reason=f"{route.reason}; then: {final.reason}",
            )

        log.info("route: %s", route.describe())

        bulk_task = self._read_bulk(spec, cells, route, syms, fetched) if route.uses_bulk else None
        live_task = self._read_live(spec, route, syms, today) if route.uses_live else None

        bulk_df: pd.DataFrame | None = None
        objects: tuple[FetchResult, ...] = ()
        live_df: pd.DataFrame | None = None
        warnings: list[str] = []

        live_outcome: pd.DataFrame | BaseException | None = None
        if bulk_task is not None and live_task is not None:
            (bulk_df, objects), live_outcome = await asyncio.gather(bulk_task, _capture(live_task))
        elif bulk_task is not None:
            bulk_df, objects = await bulk_task
        elif live_task is not None:
            live_df = await live_task

        if isinstance(live_outcome, BaseException):
            if route.forced or not _degradable(live_outcome):
                raise live_outcome
            message = (
                f"live endpoint {spec.live.path if spec.live else '?'} failed "
                f"({type(live_outcome).__name__}: {live_outcome}); returning the cached "
                f"export only, which ends {route.watermark}"
            )
            log.warning("%s: %s", spec.name, message)
            warnings.append(message)
        elif live_outcome is not None:
            live_df = live_outcome

        template = self._template(spec, cells, bulk_df)
        conformed = _conform(live_df, template, spec) if live_df is not None else None
        frame, live_rows, seam_updates = _merge(bulk_df, conformed, spec)

        result = Result(
            dataset=spec.name,
            frame=frame,
            route=route,
            date_column=spec.date_column,
            symbol_column=spec.symbol_column,
            bulk_rows=0 if bulk_df is None else len(bulk_df),
            live_rows=live_rows,
            seam_updates=seam_updates,
            served_from_cache=(None if not objects else all(o.served_from_cache for o in objects)),
            watermark=watermark,
            bulk_objects=objects,
            warnings=tuple(warnings),
        )
        log.info("result: %s", result.describe_route())
        return result

    async def download(
        self,
        dataset: str,
        *,
        symbol: str | None = None,
        force_refresh: bool = False,
    ) -> FetchResult:
        """Ensure one export is cached (downloading only if it is missing,
        expired, or ``force_refresh``); return where it landed."""

        spec = get_spec(dataset)
        if spec.bulk is None:
            raise RoutingError(f"{dataset} has no bulk export")
        if spec.templated and symbol is None:
            raise RoutingError(
                f"{dataset} is published per {spec.bulk.template.value}; pass symbol="
            )
        return await self._bulk.ensure(
            spec.bulk.namespace,
            dataset=spec.bulk.dataset,
            schema=spec.bulk.schema,
            symbol=symbol,
            force_refresh=force_refresh,
            max_age=self._config.bulk_ttl,
        )

    # ---- cells and cache state ------------------------------------------------

    def _cells(
        self,
        spec: DatasetSpec,
        start: _dt.date | None,
        end: _dt.date | None,
        symbols: list[str],
        today: _dt.date,
    ) -> list[str | None]:
        """The template values (files) a request touches."""

        if spec.bulk is None or spec.bulk.template is Template.NONE:
            return [None]
        if spec.bulk.template is Template.SYMBOL:
            if not symbols:
                raise RoutingError(
                    f"{spec.name} is published one file per symbol; pass symbols=[...]"
                )
            return list(dict.fromkeys(symbols))
        # Template.YEAR — one file per calendar year in the range; without a
        # start, just the current year (the full history is many GB).
        first = (start or end or today).year
        last = (end or today).year
        if start is None and end is not None:
            first = last
        return [str(year) for year in range(first, last + 1)]

    def _metas(self, spec: DatasetSpec, cells: list[str | None]) -> list[CacheEntryMeta | None]:
        assert spec.bulk is not None
        b = spec.bulk
        return [
            self._bulk.cached(b.namespace, dataset=b.dataset, schema=b.schema, symbol=c)
            for c in cells
        ]

    def _bulk_state(
        self, spec: DatasetSpec, cells: list[str | None]
    ) -> tuple[BulkState, pd.Timestamp | None]:
        """What the cache holds for these cells. No network access."""

        if spec.bulk is None or not cells:
            return BulkState(), None
        metas = self._metas(spec, cells)
        present = [m for m in metas if m is not None]
        if len(present) != len(metas):
            return BulkState(cached=False), None
        expired = any(m.age() > self._config.bulk_ttl for m in present)
        watermark: pd.Timestamp | None = None
        if spec.date_column:
            # A yearly series is only ever extended in its latest file; per-
            # symbol files each have their own watermark and the live diff
            # must start at the oldest of them.
            relevant = present[-1:] if spec.bulk.template is Template.YEAR else present
            marks = [
                self._bulk.watermark(m.bucket, m.object_name, spec.date_column) for m in relevant
            ]
            known = [m for m in marks if m is not None]
            watermark = min(known) if known else None
        return (
            BulkState(
                cached=True,
                watermark=None if watermark is None else watermark.date(),
                expired=expired,
            ),
            watermark,
        )

    async def _ensure_cells(
        self,
        spec: DatasetSpec,
        cells: list[str | None],
        *,
        hard: bool,
        revalidate: bool,
    ) -> dict[str | None, FetchResult]:
        """Download / refresh every cell the plan needs.

        ``hard`` re-downloads unconditionally (``force_refresh``);
        ``revalidate`` asks rw-api whether each cached copy is current and
        downloads only those that changed; otherwise only missing cells are
        fetched.
        """

        assert spec.bulk is not None
        b = spec.bulk
        semaphore = asyncio.Semaphore(self._config.download_concurrency)

        async def _one(cell: str | None) -> tuple[str | None, FetchResult]:
            async with semaphore:
                result = await self._bulk.ensure(
                    b.namespace,
                    dataset=b.dataset,
                    schema=b.schema,
                    symbol=cell,
                    force_refresh=hard,
                    max_age=0.0 if revalidate else None,
                )
                return cell, result

        pairs = await asyncio.gather(*(_one(c) for c in cells))
        return dict(pairs)

    # ---- the two halves -----------------------------------------------------------

    async def _read_bulk(
        self,
        spec: DatasetSpec,
        cells: list[str | None],
        route: RoutePlan,
        symbols: list[str],
        fetched: dict[str | None, FetchResult],
    ) -> tuple[pd.DataFrame, tuple[FetchResult, ...]]:
        assert spec.bulk is not None
        assert route.bulk_range is not None
        b = spec.bulk
        per_symbol_files = b.template is Template.SYMBOL
        window: DateRange = route.bulk_range

        async def _one(cell: str | None) -> tuple[pd.DataFrame, FetchResult]:
            result = fetched.get(cell) or await self._bulk.ensure(
                b.namespace, dataset=b.dataset, schema=b.schema, symbol=cell
            )
            try:
                frame = await self._read_object(spec, result, window, symbols, per_symbol_files)
            except (OSError, ValueError, pa.ArrowException) as exc:
                # A corrupt or truncated cached object is not worth keeping:
                # evict it and download once more rather than failing from
                # disk forever.
                log.warning("cached %s unreadable (%s); re-downloading", result.object_name, exc)
                with contextlib.suppress(CacheError):
                    self._bulk.cache.evict(result.bucket, result.object_name)
                result = await self._bulk.ensure(
                    b.namespace, dataset=b.dataset, schema=b.schema, symbol=cell, force_refresh=True
                )
                frame = await self._read_object(spec, result, window, symbols, per_symbol_files)
            if per_symbol_files and spec.symbol_column and spec.symbol_column not in frame.columns:
                frame[spec.symbol_column] = cell
            return frame, result

        parts = await asyncio.gather(*(_one(c) for c in cells))
        frames = [f for f, _ in parts]
        objects = tuple(r for _, r in parts)
        if len(frames) == 1:
            return frames[0], objects
        non_empty = [f for f in frames if not f.empty] or frames[:1]
        return pd.concat(non_empty, ignore_index=True), objects

    async def _read_object(
        self,
        spec: DatasetSpec,
        result: FetchResult,
        window: DateRange,
        symbols: list[str],
        per_symbol_files: bool,
    ) -> pd.DataFrame:
        return await asyncio.to_thread(
            formats.read_frame,
            result.path,
            object_name=result.object_name,
            date_column=spec.date_column,
            start=window.start,
            end=window.end,
            symbol_column=spec.symbol_column,
            symbols=None if per_symbol_files else (symbols or None),
        )

    async def _read_live(
        self,
        spec: DatasetSpec,
        route: RoutePlan,
        symbols: list[str],
        today: _dt.date,
    ) -> pd.DataFrame:
        assert spec.live is not None
        assert route.live_range is not None
        live = spec.live
        window = route.live_range
        rows = await self._live.fetch_rows(
            live, gte=window.start, lte=window.end, symbols=symbols or None, today=today
        )
        frame = pd.DataFrame.from_records(rows) if rows else pd.DataFrame()
        if frame.empty:
            return frame
        if live.rename:
            frame = frame.rename(columns=dict(live.rename))
        if spec.derive is not None:
            frame = spec.derive(frame)
        # Routes without a server-side symbol filter (and trailing-window
        # routes that ignore gte/lte) are narrowed here.
        return formats.filter_frame(
            frame,
            date_column=spec.date_column,
            start=window.start,
            end=window.end,
            symbol_column=spec.symbol_column,
            symbols=symbols or None,
        )

    def _template(
        self, spec: DatasetSpec, cells: list[str | None], bulk_df: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """Sample rows in the export's schema, to conform live rows against."""

        if bulk_df is not None and not bulk_df.empty:
            return bulk_df.head(50)
        if spec.bulk is None:
            return None
        for meta in self._metas(spec, cells):
            if meta is None:
                continue
            path = self._bulk.cache.path_for(meta.bucket, meta.object_name)
            try:
                head = formats.read_head(path, object_name=meta.object_name)
            except (OSError, ValueError, pa.ArrowException):
                continue
            if spec.templated and spec.symbol_column and spec.symbol_column not in head.columns:
                head[spec.symbol_column] = meta.cell[3] if meta.cell else None
            return head
        return None


# ---------------------------------------------------------------------------
# conforming and merging (pure)
# ---------------------------------------------------------------------------


def _conform(live: pd.DataFrame, template: pd.DataFrame | None, spec: DatasetSpec) -> pd.DataFrame:
    """Reshape renamed live rows to the export's columns and dtypes.

    With no export to copy from (nothing cached), columns keep the live
    order and the date column is parsed to ``datetime64``.
    """

    if live.empty:
        return live if template is None else template.iloc[0:0].copy()
    if template is None:
        out = live.copy()
        if spec.date_column and spec.date_column in out.columns:
            out[spec.date_column] = pd.to_datetime(out[spec.date_column], format="ISO8601")
        return out
    out = live.reindex(columns=template.columns)
    for column in template.columns:
        if column in live.columns:
            out[column] = _coerce_like(out[column], template[column])
    return out


def _coerce_like(values: pd.Series, like: pd.Series) -> pd.Series:
    """Convert a live column to the representation the export uses."""

    dtype = like.dtype
    try:
        if isinstance(dtype, pd.DatetimeTZDtype):
            return pd.to_datetime(values, format="ISO8601", utc=True).dt.tz_convert(dtype.tz)
        if pd.api.types.is_datetime64_any_dtype(dtype):
            parsed = pd.to_datetime(values, format="ISO8601", utc=True).dt.tz_localize(None)
            return parsed.astype(dtype)
        if pd.api.types.is_object_dtype(dtype):
            sample = next((v for v in like if v is not None and not _is_na(v)), None)
            if isinstance(sample, _dt.datetime):
                return pd.to_datetime(values, format="ISO8601")
            if isinstance(sample, _dt.date):
                parsed = pd.to_datetime(values, format="ISO8601")
                days = [None if pd.isna(ts) else ts.date() for ts in parsed]
                return pd.Series(days, index=values.index, dtype=object, name=values.name)
            return values
        if (
            pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
        ) and not values.isna().any():
            return values.astype(dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        log.debug("could not coerce live column %s to %s: %s", values.name, dtype, exc)
    return values


def _is_na(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _merge(
    bulk: pd.DataFrame | None, live: pd.DataFrame | None, spec: DatasetSpec
) -> tuple[pd.DataFrame, int, int]:
    """Join the live half onto the bulk half.

    Bulk rows keep their order; new live rows follow, ordered by date then
    symbol. On a key collision at the seam the row stays where it is but
    takes the live values for every column live supplies (see
    :func:`_refresh_seam`). Returns ``(frame, live_rows_added, seam_rows_updated)``.
    """

    if live is None or live.empty:
        return (bulk if bulk is not None else pd.DataFrame()), 0, 0

    order = [c for c in (spec.date_column, spec.symbol_column) if c and c in live.columns]
    if bulk is None or bulk.empty:
        live_keys = [k for k in spec.keys if k in live.columns]
        if live_keys:
            live = live[~_key_index(live, live_keys).duplicated(keep="last")]
        out = live.sort_values(order, kind="stable") if order else live
        return out.reset_index(drop=True), len(out), 0

    date_col = spec.date_column if spec.date_column in bulk.columns else None
    seam = bulk
    if date_col and date_col in live.columns:
        # Only bulk rows on or after the first live *day* can collide.
        live_start = formats.to_timestamps(live[date_col]).min()
        if not pd.isna(live_start):
            seam = bulk[formats.to_timestamps(bulk[date_col]) >= live_start.normalize()]

    sym = spec.symbol_column
    if spec.restrict_to_bulk_symbols and sym and sym in bulk.columns and sym in live.columns:
        live = live[live[sym].isin(pd.unique(bulk[sym]))]

    keys = [k for k in spec.keys if k in bulk.columns and k in live.columns]
    updated = 0
    if keys:
        # Overlapping requests (per-symbol fan-out, halved top-spreads
        # ranges) can return a row twice.
        live = live[~_key_index(live, keys).duplicated(keep="last")]
    if keys and not seam.empty and not live.empty:
        incoming_keys = _key_index(live, keys)
        seam_keys = _key_index(seam, keys)
        overlap = incoming_keys.isin(seam_keys)
        updated = _refresh_seam(bulk, seam, seam_keys, live[overlap], keys)
        live = live[~overlap]

    if live.empty:
        return bulk, 0, updated
    if order:
        live = live.sort_values(order, kind="stable")
    return pd.concat([bulk, live], ignore_index=True), len(live), updated


def _refresh_seam(
    bulk: pd.DataFrame,
    seam: pd.DataFrame,
    seam_keys: pd.MultiIndex,
    overlap: pd.DataFrame,
    keys: list[str],
) -> int:
    """Overwrite seam rows in ``bulk`` (in place) with the live values.

    An export's last days can be provisional — rw-api's StatArb price export
    has carried half-day volumes for its newest date — while the live route
    holds the settled values. Live wins for every column it supplies;
    export-only columns keep their values. Returns rows updated.
    """

    if overlap.empty:
        return 0
    columns = [
        c
        for c in overlap.columns
        if c not in keys and c in bulk.columns and overlap[c].notna().any()
    ]
    if not columns:
        return 0
    incoming = overlap.set_axis(_key_index(overlap, keys))
    hit = seam_keys.isin(incoming.index)
    rows = seam.index[hit]
    values = incoming.reindex(seam_keys[hit])
    for column in columns:
        fresh = values[column].to_numpy()
        keep = pd.isna(fresh)
        if keep.all():
            continue
        current = bulk.loc[rows, column].to_numpy()
        merged = pd.Series(fresh, index=rows).where(~keep, pd.Series(current, index=rows))
        try:
            bulk.loc[rows, column] = merged.astype(bulk[column].dtype)
        except (TypeError, ValueError):
            bulk[column] = bulk[column].astype(object)
            bulk.loc[rows, column] = merged
    return len(rows)


def _key_index(frame: pd.DataFrame, keys: list[str]) -> pd.MultiIndex:
    """Row identities for seam de-duplication.

    Timestamps are compared to the nearest second: exports can carry the
    exchange's raw millisecond jitter (Binance funding times such as
    ``08:00:00.006``) where the live route reports the whole hour, and an
    exact comparison would let the same event through twice.
    """

    columns: dict[str, pd.Series] = {}
    for key in keys:
        values = frame[key]
        if pd.api.types.is_datetime64_any_dtype(values.dtype):
            values = values.dt.round("s")
        columns[key] = values.astype(object)
    return pd.MultiIndex.from_frame(pd.DataFrame(columns, index=frame.index))


def _symbol_list(symbols: Sequence[str] | str | None) -> list[str]:
    if symbols is None:
        return []
    if isinstance(symbols, str):
        return [s.strip() for s in symbols.split(",") if s.strip()]
    return [str(s) for s in symbols]


async def _capture(awaitable: Any) -> Any:
    """Await, returning an exception instead of raising it."""

    try:
        return await awaitable
    except (RwApiError, OSError) as exc:
        return exc


def _degradable(exc: BaseException) -> bool:
    """Live failures that leave the cached export a useful answer on its own."""

    if isinstance(exc, (ServerError, RateLimitError)):
        return True
    return isinstance(exc, RwApiError) and exc.status_code is None


__all__ = ["AsyncSession"]
