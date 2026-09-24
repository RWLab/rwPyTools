"""Wrappers around rw-api's live JSON endpoints. Never cached.

These routes are backed by the PostgreSQL warehouse and always hold the
newest rows. Most are the *live companion* of a bulk export (see
:mod:`rwpytools.datasets`); :meth:`rwpytools.Client.get` uses them to top a
cached export up to real time. A few (Y.O.L.O, RP-Schteroids, intraday
StatArb spreads) exist only here.

Coverage is 1:1 with the endpoints published at
https://api.robotwealth.com/v1/docs:

* status
* yolo factors / weights / volatilities / historical
* rpschteroids weights
* statarb daily / hourly top spreads, liquid-universe daily / hourly top spreads
* statarb universe prices
* equities earnings recent surprises, acquisitions, ib short stocks, tickers
* forex daily
* crypto binance spot / perps / perps funding / coin-m perps / coin-m funding
* macro vix / asset classes / futures / rates

Pagination
----------

Cursor-paginated routes return at most 1,000 rows per page. Every method
here follows ``pagination.next_cursor`` until ``has_more`` is false and
returns all rows; pass ``cursor=`` only to resume from a token you already
hold. Top-spreads routes are not cursor-paginated — they take a ``limit``
(server default 10,000, max 50,000).

Date parameters
---------------

``gte`` / ``lte`` accept ISO-8601 strings (``YYYY-MM-DD``), ``datetime.date``
or ``datetime.datetime``. When both are omitted most routes default to the
last 7 days server-side.

Two classes are exposed:

* :class:`AsyncLiveClient` — async-native, used by :class:`rwpytools.AsyncClient`.
* :class:`LiveClient` — synchronous wrapper used by :class:`rwpytools.Client`.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from .datasets import LiveSource, LiveStyle
from .errors import DataFormatError
from .http import _Http, run_sync
from .routing import as_date

log = logging.getLogger(__name__)

Record = dict[str, Any]
DateLike = str | _dt.date | None

#: Server ceiling on the top-spreads ``limit`` parameter.
TOPSPREADS_MAX_LIMIT = 50_000
#: Server ceilings on the trailing-window parameters.
LOOKBACK_MAX_DAYS = 90
HISTORICAL_MAX_DAYS = 365
#: Max symbols per request on routes taking a comma-separated list.
MAX_TICKERS_PER_REQUEST = 100
#: Up to this many symbols, a symbol-filtered range read issues one filtered
#: request per symbol (a page or two each) instead of paging every symbol.
PER_SYMBOL_FANOUT_MAX = 10


class AsyncLiveClient:
    """Async wrappers around rw-api live JSON endpoints. Returns DataFrames."""

    def __init__(self, http: _Http, *, concurrency: int = 4) -> None:
        self._http = http
        self._concurrency = max(1, concurrency)

    # ---- generic ------------------------------------------------------------

    async def fetch_rows(
        self,
        live: LiveSource,
        *,
        gte: _dt.date | None = None,
        lte: _dt.date | None = None,
        symbols: Sequence[str] | None = None,
        today: _dt.date | None = None,
    ) -> list[Record]:
        """Every row a live route holds for ``[gte, lte]`` and ``symbols``.

        Dispatches on :class:`rwpytools.datasets.LiveStyle`. The symbol
        filter is applied server-side where the route supports it; callers
        must still filter client-side for routes without one. Rows are
        returned verbatim, in the route's own column names.
        """

        today = today or _dt.date.today()
        style = live.style
        syms = list(symbols or [])

        if style is LiveStyle.LATEST:
            return _rows(await self._http.get_json(live.path))

        if style is LiveStyle.DAYS:
            params: dict[str, Any] = {}
            if gte is not None:
                params["days"] = _clamp((today - gte).days + 1, 1, HISTORICAL_MAX_DAYS)
            return _rows(await self._http.get_json(live.path, params=params))

        if style is LiveStyle.TOPSPREADS:
            if gte is not None and lte is None:
                lte = today
            return await self._topspreads(live.path, gte=gte, lte=lte)

        if style is LiveStyle.LOOKBACK:
            params = {}
            if gte is not None:
                params["lookback_days"] = _clamp((today - gte).days + 1, 1, LOOKBACK_MAX_DAYS)
            return await self._chunked_by_tickers(live, params, syms, paginate=False)

        if style is LiveStyle.SNAPSHOT:
            return await self._chunked_by_tickers(live, {}, syms, paginate=True)

        # LiveStyle.RANGE
        base = _params(gte=_iso(gte), lte=_iso(lte))
        if live.symbol_param and syms and len(syms) <= PER_SYMBOL_FANOUT_MAX:
            per_symbol = await self._gather(
                [self._paginate(live.path, {**base, live.symbol_param: s}) for s in syms]
            )
            return [row for rows in per_symbol for row in rows]
        return await self._paginate(live.path, base)

    async def fetch_frame(
        self,
        live: LiveSource,
        *,
        gte: _dt.date | None = None,
        lte: _dt.date | None = None,
        symbols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """:meth:`fetch_rows` as a DataFrame."""

        return _frame(await self.fetch_rows(live, gte=gte, lte=lte, symbols=symbols))

    # ---- status -------------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        """``GET /v1/status`` — ``{"success": true, "time": ...}`` when up."""

        return await self._http.get_json("/v1/status")

    # ---- yolo ---------------------------------------------------------------

    async def yolo_factors(self) -> pd.DataFrame:
        return _frame(_rows(await self._http.get_json("/v1/yolo/factors")))

    async def yolo_weights(self) -> pd.DataFrame:
        return _frame(_rows(await self._http.get_json("/v1/yolo/weights")))

    async def yolo_volatilities(self) -> pd.DataFrame:
        return _frame(_rows(await self._http.get_json("/v1/yolo/volatilities")))

    async def yolo_historical(self, *, days: int = 90) -> pd.DataFrame:
        """Daily megafactor weights, combined weight and EWMA vol (``days`` 1-365)."""

        body = await self._http.get_json("/v1/yolo/historical", params={"days": days})
        return _frame(_rows(body))

    # ---- rpschteroids -------------------------------------------------------

    async def rpschteroids_weights(self) -> pd.DataFrame:
        return _frame(_rows(await self._http.get_json("/v1/rpschteroids/weights")))

    # ---- equities / statarb -------------------------------------------------

    async def statarb_daily_top_spreads(
        self, *, gte: DateLike = None, lte: DateLike = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Daily StatArb pair factors from the top-spreads universe.

        With neither ``gte`` nor ``lte`` the latest date is returned.
        ``limit`` caps the rows (server default 10,000, max 50,000).
        """

        return await self._top_spreads_page(
            "/v1/equities/statarb/daily/topspreads", gte=gte, lte=lte, limit=limit
        )

    async def statarb_hourly_top_spreads(
        self, *, gte: DateLike = None, lte: DateLike = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Intraday (hourly) StatArb pair factors. Same parameters as the daily route."""

        return await self._top_spreads_page(
            "/v1/equities/statarb/hourly/topspreads", gte=gte, lte=lte, limit=limit
        )

    async def statarb_liquid_daily_top_spreads(
        self, *, gte: DateLike = None, lte: DateLike = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Daily StatArb pair factors from the liquid-universe top spreads
        (500k avg volume, $10 min price, $5B market cap)."""

        return await self._top_spreads_page(
            "/v1/equities/statarb/liquid/daily/topspreads", gte=gte, lte=lte, limit=limit
        )

    async def statarb_liquid_hourly_top_spreads(
        self, *, gte: DateLike = None, lte: DateLike = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Intraday (hourly) liquid-universe StatArb pair factors."""

        return await self._top_spreads_page(
            "/v1/equities/statarb/liquid/hourly/topspreads", gte=gte, lte=lte, limit=limit
        )

    async def statarb_prices(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Daily OHLCV for the StatArb pair universe (history floored at 2019).
        ``ticker`` must fall inside the pair universe."""

        return await self._range("/v1/equities/statarb/prices", ticker, gte, lte, cursor)

    # ---- equities / other ---------------------------------------------------

    async def earnings_recent_surprises(
        self,
        *,
        lookback_days: int | None = None,
        tickers: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Reported earnings events inside a trailing window.

        ``lookback_days`` is 1-90 (server default 20). ``tickers`` filters
        to up to 100 symbols — a list or a comma-separated string.
        """

        params = _params(lookback_days=lookback_days, tickers=_join(tickers))
        body = await self._http.get_json("/v1/equities/earnings/recent-surprises", params=params)
        return _frame(_rows(body))

    async def ib_short_stocks(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        country: str | None = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """IB short-stock availability. ``ticker`` / ``country`` are exact-match filters."""

        params = _params(
            ticker=ticker, gte=_iso(gte), lte=_iso(lte), country=country, cursor=cursor
        )
        return _frame(await self._paginate("/v1/equities/ib-short-stocks", params))

    async def acquisitions(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Equity acquisitions / delistings (corporate actions)."""

        return await self._range("/v1/equities/acquisitions", ticker, gte, lte, cursor)

    async def equity_tickers(
        self,
        *,
        tickers: str | Sequence[str] | None = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Per-ticker equity metadata snapshot; freshest row per ticker.

        ``tickers`` is an optional filter — a symbol, a comma-separated
        string, or a list (max 100). Omit it for the whole universe.
        """

        params = _params(tickers=_join(tickers), cursor=cursor)
        return _frame(await self._paginate("/v1/equities/tickers", params))

    # ---- forex --------------------------------------------------------------

    async def forex_daily(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Daily FX OHLCV bars. ``ticker`` e.g. ``AUDUSD``."""

        return await self._range("/v1/fx/daily", ticker, gte, lte, cursor)

    # ---- crypto / binance ---------------------------------------------------

    async def binance_spot(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Hourly Binance spot OHLCV bars. ``ticker`` e.g. ``BTCUSDT``."""

        return await self._range("/v1/crypto/binance/spot", ticker, gte, lte, cursor)

    async def binance_perps(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Hourly Binance USDT-margined perpetual OHLCV bars."""

        return await self._range("/v1/crypto/binance/perps", ticker, gte, lte, cursor)

    async def binance_perps_funding(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Binance USDT-margined perpetual funding rates and mark prices."""

        return await self._range("/v1/crypto/binance/perps/funding", ticker, gte, lte, cursor)

    async def binance_coin_m_perps(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Hourly Binance coin-margined perpetual OHLCV bars."""

        return await self._range("/v1/crypto/binance/coin-m/perps", ticker, gte, lte, cursor)

    async def binance_coin_m_funding(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Binance coin-margined perpetual funding rates and mark prices."""

        return await self._range("/v1/crypto/binance/coin-m/funding", ticker, gte, lte, cursor)

    # ---- macro --------------------------------------------------------------

    async def macro_vix(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Daily VIX-family bars. ``ticker`` e.g. ``VIX`` / ``VIX3M``."""

        return await self._range("/v1/macro/vix", ticker, gte, lte, cursor)

    async def macro_asset_classes(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Daily main asset-class OHLCV bars."""

        return await self._range("/v1/macro/asset-classes", ticker, gte, lte, cursor)

    async def macro_futures(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Daily futures OHLCV bars by contract (companion to the rp-futures export)."""

        return await self._range("/v1/macro/futures", ticker, gte, lte, cursor)

    async def macro_rates(
        self,
        *,
        symbol: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        """Daily short-term rate series, e.g. the 13-week T-bill (``%IRX``)."""

        params = _params(symbol=symbol, gte=_iso(gte), lte=_iso(lte), cursor=cursor)
        return _frame(await self._paginate("/v1/macro/rates", params))

    # ---- internals ----------------------------------------------------------

    async def _range(
        self,
        path: str,
        ticker: str | None,
        gte: DateLike,
        lte: DateLike,
        cursor: str | None,
    ) -> pd.DataFrame:
        params = _params(ticker=ticker, gte=_iso(gte), lte=_iso(lte), cursor=cursor)
        return _frame(await self._paginate(path, params))

    async def _top_spreads_page(
        self, path: str, *, gte: DateLike, lte: DateLike, limit: int | None
    ) -> pd.DataFrame:
        params = _params(gte=_iso(gte), lte=_iso(lte), limit=limit)
        return _frame(_rows(await self._http.get_json(path, params=params)))

    async def _paginate(self, path: str, params: dict[str, Any]) -> list[Record]:
        """Follow ``pagination.next_cursor`` until the route says it is done."""

        rows: list[Record] = []
        page_params = dict(params)
        seen: set[str] = set()
        pages = 0
        while True:
            body = await self._http.get_json(path, params=page_params)
            rows.extend(_rows(body))
            pages += 1
            cursor = _next_cursor(body)
            if cursor is None or cursor in seen:
                break
            seen.add(cursor)
            page_params = {**params, "cursor": cursor}
        if pages > 1:
            log.debug("%s: %d rows over %d pages", path, len(rows), pages)
        return rows

    async def _topspreads(
        self, path: str, *, gte: _dt.date | None, lte: _dt.date | None
    ) -> list[Record]:
        """Top spreads for ``[gte, lte]``, halving the range whenever a
        response fills the ``limit`` — the route cannot paginate."""

        params = _params(gte=_iso(gte), lte=_iso(lte), limit=TOPSPREADS_MAX_LIMIT)
        rows = _rows(await self._http.get_json(path, params=params))
        if len(rows) < TOPSPREADS_MAX_LIMIT or gte is None or lte is None or gte >= lte:
            if len(rows) >= TOPSPREADS_MAX_LIMIT:
                log.warning("%s: hit the %d-row limit; result may be truncated", path, len(rows))
            return rows
        mid = gte + (lte - gte) // 2
        left, right = await self._gather(
            [
                self._topspreads(path, gte=gte, lte=mid),
                self._topspreads(path, gte=mid + _dt.timedelta(days=1), lte=lte),
            ]
        )
        return left + right

    async def _chunked_by_tickers(
        self,
        live: LiveSource,
        params: dict[str, Any],
        symbols: list[str],
        *,
        paginate: bool,
    ) -> list[Record]:
        chunks: list[list[str] | None] = (
            [
                symbols[i : i + MAX_TICKERS_PER_REQUEST]
                for i in range(0, len(symbols), MAX_TICKERS_PER_REQUEST)
            ]
            if symbols and live.symbol_param
            else [None]
        )

        async def _one(chunk: list[str] | None) -> list[Record]:
            p = dict(params)
            if chunk is not None and live.symbol_param:
                p[live.symbol_param] = ",".join(chunk)
            if paginate:
                return await self._paginate(live.path, p)
            return _rows(await self._http.get_json(live.path, params=p))

        results = await self._gather([_one(c) for c in chunks])
        return [row for rows in results for row in rows]

    async def _gather(self, coros: Sequence[Any]) -> list[list[Record]]:
        """``asyncio.gather`` bounded by the client's concurrency."""

        semaphore = asyncio.Semaphore(self._concurrency)

        async def _bounded(coro: Any) -> list[Record]:
            async with semaphore:
                result: list[Record] = await coro
                return result

        return list(await asyncio.gather(*(_bounded(c) for c in coros)))


class LiveClient:
    """Synchronous wrappers over :class:`AsyncLiveClient`.

    Method signatures mirror the async ones exactly; calls run on a
    one-shot event loop via :func:`rwpytools.http.run_sync`.
    """

    def __init__(self, http: _Http, *, concurrency: int = 4) -> None:
        self._async = AsyncLiveClient(http, concurrency=concurrency)

    def fetch_frame(
        self,
        live: LiveSource,
        *,
        gte: _dt.date | None = None,
        lte: _dt.date | None = None,
        symbols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return run_sync(self._async.fetch_frame(live, gte=gte, lte=lte, symbols=symbols))

    def status(self) -> dict[str, Any]:
        return run_sync(self._async.status())

    def yolo_factors(self) -> pd.DataFrame:
        return run_sync(self._async.yolo_factors())

    def yolo_weights(self) -> pd.DataFrame:
        return run_sync(self._async.yolo_weights())

    def yolo_volatilities(self) -> pd.DataFrame:
        return run_sync(self._async.yolo_volatilities())

    def yolo_historical(self, *, days: int = 90) -> pd.DataFrame:
        return run_sync(self._async.yolo_historical(days=days))

    def rpschteroids_weights(self) -> pd.DataFrame:
        return run_sync(self._async.rpschteroids_weights())

    def statarb_daily_top_spreads(
        self, *, gte: DateLike = None, lte: DateLike = None, limit: int | None = None
    ) -> pd.DataFrame:
        return run_sync(self._async.statarb_daily_top_spreads(gte=gte, lte=lte, limit=limit))

    def statarb_hourly_top_spreads(
        self, *, gte: DateLike = None, lte: DateLike = None, limit: int | None = None
    ) -> pd.DataFrame:
        return run_sync(self._async.statarb_hourly_top_spreads(gte=gte, lte=lte, limit=limit))

    def statarb_liquid_daily_top_spreads(
        self, *, gte: DateLike = None, lte: DateLike = None, limit: int | None = None
    ) -> pd.DataFrame:
        return run_sync(self._async.statarb_liquid_daily_top_spreads(gte=gte, lte=lte, limit=limit))

    def statarb_liquid_hourly_top_spreads(
        self, *, gte: DateLike = None, lte: DateLike = None, limit: int | None = None
    ) -> pd.DataFrame:
        return run_sync(
            self._async.statarb_liquid_hourly_top_spreads(gte=gte, lte=lte, limit=limit)
        )

    def statarb_prices(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(self._async.statarb_prices(ticker=ticker, gte=gte, lte=lte, cursor=cursor))

    def earnings_recent_surprises(
        self,
        *,
        lookback_days: int | None = None,
        tickers: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.earnings_recent_surprises(lookback_days=lookback_days, tickers=tickers)
        )

    def ib_short_stocks(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        country: str | None = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.ib_short_stocks(
                ticker=ticker, gte=gte, lte=lte, country=country, cursor=cursor
            )
        )

    def acquisitions(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(self._async.acquisitions(ticker=ticker, gte=gte, lte=lte, cursor=cursor))

    def equity_tickers(
        self,
        *,
        tickers: str | Sequence[str] | None = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(self._async.equity_tickers(tickers=tickers, cursor=cursor))

    def forex_daily(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(self._async.forex_daily(ticker=ticker, gte=gte, lte=lte, cursor=cursor))

    def binance_spot(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(self._async.binance_spot(ticker=ticker, gte=gte, lte=lte, cursor=cursor))

    def binance_perps(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(self._async.binance_perps(ticker=ticker, gte=gte, lte=lte, cursor=cursor))

    def binance_perps_funding(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.binance_perps_funding(ticker=ticker, gte=gte, lte=lte, cursor=cursor)
        )

    def binance_coin_m_perps(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.binance_coin_m_perps(ticker=ticker, gte=gte, lte=lte, cursor=cursor)
        )

    def binance_coin_m_funding(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.binance_coin_m_funding(ticker=ticker, gte=gte, lte=lte, cursor=cursor)
        )

    def macro_vix(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(self._async.macro_vix(ticker=ticker, gte=gte, lte=lte, cursor=cursor))

    def macro_asset_classes(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.macro_asset_classes(ticker=ticker, gte=gte, lte=lte, cursor=cursor)
        )

    def macro_futures(
        self,
        *,
        ticker: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(self._async.macro_futures(ticker=ticker, gte=gte, lte=lte, cursor=cursor))

    def macro_rates(
        self,
        *,
        symbol: str | None = None,
        gte: DateLike = None,
        lte: DateLike = None,
        cursor: str | None = None,
    ) -> pd.DataFrame:
        return run_sync(self._async.macro_rates(symbol=symbol, gte=gte, lte=lte, cursor=cursor))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _params(**kwargs: Any) -> dict[str, Any]:
    """Drop ``None`` values from a kwargs dict so the URL stays clean."""

    return {k: v for k, v in kwargs.items() if v is not None}


def _iso(value: DateLike) -> str | None:
    """Render a date bound as ``YYYY-MM-DD``; validates strings early."""

    parsed = as_date(value)
    return None if parsed is None else parsed.isoformat()


def _join(tickers: str | Sequence[str] | None) -> str | None:
    if tickers is None:
        return None
    if isinstance(tickers, str):
        return tickers or None
    joined = ",".join(str(t) for t in tickers)
    return joined or None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _rows(body: Any) -> list[Record]:
    """Extract the row list from an rw-api envelope.

    Most routes put rows under ``data``; the top-spreads routes use
    ``rows``. A ``data`` object (rather than a list) is one row.
    """

    if not isinstance(body, dict):
        raise DataFormatError(f"live endpoint returned {type(body).__name__}, expected an object")
    payload = body.get("data")
    if payload is None:
        payload = body.get("rows")
    if payload is None:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    raise DataFormatError(f"unexpected `data` shape from rw-api: {type(payload).__name__}")


def _next_cursor(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    pagination = body.get("pagination")
    if not isinstance(pagination, dict) or pagination.get("has_more") is False:
        return None
    cursor = pagination.get("next_cursor")
    return str(cursor) if cursor else None


def _frame(rows: list[Record]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows) if rows else pd.DataFrame()


__all__ = ["AsyncLiveClient", "LiveClient"]
