"""Equity-factors namespace.

Maps to ``/v1/equities/file`` (bulk exports) and the ``/v1/equities/*``
live routes. Mirrors the ``equity_*`` rwRTools surface (which keeps statarb
inside the equity factors pod — there is no separate ``statarb`` pod).

``client.equity_factors`` is the public attribute name — it matches the
``equity_factors_research_pod`` bucket; the rw-api namespace itself is
``equities`` (plural).
"""

from __future__ import annotations

import pandas as pd

from ..http import run_sync
from ..routing import Source
from ._base import AsyncPodAccessor, DateLike, SourceLike, SymbolsLike, SyncPodAccessor


class AsyncEquityFactorsPod(AsyncPodAccessor):
    """Async accessor for the ``equities`` namespace."""

    namespace = "equities"

    async def get_tickers(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Equity ticker metadata snapshot. ``source='live'`` reads ``/v1/equities/tickers`` instead of the export."""

        return await self._get(
            "equity_tickers",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_liquid_universe(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Daily prices for the liquid universe of equities (bulk only)."""

        return await self._get(
            "liquid_universe_prices",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_liquid_universe_sectors(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Sector classification for the liquid universe (bulk only)."""

        return await self._get(
            "liquid_universe_sectors",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_streaks(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Extended streak factors (bulk only)."""

        return await self._get(
            "streaks",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_statarb_acquisitions(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Acquisitions / delistings; live companion ``/v1/equities/acquisitions``."""

        return await self._get(
            "statarb_acquisitions",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_statarb_prices(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """StatArb pair-universe daily OHLCV; live companion ``/v1/equities/statarb/prices``."""

        return await self._get(
            "statarb_prices",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_statarb_spreads(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """StatArb daily pair factors; live companion ``/v1/equities/statarb/daily/topspreads``."""

        return await self._get(
            "statarb_spreads",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_statarb_liquid_acquisitions(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Liquid-universe acquisitions / delistings; live companion ``/v1/equities/acquisitions``."""

        return await self._get(
            "statarb_liquid_acquisitions",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_statarb_liquid_prices(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Liquid-universe StatArb daily OHLCV (bulk only)."""

        return await self._get(
            "statarb_liquid_prices",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_statarb_liquid_spreads(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Liquid-universe StatArb daily pair factors; live companion ``/v1/equities/statarb/liquid/daily/topspreads``."""

        return await self._get(
            "statarb_liquid_spreads",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_earnings_surprises(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Equity earnings surprises; live companion ``/v1/equities/earnings/recent-surprises`` (90-day window)."""

        return await self._get(
            "earnings_surprises",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )


class EquityFactorsPod(SyncPodAccessor[AsyncEquityFactorsPod]):
    """Synchronous accessor for the ``equities`` namespace."""

    namespace = "equities"

    def get_tickers(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_tickers(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_liquid_universe(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_liquid_universe(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_liquid_universe_sectors(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_liquid_universe_sectors(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_streaks(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_streaks(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_statarb_acquisitions(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_statarb_acquisitions(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_statarb_prices(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_statarb_prices(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_statarb_spreads(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_statarb_spreads(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_statarb_liquid_acquisitions(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_statarb_liquid_acquisitions(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_statarb_liquid_prices(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_statarb_liquid_prices(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_statarb_liquid_spreads(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_statarb_liquid_spreads(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_earnings_surprises(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_earnings_surprises(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )


__all__ = ["AsyncEquityFactorsPod", "EquityFactorsPod"]
