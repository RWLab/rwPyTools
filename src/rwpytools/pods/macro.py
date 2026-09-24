"""Macro namespace data loaders.

Maps to ``/v1/macro/file`` (bulk exports) and the ``/v1/macro/*`` live
routes. Mirrors the ``macro_get_*`` rwRTools surface; methods whose dataset
is not published by rw-api raise :class:`rwpytools.errors.DatasetNotSeededError`.
"""

from __future__ import annotations

import pandas as pd

from ..errors import DatasetNotSeededError
from ..http import run_sync
from ..routing import Source
from ._base import AsyncPodAccessor, DateLike, SourceLike, SymbolsLike, SyncPodAccessor


class AsyncMacroPod(AsyncPodAccessor):
    """Async accessor for the ``macro`` namespace."""

    namespace = "macro"

    # ---- published ---------------------------------------------------------

    async def get_earnings(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Finnhub earnings calendar (bulk only)."""

        return await self._get(
            "earnings_calendar",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_vix_vix3m(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """VIX + VIX3M daily OHLC; live companion ``/v1/macro/vix``."""

        return await self._get(
            "vix",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_expiring_futures(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Norgate continuous futures, daily OHLC with contract metadata (bulk only)."""

        return await self._get(
            "futures",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_expiring_rp_futures(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Risk-premia futures, daily OHLC; live companion ``/v1/macro/futures``."""

        return await self._get(
            "rp_futures",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_expiring_vx_futures(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """VX expiring futures, daily OHLC (bulk only)."""

        return await self._get(
            "vix_futures",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_historical_asset_class(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Main asset-class daily OHLC; live companion ``/v1/macro/asset-classes``."""

        return await self._get(
            "asset_classes",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_rates(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """T-bill / short-rate series; live companion ``/v1/macro/rates``."""

        return await self._get(
            "tbill_rates",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_historical_short_sale(
        self,
        year: str | int,
        *,
        source: SourceLike = Source.AUTO,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """IB short-sale availability for one calendar year.

        rwRTools' ``macro_get_historical_short_sale(year)`` returns one file
        per year; on rw-api that is ``ib.shortstock`` / ``yearly`` with the
        year as the symbol. The current year is topped up from
        ``/v1/equities/ib-short-stocks``.
        """

        return await self._get(
            "ib_shortstock",
            source=source,
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            symbols=symbols,
            force_refresh=force_refresh,
        )

    # ---- not published by rw-api -----------------------------------------------

    async def get_close_price_momo(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="macro",
            dataset="close_price_momo",
            schema="snapshot",
            method="macro_get_close_price_momo",
        )

    async def get_govt_bond_returns(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="macro",
            dataset="bond_returns",
            schema="snapshot",
            method="macro_get_govt_bond_returns",
        )

    async def get_nyse_holidays(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="macro",
            dataset="nyse_holidays",
            schema="snapshot",
            method="macro_get_nyse_holidays",
        )

    async def get_straddles_over_earnings(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="macro",
            dataset="earnings",
            schema="straddles",
            method="macro_get_straddles_over_earnings",
        )


class MacroPod(SyncPodAccessor[AsyncMacroPod]):
    """Synchronous accessor for the ``macro`` namespace."""

    namespace = "macro"

    def get_earnings(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_earnings(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_vix_vix3m(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_vix_vix3m(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_expiring_futures(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_expiring_futures(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_expiring_rp_futures(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_expiring_rp_futures(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_expiring_vx_futures(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_expiring_vx_futures(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_historical_asset_class(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_historical_asset_class(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_rates(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_rates(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_historical_short_sale(
        self,
        year: str | int,
        *,
        source: SourceLike = Source.AUTO,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_historical_short_sale(
                year, source=source, symbols=symbols, force_refresh=force_refresh
            )
        )

    def get_close_price_momo(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_close_price_momo(force_refresh=force_refresh))

    def get_govt_bond_returns(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_govt_bond_returns(force_refresh=force_refresh))

    def get_nyse_holidays(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_nyse_holidays(force_refresh=force_refresh))

    def get_straddles_over_earnings(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_straddles_over_earnings(force_refresh=force_refresh))


__all__ = ["AsyncMacroPod", "MacroPod"]
