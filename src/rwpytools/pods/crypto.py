"""Crypto namespace data loaders.

Maps to ``/v1/crypto/file`` (bulk exports) and the ``/v1/crypto/binance/*``
live routes. Every routed method returns the cached export topped up to
real time from its live companion (see :mod:`rwpytools.session`); pass
``source="bulk"`` or ``source="live"`` to force one side.

This file mirrors the ``crypto_get_*`` functions exported by R's
`rwRTools <https://github.com/RWLab/rwRtools>`_; methods whose dataset is
not published by rw-api raise :class:`rwpytools.errors.DatasetNotSeededError`.
The FTX series are legacy (the exchange is defunct).
"""

from __future__ import annotations

import pandas as pd

from ..errors import DatasetNotSeededError
from ..http import run_sync
from ..routing import Source
from ._base import AsyncPodAccessor, DateLike, SourceLike, SymbolsLike, SyncPodAccessor


class AsyncCryptoPod(AsyncPodAccessor):
    """Async accessor for the ``crypto`` namespace."""

    namespace = "crypto"

    # ---- published ---------------------------------------------------------

    async def get_binance_spot_1h(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Binance spot hourly OHLCV; cached export topped up from ``/v1/crypto/binance/spot``."""

        return await self._get(
            "binance_spot_1h",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_binance_spot_1d(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Binance spot daily OHLCV (bulk only; no live daily route)."""

        return await self._get(
            "binance_spot_1d",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_binance_perps_1h(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """USDT-margined Binance perps hourly OHLCV; live companion ``/v1/crypto/binance/perps``."""

        return await self._get(
            "binance_perps_1h",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_binance_perps_funding(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """USDT-margined perps funding rates and mark prices; live companion ``/v1/crypto/binance/perps/funding``."""

        return await self._get(
            "binance_perps_funding",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_binance_coin_m_perps_1h(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Coin-margined (inverse) perps hourly OHLCV; live companion ``/v1/crypto/binance/coin-m/perps``."""

        return await self._get(
            "binance_coin_m_perps_1h",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_binance_coin_m_perps_funding(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Coin-margined perps funding rates; live companion ``/v1/crypto/binance/coin-m/funding``."""

        return await self._get(
            "binance_coin_m_funding",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    async def get_coincodex(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """CoinCodex daily price / volume / market cap (bulk only)."""

        return await self._get(
            "coincodex",
            source=source,
            start=start,
            end=end,
            symbols=symbols,
            force_refresh=force_refresh,
        )

    # ---- binance perps (not seeded yet) ----------------------------------

    async def get_binance_perps_1h_all(self, *, force_refresh: bool = False) -> pd.DataFrame:
        """Unfiltered Binance perps OHLCV (the ``_all`` rwRTools variant)."""

        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="binance.perps",
            schema="ohlcv-1h-all",
            method="crypto_get_binance_perps_1h_all",
        )

    # ---- index / aggregated series (not seeded yet) ----------------------

    async def get_coinmetrics(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="coinmetrics",
            schema="snapshot",
            method="crypto_get_coinmetrics",
        )

    # ---- FTX legacy series (not seeded yet; FTX exchange defunct) --------

    async def get_spot(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="ftx.spot",
            schema="ohlcv-1h",
            method="crypto_get_spot",
        )

    async def get_clean_spot(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="ftx.spot.clean",
            schema="ohlcv-1h",
            method="crypto_get_clean_spot",
        )

    async def get_index(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="ftx.index",
            schema="ohlcv-1h",
            method="crypto_get_index",
        )

    async def get_futures(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="ftx.futures",
            schema="ohlcv-1h",
            method="crypto_get_futures",
        )

    async def get_expired_futures(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="ftx.futures.expired",
            schema="ohlcv-1h",
            method="crypto_get_expired_futures",
        )

    async def get_lending_rates(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="ftx",
            schema="lending-rates",
            method="crypto_get_lending_rates",
        )

    async def get_perp_rates(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="ftx.perps",
            schema="funding",
            method="crypto_get_perp_rates",
        )

    async def get_minute_perpetuals(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="ftx.perps",
            schema="ohlcv-1m",
            method="crypto_get_minute_perpetuals",
        )

    async def get_rebalance_trades(self, *, force_refresh: bool = False) -> pd.DataFrame:
        raise DatasetNotSeededError(
            namespace="crypto",
            dataset="ftx.rebalance_trades",
            schema="snapshot",
            method="crypto_get_rebalance_trades",
        )


class CryptoPod(SyncPodAccessor[AsyncCryptoPod]):
    """Synchronous accessor for the ``crypto`` namespace."""

    namespace = "crypto"

    def get_binance_spot_1h(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_binance_spot_1h(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_binance_spot_1d(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_binance_spot_1d(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_binance_perps_1h(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_binance_perps_1h(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_binance_perps_funding(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_binance_perps_funding(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_binance_coin_m_perps_1h(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_binance_coin_m_perps_1h(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_binance_coin_m_perps_funding(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_binance_coin_m_perps_funding(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_coincodex(
        self,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        symbols: SymbolsLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_coincodex(
                source=source,
                start=start,
                end=end,
                symbols=symbols,
                force_refresh=force_refresh,
            )
        )

    def get_binance_perps_1h_all(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_binance_perps_1h_all(force_refresh=force_refresh))

    def get_coinmetrics(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_coinmetrics(force_refresh=force_refresh))

    def get_spot(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_spot(force_refresh=force_refresh))

    def get_clean_spot(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_clean_spot(force_refresh=force_refresh))

    def get_index(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_index(force_refresh=force_refresh))

    def get_futures(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_futures(force_refresh=force_refresh))

    def get_expired_futures(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_expired_futures(force_refresh=force_refresh))

    def get_lending_rates(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_lending_rates(force_refresh=force_refresh))

    def get_perp_rates(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_perp_rates(force_refresh=force_refresh))

    def get_minute_perpetuals(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_minute_perpetuals(force_refresh=force_refresh))

    def get_rebalance_trades(self, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_rebalance_trades(force_refresh=force_refresh))


__all__ = ["AsyncCryptoPod", "CryptoPod"]
