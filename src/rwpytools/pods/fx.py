"""FX namespace data loaders and client-side helpers.

Bulk API surface maps to ``/v1/fx/file`` on rw-api. The FX datasets are
symbol-templated: one file per (schema, ticker) pair under
``feather/{Daily,Hourly}/`` in the source bucket, and one policy-rate
history per currency.

Daily bars are routed like every other dataset: each pair's cached export
is topped up to real time from ``/v1/fx/daily`` (one live request covers
every pair). Hourly bars and policy rates have no live route and are served
from the export. Multi-ticker methods download missing pairs in parallel,
bounded by ``ClientConfig.download_concurrency``.

Mirrors the ``fx_*`` rwRTools surface 1:1. ``get_asset_list`` raises
:class:`rwpytools.errors.DatasetNotSeededError` — rw-api does not publish
the Zorro asset lists.

The three pure-helper functions at the bottom (``get_unique_currencies``,
``total_return_index``, ``convert_common_quote_currency``) are
client-side computations — they do not hit rw-api and accept dataframes
as inputs. They are exposed as ``@staticmethod`` so they're reachable as
``client.fx.<name>(...)`` for parity with rwRtools, but they don't
require an instance state. Ported from rwRtools' ``fx_data_utils.R``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..errors import DatasetNotSeededError
from ..http import run_sync
from ..routing import Source
from ._base import AsyncPodAccessor, DateLike, SourceLike, SyncPodAccessor

_DATASET = "pairs"

# rwRtools rename, positional (R does `colnames(td) <- c(...)` which is a
# positional assignment regardless of source column names). Match it 1:1
# so a notebook ported from R doesn't need to touch column references.
_DAILY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
_HOURLY_COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume", "Ticker"]


def _apply_rwrtools_fx_rename(frame: pd.DataFrame, *, daily: bool) -> pd.DataFrame:
    """Rename and coerce columns to match the rwRtools FX shape.

    The export's columns are ``Date|datetime, open, high, low, close,
    volume, ticker``; they are matched **by name** (case-insensitive), so
    the result is right whatever order the columns arrive in — rows served
    live-only come from JSON in alphabetical key order. Only when the names
    are unrecognised does this fall back to rwRtools' positional
    ``colnames(td) <- c(...)``. Extra columns pass through untouched.
    """

    target = _DAILY_COLUMNS if daily else _HOURLY_COLUMNS
    time_col = target[0]
    out = frame.copy()
    by_lower = {str(c).lower(): c for c in out.columns}
    wanted = [time_col.lower(), "open", "high", "low", "close", "volume", "ticker"]
    aliases = {"date": ("date", "datetime"), "datetime": ("datetime", "date")}
    rename_map: dict[str, str] = {}
    for lower, new_name in zip(wanted, target, strict=True):
        for candidate in aliases.get(lower, (lower,)):
            if candidate in by_lower:
                rename_map[by_lower[candidate]] = new_name
                break
    if len(rename_map) < len(target):
        # Unrecognised schema: positional, exactly as rwRtools does it.
        rename_map = {
            out.columns[idx]: new_name
            for idx, new_name in enumerate(target)
            if idx < len(out.columns)
        }
    out = out.rename(columns=rename_map)
    ordered = [c for c in target if c in out.columns]
    out = out[ordered + [c for c in out.columns if c not in ordered]]
    # rwRtools coerces the time column and ticker to canonical types.
    if time_col in out.columns:
        out[time_col] = (
            pd.to_datetime(out[time_col]).dt.date if daily else pd.to_datetime(out[time_col])
        )
    if "Ticker" in out.columns:
        out["Ticker"] = out["Ticker"].astype(str)
    return out


def _order_by_input(frame: pd.DataFrame, tickers: Sequence[str], *, daily: bool) -> pd.DataFrame:
    """Group rows by ticker in the caller's input order, by time within each.

    Mirrors rwRtools' ``dplyr::bind_rows(purrr::map(tickers, ...))`` —
    rows from the first ticker appear before rows from the second, and so
    on. Needed because the live top-up appends its rows after the bulk
    rows of every pair.
    """

    if frame.empty or "Ticker" not in frame.columns:
        return frame
    time_col = "Date" if daily else "Datetime"
    rank = {t: i for i, t in enumerate(tickers)}
    order = frame["Ticker"].map(rank).fillna(len(rank))
    keys = pd.DataFrame({"_rank": order, "_time": frame[time_col]}, index=frame.index)
    ordered = keys.sort_values(["_rank", "_time"], kind="stable").index
    return frame.loc[ordered].reset_index(drop=True)


def _split_by_ticker(frame: pd.DataFrame, tickers: Sequence[str]) -> dict[str, pd.DataFrame]:
    if "Ticker" not in frame.columns:
        return {t: frame.iloc[0:0] for t in tickers}
    return {t: frame[frame["Ticker"] == t].reset_index(drop=True) for t in tickers}


def _reject_string_ticker_list(tickers: Sequence[str]) -> list[str]:
    """Guard against ``get_daily_ohlc("EURUSD")`` — iterating chars.

    The plural form requires an iterable of tickers; if the caller
    accidentally passes a bare string we iterate over its characters
    and silently produce six garbage requests. Raise loudly instead.
    """

    if isinstance(tickers, str | bytes):
        raise TypeError(
            "get_*_ohlc expects a sequence of ticker strings, not a "
            'single string; did you mean get_*_ohlc_ticker("...")?'
        )
    out = [str(t) for t in tickers]
    if not out:
        raise ValueError("ticker list must contain at least one symbol")
    return out


# ---------------------------------------------------------------------------
# Client-side helpers (no API call). Ported 1:1 from rwRtools'
# fx_data_utils.R. Implemented as module-level functions and re-exposed
# as @staticmethod on Async/Sync pods so callers can reach them via
# ``client.fx.<name>(...)``.
# ---------------------------------------------------------------------------


def _get_unique_currencies_impl(prices_df: pd.DataFrame) -> list[str]:
    """Return the unique base+quote currencies present in ``prices_df``.

    Mirrors ``fx_get_unique_currencies(prices_df)`` from rwRtools.
    ``prices_df`` must have a ``Ticker`` column whose values are
    six-character pairs (``EURUSD``, ``GBPJPY``, ...).
    """

    if "Ticker" not in prices_df.columns:
        raise ValueError("prices_df must have a 'Ticker' column")
    tickers = prices_df["Ticker"].dropna().astype(str)
    bases = tickers.str.slice(0, 3)
    quotes = tickers.str.slice(-3)
    return sorted(set(bases.unique()) | set(quotes.unique()))


def _total_return_index_impl(
    prices_df: pd.DataFrame, policy_rates_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute the FX total-return index for each ticker.

    Mirrors ``fx_total_return_index(prices_df, policy_rates_df)`` from
    rwRtools. Inputs:

    * ``prices_df`` must carry ``Ticker``, ``Date``, ``Close``.
    * ``policy_rates_df`` must carry ``Currency``, ``Date``, ``Rate``
      (rates are in *percent*, matching the rwRtools convention).

    Returns the input dataframe extended with ``Base``, ``Quote``,
    ``Base_Rate``, ``Quote_Rate``, ``Rate_Diff``,
    ``Daycount_Fraction``, ``Interest_Returns``,
    ``Interest_Accrual_on_Spot``, ``Spot_Returns``,
    ``Spot_Return_Index``, ``Interest_Return_Index``, and
    ``Total_Return_Index`` columns.

    Day-count uses Actual/365 like rwRtools.
    """

    required_prices = {"Ticker", "Date", "Close"}
    required_rates = {"Currency", "Date", "Rate"}
    missing_p = required_prices - set(prices_df.columns)
    missing_r = required_rates - set(policy_rates_df.columns)
    if missing_p:
        raise ValueError(f"prices_df missing columns: {sorted(missing_p)}")
    if missing_r:
        raise ValueError(f"policy_rates_df missing columns: {sorted(missing_r)}")

    out = prices_df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    rates = policy_rates_df.copy()
    rates["Date"] = pd.to_datetime(rates["Date"])

    out["Base"] = out["Ticker"].astype(str).str.slice(0, 3)
    out["Quote"] = out["Ticker"].astype(str).str.slice(-3)

    base_rates = rates.rename(columns={"Currency": "Base", "Rate": "Base_Rate_pct"})
    quote_rates = rates.rename(columns={"Currency": "Quote", "Rate": "Quote_Rate_pct"})
    out = out.merge(base_rates, on=["Base", "Date"], how="left")
    out = out.merge(quote_rates, on=["Quote", "Date"], how="left")

    out = out.sort_values(["Ticker", "Date"])
    # Forward-fill rates per ticker (rwRtools uses zoo::na.locf(na.rm=FALSE)).
    out["Base_Rate"] = out.groupby("Ticker")["Base_Rate_pct"].ffill() * 0.01
    out["Quote_Rate"] = out.groupby("Ticker")["Quote_Rate_pct"].ffill() * 0.01
    out["Rate_Diff"] = out["Base_Rate"] - out["Quote_Rate"]

    out["_PrevDate"] = out.groupby("Ticker")["Date"].shift(1)
    out["Daycount_Fraction"] = (out["Date"] - out["_PrevDate"]).dt.days / 365.0
    out["Interest_Returns"] = out["Daycount_Fraction"] * out["Rate_Diff"]
    out["Interest_Accrual_on_Spot"] = out["Interest_Returns"] * out["Close"]
    out["Spot_Returns"] = out.groupby("Ticker")["Close"].pct_change()

    out = out.drop(columns=["Base_Rate_pct", "Quote_Rate_pct", "_PrevDate"])
    out = out.dropna(subset=["Spot_Returns", "Interest_Returns", "Daycount_Fraction"])

    # rwRtools uses cumprod(1 + Spot_Returns); replicate that exactly.
    grp = out.groupby("Ticker")
    out["Spot_Return_Index"] = grp["Spot_Returns"].transform(lambda s: (1.0 + s).cumprod())
    out["Interest_Return_Index"] = grp["Interest_Returns"].transform(lambda s: (1.0 + s).cumprod())
    # Total_Return_Index needs both columns; compute via a scratch column
    # so the transform stays a Series-in Series-out shape.
    out["_combined"] = out["Spot_Returns"] + out["Interest_Returns"]
    out["Total_Return_Index"] = out.groupby("Ticker")["_combined"].transform(
        lambda s: (1.0 + s).cumprod()
    )
    out = out.drop(columns=["_combined"])
    return out


def _convert_common_quote_currency_impl(
    prices_df: pd.DataFrame, quote_currency: str = "USD"
) -> pd.DataFrame:
    """Re-express every pair so the quote currency matches ``quote_currency``.

    Mirrors ``fx_convert_common_quote_currency(prices_df, quote_currency)``
    from rwRtools. Pairs where the *base* is the target currency are
    inverted (``USDJPY → JPYUSD``, OHLC reciprocated); pairs already
    quoted in the target are passed through unchanged. Pairs unrelated
    to ``quote_currency`` are dropped.

    Requires ``Ticker``, ``Open``, ``High``, ``Low``, ``Close`` columns
    on ``prices_df``.
    """

    required = {"Ticker", "Open", "High", "Low", "Close"}
    missing = required - set(prices_df.columns)
    if missing:
        raise ValueError(f"prices_df missing columns: {sorted(missing)}")

    df = prices_df.copy()
    df["Base"] = df["Ticker"].astype(str).str.slice(0, 3)
    df["Quote"] = df["Ticker"].astype(str).str.slice(-3)

    # Already-quoted pairs (Base != target, Quote == target) pass through.
    direct = df[(df["Base"] != quote_currency) & (df["Quote"] == quote_currency)].copy()
    direct = direct.drop(columns=["Base", "Quote"])

    # Inverted pairs (Base == target, Quote != target) — invert OHLC.
    inverse = df[(df["Base"] == quote_currency) & (df["Quote"] != quote_currency)].copy()
    inverse["Ticker"] = inverse["Quote"] + inverse["Base"]
    new_high = 1.0 / inverse["Low"]
    new_low = 1.0 / inverse["High"]
    inverse["Open"] = 1.0 / inverse["Open"]
    inverse["Close"] = 1.0 / inverse["Close"]
    inverse["High"] = new_high
    inverse["Low"] = new_low
    inverse = inverse.drop(columns=["Base", "Quote"])

    return pd.concat([inverse, direct], ignore_index=True)


# ---------------------------------------------------------------------------
# Async / sync pod classes
# ---------------------------------------------------------------------------


class AsyncFxPod(AsyncPodAccessor):
    """Async accessor for the ``fx`` namespace."""

    namespace = "fx"
    DATASET = _DATASET

    # ---- single ticker ------------------------------------------------------

    async def get_daily_ohlc_ticker(
        self,
        ticker: str,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Daily OHLC for one ticker, topped up live. Returns rwRtools-shaped
        columns (``Date, Open, High, Low, Close, Volume, Ticker``)."""

        return await self.get_daily_ohlc(
            [ticker], source=source, start=start, end=end, force_refresh=force_refresh
        )

    async def get_hourly_ohlc_ticker(
        self,
        ticker: str,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Hourly OHLC for one ticker (bulk only). Returns rwRtools-shaped
        columns (``Datetime, Open, High, Low, Close, Volume, Ticker``)."""

        return await self.get_hourly_ohlc(
            [ticker], source=source, start=start, end=end, force_refresh=force_refresh
        )

    # ---- many tickers -----------------------------------------------------------

    async def get_daily_ohlc(
        self,
        tickers: Sequence[str],
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Daily OHLC for many tickers as a single concatenated frame.

        Mirrors rwRtools' ``fx_get_daily_OHLC(tickers)``: rows are grouped
        by ticker in input order with columns ``Date, Open, High, Low,
        Close, Volume, Ticker``. Each pair's cached export is topped up from
        the live ``/v1/fx/daily`` route.
        """

        ticker_list = _reject_string_ticker_list(tickers)
        frame = await self._get(
            "fx_daily",
            source=source,
            start=start,
            end=end,
            symbols=ticker_list,
            force_refresh=force_refresh,
        )
        return _order_by_input(
            _apply_rwrtools_fx_rename(frame, daily=True), ticker_list, daily=True
        )

    async def get_hourly_ohlc(
        self,
        tickers: Sequence[str],
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Hourly OHLC for many tickers as a single concatenated frame.

        See :meth:`get_daily_ohlc` for shape notes; this variant returns
        columns ``Datetime, Open, High, Low, Close, Volume, Ticker``.
        """

        ticker_list = _reject_string_ticker_list(tickers)
        frame = await self._get(
            "fx_hourly",
            source=source,
            start=start,
            end=end,
            symbols=ticker_list,
            force_refresh=force_refresh,
        )
        return _order_by_input(
            _apply_rwrtools_fx_rename(frame, daily=False), ticker_list, daily=False
        )

    async def get_daily_ohlc_frames(
        self,
        tickers: Sequence[str],
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Same data as :meth:`get_daily_ohlc`, keyed by ticker."""

        ticker_list = _reject_string_ticker_list(tickers)
        frame = await self.get_daily_ohlc(
            ticker_list, source=source, start=start, end=end, force_refresh=force_refresh
        )
        return _split_by_ticker(frame, ticker_list)

    async def get_hourly_ohlc_frames(
        self,
        tickers: Sequence[str],
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Same data as :meth:`get_hourly_ohlc`, keyed by ticker."""

        ticker_list = _reject_string_ticker_list(tickers)
        frame = await self.get_hourly_ohlc(
            ticker_list, source=source, start=start, end=end, force_refresh=force_refresh
        )
        return _split_by_ticker(frame, ticker_list)

    # ---- discovery ------------------------------------------------------------------

    async def list_symbols(
        self, *, dataset: str = _DATASET, schema: str | None = None
    ) -> list[str]:
        """Symbols backed by actual files (``dataset="policy.rates"`` lists currencies)."""

        body = await self._bulk.list_symbols(self.namespace, dataset=dataset, schema=schema)
        return list(body.symbols)

    # ---- policy rates -----------------------------------------------------------------

    async def get_policy_rates(
        self,
        currency: str,
        *,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Central-bank policy-rate history for one currency (``USD``, ``EUR``, ...).

        Columns are as published: a ``Time Period`` date and one value
        column named for the central bank's country code (e.g. ``US``).
        """

        frame = await self._get(
            "fx_policy_rates",
            source=Source.BULK,
            start=start,
            end=end,
            symbols=[currency],
            force_refresh=force_refresh,
        )
        return frame.drop(columns=["currency"], errors="ignore")

    # ---- rwRTools parity, not published by rw-api ------------------------------------

    async def get_asset_list(self, asset_list: str, *, force_refresh: bool = False) -> pd.DataFrame:
        """Asset-list CSV (Zorro-style universe definition). Not published
        by rw-api; raises :class:`rwpytools.errors.DatasetNotSeededError`."""

        raise DatasetNotSeededError(
            namespace="fx",
            dataset="asset_list",
            schema="snapshot",
            method="fx_get_asset_list",
        )

    # ---- rwRTools parity, client-side helpers (no API call) --------------------------

    get_unique_currencies = staticmethod(_get_unique_currencies_impl)
    total_return_index = staticmethod(_total_return_index_impl)
    convert_common_quote_currency = staticmethod(_convert_common_quote_currency_impl)


class FxPod(SyncPodAccessor[AsyncFxPod]):
    """Synchronous accessor for the ``fx`` namespace."""

    namespace = "fx"
    DATASET = _DATASET

    def get_daily_ohlc_ticker(
        self,
        ticker: str,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_daily_ohlc_ticker(
                ticker, source=source, start=start, end=end, force_refresh=force_refresh
            )
        )

    def get_hourly_ohlc_ticker(
        self,
        ticker: str,
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_hourly_ohlc_ticker(
                ticker, source=source, start=start, end=end, force_refresh=force_refresh
            )
        )

    def get_daily_ohlc(
        self,
        tickers: Sequence[str],
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_daily_ohlc(
                tickers, source=source, start=start, end=end, force_refresh=force_refresh
            )
        )

    def get_hourly_ohlc(
        self,
        tickers: Sequence[str],
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_hourly_ohlc(
                tickers, source=source, start=start, end=end, force_refresh=force_refresh
            )
        )

    def get_daily_ohlc_frames(
        self,
        tickers: Sequence[str],
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        return run_sync(
            self._async.get_daily_ohlc_frames(
                tickers, source=source, start=start, end=end, force_refresh=force_refresh
            )
        )

    def get_hourly_ohlc_frames(
        self,
        tickers: Sequence[str],
        *,
        source: SourceLike = Source.AUTO,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        return run_sync(
            self._async.get_hourly_ohlc_frames(
                tickers, source=source, start=start, end=end, force_refresh=force_refresh
            )
        )

    def list_symbols(self, *, dataset: str = _DATASET, schema: str | None = None) -> list[str]:
        return run_sync(self._async.list_symbols(dataset=dataset, schema=schema))

    def get_policy_rates(
        self,
        currency: str,
        *,
        start: DateLike = None,
        end: DateLike = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return run_sync(
            self._async.get_policy_rates(
                currency, start=start, end=end, force_refresh=force_refresh
            )
        )

    def get_asset_list(self, asset_list: str, *, force_refresh: bool = False) -> pd.DataFrame:
        return run_sync(self._async.get_asset_list(asset_list, force_refresh=force_refresh))

    get_unique_currencies = staticmethod(_get_unique_currencies_impl)
    total_return_index = staticmethod(_total_return_index_impl)
    convert_common_quote_currency = staticmethod(_convert_common_quote_currency_impl)


__all__ = ["AsyncFxPod", "FxPod"]
