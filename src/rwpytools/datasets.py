"""The dataset registry: which bulk export pairs with which live endpoint.

rw-api has no coverage catalog, and a dataset's bulk export and its live
companion are different products with different column names — the
Binance hourly export says ``Ticker`` / ``Datetime`` / ``Open`` where the
live route says ``ticker`` / ``date`` / ``open``. This module is the one
place that knowledge lives. Each :class:`DatasetSpec` names:

* the bulk cell (``namespace`` / ``dataset`` / ``schema``) and whether it is
  published once per symbol or per year;
* the live route, its paging style, and the ``live -> bulk`` column rename;
* the date and symbol columns (bulk names) used to filter, find the
  watermark, and de-duplicate the seam.

Rows are always returned in the **bulk** schema: live rows are renamed to
bulk column names and reindexed to the bulk columns, so a result that
straddles the watermark is rectangular no matter which source served each
row. Bulk-only columns (e.g. ``Number of trades``) are ``NaN`` on live rows.

Every entry here was checked against the production API.
"""

from __future__ import annotations

import difflib
import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import pandas as pd

from .errors import RoutingError


class Template(str, enum.Enum):
    """How a bulk dataset is split into files."""

    NONE = "none"  # one file for the whole dataset
    SYMBOL = "symbol"  # one file per symbol (FX pairs, policy-rate currencies)
    YEAR = "year"  # one file per calendar year (IB short-stock history)


class LiveStyle(str, enum.Enum):
    """The request shape of a live endpoint."""

    #: ``gte`` / ``lte`` dates, an optional symbol filter, cursor-paginated.
    RANGE = "range"
    #: ``gte`` / ``lte`` / ``limit`` (max 50,000); not paginated.
    TOPSPREADS = "topspreads"
    #: ``lookback_days`` (1-90) trailing window plus a ``tickers`` filter.
    LOOKBACK = "lookback"
    #: No date filter; the freshest row per symbol, cursor-paginated.
    SNAPSHOT = "snapshot"
    #: No parameters; the latest published snapshot.
    LATEST = "latest"
    #: ``days`` (1-365) trailing window.
    DAYS = "days"


@dataclass(frozen=True)
class BulkSource:
    namespace: str
    dataset: str
    schema: str
    template: Template = Template.NONE


@dataclass(frozen=True)
class LiveSource:
    path: str
    style: LiveStyle = LiveStyle.RANGE
    #: Query parameter that filters by symbol (``ticker``, ``symbol``,
    #: ``tickers``); ``None`` when the route has no symbol filter.
    symbol_param: str | None = None
    #: ``live column -> bulk column`` for columns whose names differ.
    rename: Mapping[str, str] = field(default_factory=dict)


Deriver = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    bulk: BulkSource | None = None
    live: LiveSource | None = None
    #: Date / timestamp column, in bulk naming. ``None`` for snapshots.
    date_column: str | None = None
    #: Symbol column, in bulk naming. ``None`` when rows carry no symbol.
    symbol_column: str | None = None
    #: Columns identifying a row, used to drop live rows already present in
    #: the bulk half. Defaults to ``(symbol_column, date_column)``.
    key_columns: tuple[str, ...] = ()
    #: Fix-ups applied to live rows after renaming (e.g. derive a bulk-only
    #: epoch column from a timestamp).
    derive: Deriver | None = None
    #: Only keep live rows whose symbol appears in the bulk half. For live
    #: tables that carry extra series (``/macro/vix`` adds ``SPVIXSTR``),
    #: which would otherwise appear for the last few days only. Off by
    #: default: new listings and new reporters are real rows.
    restrict_to_bulk_symbols: bool = False

    @property
    def has_bulk(self) -> bool:
        return self.bulk is not None

    @property
    def has_live(self) -> bool:
        return self.live is not None

    @property
    def keys(self) -> tuple[str, ...]:
        if self.key_columns:
            return self.key_columns
        return tuple(c for c in (self.symbol_column, self.date_column) if c)

    @property
    def templated(self) -> bool:
        return self.bulk is not None and self.bulk.template is not Template.NONE


# ---------------------------------------------------------------------------
# column maps shared by several entries
# ---------------------------------------------------------------------------

_BINANCE_OHLCV = MappingProxyType(
    {
        "ticker": "Ticker",
        "date": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
)
_BINANCE_FUNDING = MappingProxyType(
    {
        "ticker": "Ticker",
        "funding_time": "fundingTimeHR",
        "funding_rate": "fundingRate",
        "mark_price": "markPrice",
    }
)


def _derive_funding_epoch(frame: pd.DataFrame) -> pd.DataFrame:
    """Live funding rows carry only the timestamp; the export also has the
    raw Binance ``fundingTime`` (epoch milliseconds). Rebuild it."""

    if "fundingTimeHR" in frame.columns and len(frame):
        stamps = pd.to_datetime(frame["fundingTimeHR"], format="ISO8601")
        frame["fundingTime"] = stamps.astype("datetime64[ms]").astype("int64")
    return frame


_SPECS: tuple[DatasetSpec, ...] = (
    # ---- crypto --------------------------------------------------------------
    DatasetSpec(
        name="binance_spot_1h",
        description="Binance spot hourly OHLCV.",
        bulk=BulkSource("crypto", "binance.spot", "ohlcv-1h"),
        live=LiveSource("/v1/crypto/binance/spot", symbol_param="ticker", rename=_BINANCE_OHLCV),
        date_column="Datetime",
        symbol_column="Ticker",
    ),
    DatasetSpec(
        name="binance_spot_1d",
        description="Binance spot daily OHLCV.",
        bulk=BulkSource("crypto", "binance.spot", "ohlcv-1d"),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="binance_perps_1h",
        description="Binance USDT-margined perpetuals hourly OHLCV.",
        bulk=BulkSource("crypto", "binance.perps", "ohlcv-1h"),
        live=LiveSource("/v1/crypto/binance/perps", symbol_param="ticker", rename=_BINANCE_OHLCV),
        date_column="Datetime",
        symbol_column="Ticker",
    ),
    DatasetSpec(
        name="binance_perps_funding",
        description="Binance USDT-margined perpetuals funding rates and mark prices.",
        bulk=BulkSource("crypto", "binance.perps", "funding"),
        live=LiveSource(
            "/v1/crypto/binance/perps/funding", symbol_param="ticker", rename=_BINANCE_FUNDING
        ),
        date_column="fundingTimeHR",
        symbol_column="Ticker",
        derive=_derive_funding_epoch,
    ),
    DatasetSpec(
        name="binance_coin_m_perps_1h",
        description="Binance coin-margined perpetuals hourly OHLCV.",
        bulk=BulkSource("crypto", "binance.perps.coin_m", "ohlcv-1h"),
        live=LiveSource(
            "/v1/crypto/binance/coin-m/perps", symbol_param="ticker", rename=_BINANCE_OHLCV
        ),
        date_column="Datetime",
        symbol_column="Ticker",
    ),
    DatasetSpec(
        name="binance_coin_m_funding",
        description="Binance coin-margined perpetuals funding rates and mark prices.",
        bulk=BulkSource("crypto", "binance.perps.coin_m", "funding"),
        live=LiveSource(
            "/v1/crypto/binance/coin-m/funding", symbol_param="ticker", rename=_BINANCE_FUNDING
        ),
        date_column="fundingTimeHR",
        symbol_column="Ticker",
        derive=_derive_funding_epoch,
    ),
    DatasetSpec(
        name="coincodex",
        description="CoinCodex daily price, volume and market cap.",
        bulk=BulkSource("crypto", "coincodex.marketcap", "snapshot"),
        date_column="Date",
        symbol_column="Ticker",
    ),
    # ---- macro ---------------------------------------------------------------
    DatasetSpec(
        name="asset_classes",
        description="Main asset-class ETFs, daily OHLCV.",
        bulk=BulkSource("macro", "asset_classes", "ohlcv-1d"),
        live=LiveSource("/v1/macro/asset-classes", symbol_param="ticker"),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="earnings_calendar",
        description="Finnhub earnings calendar.",
        bulk=BulkSource("macro", "earnings", "calendar"),
        date_column="date",
        symbol_column="symbol",
    ),
    DatasetSpec(
        name="futures",
        description="Norgate continuous futures, daily OHLCV with contract metadata.",
        bulk=BulkSource("macro", "futures", "ohlcv-1d"),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="rp_futures",
        description="Risk-premia futures universe, daily OHLCV by contract.",
        bulk=BulkSource("macro", "rp.futures", "ohlcv-1d"),
        live=LiveSource("/v1/macro/futures", symbol_param="ticker"),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="vix",
        description="VIX and VIX3M daily OHLC.",
        bulk=BulkSource("macro", "vix", "ohlcv-1d"),
        live=LiveSource("/v1/macro/vix", symbol_param="ticker"),
        date_column="date",
        symbol_column="ticker",
        # The live table also carries SPVIXSTR, which the export does not.
        restrict_to_bulk_symbols=True,
    ),
    DatasetSpec(
        name="vix_futures",
        description="VX futures, daily OHLCV by contract.",
        bulk=BulkSource("macro", "vix.futures", "ohlcv-1d"),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="tbill_rates",
        description="Short-term rate series (e.g. 13-week T-bill, %IRX).",
        bulk=BulkSource("macro", "tbill", "rates"),
        live=LiveSource("/v1/macro/rates", symbol_param="symbol"),
        date_column="date",
        symbol_column="symbol",
    ),
    DatasetSpec(
        name="ib_shortstock",
        description="Interactive Brokers short-stock availability; one export per year.",
        bulk=BulkSource("macro", "ib.shortstock", "yearly", Template.YEAR),
        live=LiveSource("/v1/equities/ib-short-stocks", symbol_param="ticker"),
        date_column="date",
        symbol_column="ticker",
        key_columns=("date", "con"),
    ),
    # ---- fx ------------------------------------------------------------------
    DatasetSpec(
        name="fx_daily",
        description="FX pairs daily OHLCV; one export per pair.",
        bulk=BulkSource("fx", "pairs", "ohlcv-1d", Template.SYMBOL),
        live=LiveSource("/v1/fx/daily", symbol_param="ticker", rename={"date": "Date"}),
        date_column="Date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="fx_hourly",
        description="FX pairs hourly OHLCV; one export per pair.",
        bulk=BulkSource("fx", "pairs", "ohlcv-1h", Template.SYMBOL),
        date_column="datetime",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="fx_policy_rates",
        description="Central-bank policy-rate history; one export per currency.",
        bulk=BulkSource("fx", "policy.rates", "history", Template.SYMBOL),
        date_column="Time Period",
        symbol_column="currency",
    ),
    # ---- equities ------------------------------------------------------------
    DatasetSpec(
        name="earnings_surprises",
        description="Equity earnings surprises (actuals vs estimates) with fundamentals.",
        bulk=BulkSource("equities", "earnings", "surprises"),
        live=LiveSource(
            "/v1/equities/earnings/recent-surprises",
            style=LiveStyle.LOOKBACK,
            symbol_param="tickers",
        ),
        date_column="report_date",
        symbol_column="symbol",
        key_columns=("event_key",),
    ),
    DatasetSpec(
        name="liquid_universe_prices",
        description="Daily prices for the liquid equity universe.",
        bulk=BulkSource("equities", "equity.factors", "prices"),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="liquid_universe_sectors",
        description="Sector classification for the liquid equity universe.",
        bulk=BulkSource("equities", "equity.factors", "sectors"),
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="statarb_spreads",
        description="StatArb daily pair factors (top-spreads universe).",
        bulk=BulkSource("equities", "statarb", "spreads"),
        live=LiveSource("/v1/equities/statarb/daily/topspreads", style=LiveStyle.TOPSPREADS),
        date_column="date",
        symbol_column="ticker",
        key_columns=("date", "ticker", "stock2"),
    ),
    DatasetSpec(
        name="statarb_prices",
        description="Daily OHLCV for the StatArb pair universe.",
        bulk=BulkSource("equities", "statarb", "prices"),
        live=LiveSource("/v1/equities/statarb/prices", symbol_param="ticker"),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="statarb_acquisitions",
        description="Equity acquisitions and delistings (corporate actions).",
        bulk=BulkSource("equities", "statarb", "acquisitions"),
        live=LiveSource("/v1/equities/acquisitions", symbol_param="ticker"),
        date_column="date",
        symbol_column="ticker",
        key_columns=("date", "ticker", "action"),
    ),
    DatasetSpec(
        name="statarb_liquid_spreads",
        description="StatArb daily pair factors (liquid-universe top spreads).",
        bulk=BulkSource("equities", "statarb.liquid", "spreads"),
        live=LiveSource("/v1/equities/statarb/liquid/daily/topspreads", style=LiveStyle.TOPSPREADS),
        date_column="date",
        symbol_column="ticker",
        key_columns=("date", "ticker", "stock2"),
    ),
    DatasetSpec(
        name="statarb_liquid_prices",
        description="Daily OHLCV for the liquid-universe StatArb pairs.",
        bulk=BulkSource("equities", "statarb.liquid", "prices"),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="statarb_liquid_acquisitions",
        description="Acquisitions and delistings for the liquid StatArb universe.",
        bulk=BulkSource("equities", "statarb.liquid", "acquisitions"),
        live=LiveSource("/v1/equities/acquisitions", symbol_param="ticker"),
        date_column="date",
        symbol_column="ticker",
        key_columns=("date", "ticker", "action"),
    ),
    DatasetSpec(
        name="streaks",
        description="Extended streak factors.",
        bulk=BulkSource("equities", "streaks", "factors"),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="equity_tickers",
        description="Equity ticker metadata snapshot (sector, industry, exchange, ...).",
        bulk=BulkSource("equities", "tickers", "snapshot"),
        live=LiveSource("/v1/equities/tickers", style=LiveStyle.SNAPSHOT, symbol_param="tickers"),
        symbol_column="ticker",
        key_columns=("ticker",),
    ),
    # ---- live-only -------------------------------------------------------------
    DatasetSpec(
        name="statarb_hourly_top_spreads",
        description="StatArb intraday (hourly) pair factors, top-spreads universe.",
        live=LiveSource("/v1/equities/statarb/hourly/topspreads", style=LiveStyle.TOPSPREADS),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="statarb_liquid_hourly_top_spreads",
        description="StatArb intraday (hourly) pair factors, liquid-universe top spreads.",
        live=LiveSource(
            "/v1/equities/statarb/liquid/hourly/topspreads", style=LiveStyle.TOPSPREADS
        ),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="yolo_factors",
        description="Y.O.L.O strategy: latest factor values.",
        live=LiveSource("/v1/yolo/factors", style=LiveStyle.LATEST),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="yolo_weights",
        description="Y.O.L.O strategy: latest weights with arrival prices.",
        live=LiveSource("/v1/yolo/weights", style=LiveStyle.LATEST),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="yolo_volatilities",
        description="Y.O.L.O strategy: latest EWMA volatility estimates.",
        live=LiveSource("/v1/yolo/volatilities", style=LiveStyle.LATEST),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="yolo_historical",
        description="Y.O.L.O strategy: daily weights / volatility history (up to 365 days).",
        live=LiveSource("/v1/yolo/historical", style=LiveStyle.DAYS),
        date_column="date",
        symbol_column="ticker",
    ),
    DatasetSpec(
        name="rpschteroids_weights",
        description="Risk Premia on Schteroids: latest weights.",
        live=LiveSource("/v1/rpschteroids/weights", style=LiveStyle.LATEST),
        date_column="date",
        symbol_column="ticker",
    ),
)

DATASETS: Mapping[str, DatasetSpec] = MappingProxyType({s.name: s for s in _SPECS})


def get_spec(name: str) -> DatasetSpec:
    """Look up a dataset by name, with a did-you-mean on a typo."""

    try:
        return DATASETS[name]
    except KeyError:
        close = difflib.get_close_matches(name, list(DATASETS), n=3)
        hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else ""
        raise RoutingError(
            f"unknown dataset {name!r}{hint} (see rwpytools.DATASETS for the full list)"
        ) from None


def find_spec(namespace: str, dataset: str, schema: str) -> DatasetSpec | None:
    """The registry entry for a bulk ``(namespace, dataset, schema)`` cell."""

    for spec in _SPECS:
        b = spec.bulk
        if b is not None and (b.namespace, b.dataset, b.schema) == (namespace, dataset, schema):
            return spec
    return None


__all__ = [
    "DATASETS",
    "BulkSource",
    "DatasetSpec",
    "LiveSource",
    "LiveStyle",
    "Template",
    "find_spec",
    "get_spec",
]
