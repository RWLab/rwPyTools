"""Per-pod check that each method routes to the right dataset.

Download and merge mechanics are covered by ``test_bulk.py`` and
``test_session.py``; here we verify the contract each pod method encodes:
which registry dataset it asks the session for — and, through the
registry, which rw-api ``(namespace, dataset, schema)`` cell that is. This
mirrors the rw-api ``_FILES`` registry.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import pytest

from rwpytools import Client
from rwpytools.datasets import get_spec
from rwpytools.result import Result
from rwpytools.routing import DateRange, RoutePlan


def _plan(dataset: str) -> RoutePlan:
    return RoutePlan(
        dataset=dataset,
        requested=DateRange(),
        watermark=None,
        bulk_range=DateRange(),
        live_range=None,
        split_at=None,
        bulk_action=None,
        reason="stub",
        forced=False,
    )


class _RecordingSession:
    """Stand-in for :class:`AsyncSession` that records calls instead of routing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.frame = pd.DataFrame({"_": [1]})

    async def get(self, dataset: str, **kwargs: Any) -> Result:
        self.calls.append((dataset, kwargs))
        return Result(dataset=dataset, frame=self.frame, route=_plan(dataset))

    def cell(self, index: int = -1) -> tuple[str, str, str]:
        spec = get_spec(self.calls[index][0])
        assert spec.bulk is not None
        return (spec.bulk.namespace, spec.bulk.dataset, spec.bulk.schema)


@pytest.fixture
def stub_client(client: Client) -> tuple[Client, _RecordingSession]:
    """Replace the session behind every pod with a recording stub."""

    stub = _RecordingSession()
    for ns_attr in ("crypto", "fx", "macro", "equity_factors"):
        getattr(client, ns_attr)._async._session = stub
    return client, stub


# ---- crypto ---------------------------------------------------------------

_CRYPTO_EXPECTATIONS = [
    ("get_binance_spot_1h", "binance.spot", "ohlcv-1h"),
    ("get_binance_spot_1d", "binance.spot", "ohlcv-1d"),
    ("get_binance_perps_1h", "binance.perps", "ohlcv-1h"),
    ("get_binance_perps_funding", "binance.perps", "funding"),
    ("get_binance_coin_m_perps_1h", "binance.perps.coin_m", "ohlcv-1h"),
    ("get_binance_coin_m_perps_funding", "binance.perps.coin_m", "funding"),
    ("get_coincodex", "coincodex.marketcap", "snapshot"),
]


@pytest.mark.parametrize(("method", "dataset", "schema"), _CRYPTO_EXPECTATIONS)
def test_crypto_pod_method_maps_to_dataset_schema(
    stub_client: tuple[Client, _RecordingSession],
    method: str,
    dataset: str,
    schema: str,
) -> None:
    client, stub = stub_client
    getattr(client.crypto, method)()
    assert stub.cell() == ("crypto", dataset, schema)


def test_pod_methods_forward_source_and_range(
    stub_client: tuple[Client, _RecordingSession],
) -> None:
    client, stub = stub_client
    client.crypto.get_binance_spot_1h(
        source="live", start="2024-01-01", end=dt.date(2024, 2, 1), symbols=["BTCUSDT"]
    )
    dataset, kwargs = stub.calls[-1]
    assert dataset == "binance_spot_1h"
    assert kwargs == {
        "source": "live",
        "start": "2024-01-01",
        "end": dt.date(2024, 2, 1),
        "symbols": ["BTCUSDT"],
        "force_refresh": False,
    }


# ---- macro ----------------------------------------------------------------

_MACRO_EXPECTATIONS = [
    ("get_earnings", "earnings", "calendar"),
    ("get_vix_vix3m", "vix", "ohlcv-1d"),
    ("get_rates", "tbill", "rates"),
    ("get_historical_asset_class", "asset_classes", "ohlcv-1d"),
    ("get_expiring_rp_futures", "rp.futures", "ohlcv-1d"),
    ("get_expiring_futures", "futures", "ohlcv-1d"),
    ("get_expiring_vx_futures", "vix.futures", "ohlcv-1d"),
]


@pytest.mark.parametrize(("method", "dataset", "schema"), _MACRO_EXPECTATIONS)
def test_macro_pod_method_maps_to_dataset_schema(
    stub_client: tuple[Client, _RecordingSession],
    method: str,
    dataset: str,
    schema: str,
) -> None:
    client, stub = stub_client
    getattr(client.macro, method)()
    assert stub.cell() == ("macro", dataset, schema)


def test_macro_get_historical_short_sale_selects_the_year(
    stub_client: tuple[Client, _RecordingSession],
) -> None:
    """Templated by year; mirrors rwRtools macro_get_historical_short_sale."""

    client, stub = stub_client
    client.macro.get_historical_short_sale("2023")
    _dataset, kwargs = stub.calls[-1]
    assert stub.cell() == ("macro", "ib.shortstock", "yearly")
    assert (kwargs["start"], kwargs["end"]) == ("2023-01-01", "2023-12-31")


# ---- fx -------------------------------------------------------------------

_FX_DAILY = pd.DataFrame(
    {
        "Date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)] * 2,
        "open": [1.1, 1.2, 1.3, 1.4],
        "high": [1.1, 1.2, 1.3, 1.4],
        "low": [1.1, 1.2, 1.3, 1.4],
        "close": [1.1, 1.2, 1.3, 1.4],
        "volume": [1, 2, 3, 4],
        "ticker": ["GBPUSD", "GBPUSD", "EURUSD", "EURUSD"],
    }
)


def test_fx_daily_ticker_passes_symbol(stub_client: tuple[Client, _RecordingSession]) -> None:
    client, stub = stub_client
    client.fx.get_daily_ohlc_ticker("EURUSD")
    _dataset, kwargs = stub.calls[-1]
    assert stub.cell() == ("fx", "pairs", "ohlcv-1d")
    assert kwargs["symbols"] == ["EURUSD"]


def test_fx_hourly_ticker_passes_symbol(stub_client: tuple[Client, _RecordingSession]) -> None:
    client, stub = stub_client
    client.fx.get_hourly_ohlc_ticker("GBPJPY")
    assert stub.cell() == ("fx", "pairs", "ohlcv-1h")
    assert stub.calls[-1][1]["symbols"] == ["GBPJPY"]


def test_fx_bulk_daily_returns_rwrtools_shaped_frame_in_input_order(
    stub_client: tuple[Client, _RecordingSession],
) -> None:
    """rwRtools-1:1 shape: renamed columns, rows grouped by ticker in input order."""

    client, stub = stub_client
    stub.frame = _FX_DAILY
    out = client.fx.get_daily_ohlc(["EURUSD", "GBPUSD"])
    assert list(out.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    assert list(out["Ticker"]) == ["EURUSD", "EURUSD", "GBPUSD", "GBPUSD"]
    assert stub.calls[-1][1]["symbols"] == ["EURUSD", "GBPUSD"]


def test_fx_rename_is_by_name_not_position(
    stub_client: tuple[Client, _RecordingSession],
) -> None:
    """Live-only rows arrive with JSON's alphabetical key order."""

    client, stub = stub_client
    stub.frame = _FX_DAILY[["close", "Date", "high", "low", "open", "ticker", "volume"]]
    out = client.fx.get_daily_ohlc(["EURUSD"])
    assert list(out.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    assert out.loc[out["Ticker"] == "EURUSD", "Volume"].tolist() == [3, 4]


def test_fx_bulk_hourly_uses_hourly_schema(
    stub_client: tuple[Client, _RecordingSession],
) -> None:
    client, stub = stub_client
    client.fx.get_hourly_ohlc(["EURUSD", "GBPUSD"])
    assert stub.cell() == ("fx", "pairs", "ohlcv-1h")


def test_fx_bulk_daily_frames_dict_keys_by_ticker(
    stub_client: tuple[Client, _RecordingSession],
) -> None:
    """The `_frames` variant keeps the dict shape for power users."""

    client, stub = stub_client
    stub.frame = _FX_DAILY
    out = client.fx.get_daily_ohlc_frames(["EURUSD", "GBPUSD"])
    assert set(out.keys()) == {"EURUSD", "GBPUSD"}
    assert out["EURUSD"]["Close"].tolist() == [1.3, 1.4]


def test_fx_get_policy_rates_reads_the_currency_export(
    stub_client: tuple[Client, _RecordingSession],
) -> None:
    client, stub = stub_client
    stub.frame = pd.DataFrame({"Time Period": ["2024-01-01"], "US": [5.25], "currency": ["USD"]})
    out = client.fx.get_policy_rates("USD")
    assert stub.cell() == ("fx", "policy.rates", "history")
    assert stub.calls[-1][1]["symbols"] == ["USD"]
    assert list(out.columns) == ["Time Period", "US"]


# ---- equities (rwRTools parity) ------------------------------------------


_EQUITY_EXPECTATIONS = [
    ("get_tickers", "tickers", "snapshot"),
    ("get_liquid_universe", "equity.factors", "prices"),
    ("get_liquid_universe_sectors", "equity.factors", "sectors"),
    ("get_streaks", "streaks", "factors"),
    ("get_statarb_acquisitions", "statarb", "acquisitions"),
    ("get_statarb_prices", "statarb", "prices"),
    ("get_statarb_spreads", "statarb", "spreads"),
    ("get_statarb_liquid_acquisitions", "statarb.liquid", "acquisitions"),
    ("get_statarb_liquid_prices", "statarb.liquid", "prices"),
    ("get_statarb_liquid_spreads", "statarb.liquid", "spreads"),
    ("get_earnings_surprises", "earnings", "surprises"),
]


@pytest.mark.parametrize(("method", "dataset", "schema"), _EQUITY_EXPECTATIONS)
def test_equity_pod_method_maps_to_dataset_schema(
    stub_client: tuple[Client, _RecordingSession],
    method: str,
    dataset: str,
    schema: str,
) -> None:
    client, stub = stub_client
    getattr(client.equity_factors, method)()
    assert stub.cell() == ("equities", dataset, schema)


# ---- crypto / macro stubs -------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        "get_coinmetrics",
        "get_minute_perpetuals",
        "get_lending_rates",
        "get_perp_rates",
        "get_binance_perps_1h_all",
        "get_clean_spot",
        "get_spot",
        "get_index",
        "get_futures",
        "get_expired_futures",
        "get_rebalance_trades",
    ],
)
def test_crypto_rwrtools_stubs_raise_dataset_not_seeded(client: Client, method: str) -> None:
    from rwpytools.errors import DatasetNotSeededError

    with pytest.raises(DatasetNotSeededError):
        getattr(client.crypto, method)()


@pytest.mark.parametrize(
    "method",
    [
        "get_close_price_momo",
        "get_govt_bond_returns",
        "get_nyse_holidays",
        "get_straddles_over_earnings",
    ],
)
def test_macro_rwrtools_stubs_raise_dataset_not_seeded(client: Client, method: str) -> None:
    from rwpytools.errors import DatasetNotSeededError

    with pytest.raises(DatasetNotSeededError):
        getattr(client.macro, method)()


# ---- fx rwRTools-parity helpers (no API call) -----------------------------


def test_fx_get_unique_currencies_returns_sorted_set(client: Client) -> None:
    df = pd.DataFrame({"Ticker": ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"]})
    out = client.fx.get_unique_currencies(df)
    assert out == ["EUR", "GBP", "JPY", "USD"]


def test_fx_get_unique_currencies_requires_ticker_column(client: Client) -> None:
    with pytest.raises(ValueError, match="Ticker"):
        client.fx.get_unique_currencies(pd.DataFrame({"foo": [1]}))


def test_fx_convert_common_quote_currency_inverts_base_pairs(client: Client) -> None:
    prices = pd.DataFrame(
        [
            # USDJPY 110 → JPYUSD 1/110
            {"Ticker": "USDJPY", "Open": 110.0, "High": 111.0, "Low": 109.0, "Close": 110.5},
            # EURUSD passes through unchanged
            {"Ticker": "EURUSD", "Open": 1.10, "High": 1.11, "Low": 1.09, "Close": 1.105},
            # Unrelated to USD — dropped
            {"Ticker": "EURGBP", "Open": 0.85, "High": 0.86, "Low": 0.84, "Close": 0.855},
        ]
    )
    out = client.fx.convert_common_quote_currency(prices, quote_currency="USD")
    tickers = set(out["Ticker"])
    assert tickers == {"JPYUSD", "EURUSD"}
    jpyusd = out.loc[out["Ticker"] == "JPYUSD"].iloc[0]
    assert jpyusd["Close"] == pytest.approx(1 / 110.5)
    # High/Low are swapped on inversion.
    assert jpyusd["High"] == pytest.approx(1 / 109.0)
    assert jpyusd["Low"] == pytest.approx(1 / 111.0)


def test_fx_total_return_index_produces_cumulative_columns(client: Client) -> None:
    prices = pd.DataFrame(
        [
            {"Ticker": "EURUSD", "Date": "2024-01-01", "Close": 1.10},
            {"Ticker": "EURUSD", "Date": "2024-01-02", "Close": 1.11},
            {"Ticker": "EURUSD", "Date": "2024-01-03", "Close": 1.12},
        ]
    )
    rates = pd.DataFrame(
        [
            {"Currency": "EUR", "Date": "2024-01-01", "Rate": 4.0},
            {"Currency": "USD", "Date": "2024-01-01", "Rate": 5.25},
            {"Currency": "EUR", "Date": "2024-01-02", "Rate": 4.0},
            {"Currency": "USD", "Date": "2024-01-02", "Rate": 5.25},
            {"Currency": "EUR", "Date": "2024-01-03", "Rate": 4.0},
            {"Currency": "USD", "Date": "2024-01-03", "Rate": 5.25},
        ]
    )
    out = client.fx.total_return_index(prices, rates)
    assert {
        "Base",
        "Quote",
        "Base_Rate",
        "Quote_Rate",
        "Spot_Returns",
        "Interest_Returns",
        "Spot_Return_Index",
        "Total_Return_Index",
    }.issubset(out.columns)
    # Spot index grows from 1 with the realised returns.
    assert out["Spot_Return_Index"].iloc[-1] > 1.0


# ---- fx not-seeded-yet stubs ---------------------------------------------


def test_fx_get_asset_list_raises_dataset_not_seeded(client: Client) -> None:
    from rwpytools.errors import DatasetNotSeededError

    with pytest.raises(DatasetNotSeededError):
        client.fx.get_asset_list("Major")


# ---- fx footgun guard -----------------------------------------------------


def test_get_daily_ohlc_rejects_bare_string(client: Client) -> None:
    """``get_daily_ohlc("EURUSD")`` is a footgun — must raise TypeError."""

    with pytest.raises(TypeError):
        client.fx.get_daily_ohlc("EURUSD")  # type: ignore[arg-type]


# ---- async pod parity -----------------------------------------------------


def test_async_client_exposes_pod_accessors() -> None:
    """AsyncClient must expose the same four pod accessors as Client."""

    import rwpytools
    from rwpytools.pods import (
        AsyncCryptoPod,
        AsyncEquityFactorsPod,
        AsyncFxPod,
        AsyncMacroPod,
    )

    assert hasattr(rwpytools, "AsyncCryptoPod")
    assert AsyncCryptoPod.__name__ == "AsyncCryptoPod"
    assert AsyncEquityFactorsPod.__name__ == "AsyncEquityFactorsPod"
    assert AsyncFxPod.__name__ == "AsyncFxPod"
    assert AsyncMacroPod.__name__ == "AsyncMacroPod"


def test_sync_and_async_pods_have_same_methods() -> None:
    """Every public method on the async pod must have a sync mirror."""

    from rwpytools.pods import (
        AsyncCryptoPod,
        AsyncEquityFactorsPod,
        AsyncFxPod,
        AsyncMacroPod,
        CryptoPod,
        EquityFactorsPod,
        FxPod,
        MacroPod,
    )

    pairs = [
        (AsyncCryptoPod, CryptoPod),
        (AsyncFxPod, FxPod),
        (AsyncMacroPod, MacroPod),
        (AsyncEquityFactorsPod, EquityFactorsPod),
    ]
    for async_cls, sync_cls in pairs:
        async_methods = {
            n for n in dir(async_cls) if not n.startswith("_") and callable(getattr(async_cls, n))
        }
        sync_methods = {
            n for n in dir(sync_cls) if not n.startswith("_") and callable(getattr(sync_cls, n))
        }
        missing = async_methods - sync_methods
        assert not missing, (
            f"{sync_cls.__name__} is missing sync mirrors for "
            f"{sorted(missing)} (present on {async_cls.__name__})"
        )
