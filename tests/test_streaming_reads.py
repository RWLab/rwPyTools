"""Filtered reads of cached exports must not load the whole export.

A long-running worker backfilling a large export one window at a time
(e.g. ``liquid_universe_prices``, ~5M rows, year by year) used to read the
*entire* export into memory on every ``get()`` and filter afterwards, and
Arrow's allocator kept the freed memory — RSS ratcheted up until the
worker was OOM-killed. These tests pin the fix: the filter is pushed into
a batch-by-batch scan, results match the old full-read-then-filter path
exactly, and freed Arrow memory is handed back to the OS after every read.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather
import pyarrow.parquet
import pytest

from rwpytools import formats
from rwpytools.errors import RwPyToolsError
from rwpytools.formats import DataFormat, filter_frame, read_auto, read_frame

#: Metered pools must outlive every buffer allocated from them; see the
#: memory-bound test. (The library itself always uses Arrow's process-wide
#: default pool, so it has no such lifetime hazard.)
_METERED_POOLS: list[Any] = []

N_TICKERS = 200
N_DAYS = 2_000  # 400k rows


def _wide_table(*, tz: str | None = None) -> pa.Table:
    """A long export: every ticker on every day, like the equity price exports."""

    rng = np.random.default_rng(0)
    days = pd.date_range("2015-01-01", periods=N_DAYS, freq="D")
    tickers = np.repeat([f"T{i:03d}" for i in range(N_TICKERS)], N_DAYS)
    stamps = pd.DatetimeIndex(np.tile(days.values, N_TICKERS)).as_unit("us")
    if tz:
        stamps = stamps.tz_localize(tz)
    n = N_TICKERS * N_DAYS
    return pa.table(
        {
            "ticker": tickers,
            "date": pa.array(stamps.date, pa.date32()),
            "Datetime": pa.array(stamps.to_pydatetime(), pa.timestamp("us", tz=tz)),
            "iso": pa.array(np.tile(days.strftime("%Y-%m-%d").to_numpy(), N_TICKERS)),
            "close": rng.random(n),
            "volume": rng.random(n),
        }
    )


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same data in every format the exports use, written once."""

    root = tmp_path_factory.mktemp("exports")
    table = _wide_table()
    # pandas' to_feather default: Arrow IPC (Feather V2), lz4, many batches.
    pyarrow.feather.write_feather(table, root / "v2.feather", chunksize=2_000)
    pyarrow.feather.write_feather(
        table, root / "v2_raw.feather", chunksize=2_000, compression="uncompressed"
    )
    pyarrow.feather.write_feather(table.drop_columns(["Datetime"]), root / "v1.feather", version=1)
    pyarrow.parquet.write_table(table, root / "x.parquet", row_group_size=20_000)
    pyarrow.feather.write_feather(_wide_table(tz="UTC"), root / "tz.feather", chunksize=2_000)
    dictionary = table.set_column(0, "ticker", pa.compute.dictionary_encode(table.column("ticker")))
    pyarrow.feather.write_feather(dictionary, root / "dict.feather", chunksize=2_000)
    table.to_pandas().to_csv(root / "x.csv", index=False)
    return root


# ---- the memory property --------------------------------------------------------------


def test_a_narrow_window_never_loads_the_whole_export(export_dir: Path) -> None:
    """The fix itself: peak Arrow memory for a one-year window over a multi-
    batch export is a small fraction of the export, not all of it."""

    path = export_dir / "v2.feather"
    full_bytes = pyarrow.feather.read_table(path).nbytes

    def peak(**filters: Any) -> tuple[int, int]:
        pool = pa.proxy_memory_pool(pa.system_memory_pool())
        # The scanner's background read-ahead can still release buffers to
        # this pool after the scan returns; a pool destroyed first crashes the
        # interpreter later, in some unrelated test. Keep it for the process.
        _METERED_POOLS.append(pool)
        scanned = formats._scan_filtered(
            path,
            DataFormat.FEATHER,
            date_column=filters.get("date_column"),
            start=filters.get("start"),
            end=filters.get("end"),
            symbol_column=None,
            symbols=None,
            memory_pool=pool,
        )
        assert scanned is not None
        table, complete = scanned
        assert complete
        measured = (pool.max_memory(), table.num_rows)
        # The table's buffers belong to ``pool``: free them while the pool
        # is still alive, or local teardown order can crash the interpreter.
        del table, scanned
        return measured

    # Control: an unfiltered scan really does allocate the whole export, so
    # the meter is measuring what we think it is.
    whole_peak, whole_rows = peak()
    assert whole_rows == N_TICKERS * N_DAYS
    assert whole_peak >= 0.9 * full_bytes

    year_peak, year_rows = peak(
        date_column="date", start=dt.date(2016, 1, 1), end=dt.date(2016, 12, 31)
    )
    assert year_rows == N_TICKERS * 366
    assert year_peak < 0.35 * full_bytes, (
        f"one-year window peaked at {year_peak:,} bytes of a {full_bytes:,}-byte export"
    )


def test_filtered_reads_do_not_fall_back_to_a_full_table_read(
    export_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _full_read_forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("filtered read loaded the whole export")

    monkeypatch.setattr(formats, "_read_arrow_table", _full_read_forbidden)
    monkeypatch.setattr(pyarrow.feather, "read_table", _full_read_forbidden)
    monkeypatch.setattr(pyarrow.parquet, "read_table", _full_read_forbidden)

    for name in ("v2.feather", "x.parquet"):
        out = read_frame(
            export_dir / name,
            object_name=name,
            date_column="date",
            start=dt.date(2016, 1, 1),
            end=dt.date(2016, 1, 31),
            symbol_column="ticker",
            symbols=["T007"],
        )
        assert len(out) == 31
        assert set(out["ticker"]) == {"T007"}


def test_arrow_memory_is_released_after_every_read(
    export_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this, freed Arrow memory stays in the allocator and a worker's
    RSS ratchets up read after read."""

    calls: list[str] = []
    monkeypatch.setattr(formats, "release_arrow_memory", lambda: calls.append("released"))

    read_frame(
        export_dir / "v2.feather",
        object_name="v2.feather",
        start=dt.date(2016, 1, 1),
        date_column="date",
    )
    read_frame(export_dir / "v2.feather", object_name="v2.feather")  # unfiltered
    formats.max_timestamp(export_dir / "v2.feather", object_name="v2.feather", column="date")
    assert calls == ["released"] * 3

    calls.clear()
    with pytest.raises(RwPyToolsError, match="max_bytes"):  # released on the error path too
        read_frame(
            export_dir / "v2.feather",
            object_name="v2.feather",
            date_column="date",
            start=dt.date(2016, 1, 1),
            max_bytes=1,
        )
    assert calls == ["released"]


def test_release_arrow_memory_is_safe_to_call() -> None:
    formats.release_arrow_memory()


# ---- results identical to the old full-read-then-filter path ---------------------------

CASES = [
    pytest.param(
        {"date_column": "date", "start": dt.date(2016, 3, 1), "end": dt.date(2016, 3, 31)},
        id="date32-range",
    ),
    pytest.param({"date_column": "Datetime", "start": dt.date(2017, 1, 1)}, id="timestamp-start"),
    pytest.param(
        {"date_column": "Datetime", "end": dt.date(2015, 2, 1)}, id="timestamp-end-inclusive"
    ),
    pytest.param(
        {"date_column": "iso", "start": dt.date(2016, 1, 1), "end": dt.date(2016, 1, 2)},
        id="iso-string",
    ),
    pytest.param({"symbol_column": "ticker", "symbols": ["T001", "T150"]}, id="symbols"),
    pytest.param(
        {
            "date_column": "date",
            "start": dt.date(2016, 6, 1),
            "end": dt.date(2016, 6, 30),
            "symbol_column": "ticker",
            "symbols": ["T042"],
        },
        id="window-and-symbol",
    ),
    pytest.param({"date_column": "date", "start": dt.date(2030, 1, 1)}, id="empty-result"),
]


def _reference(path: Path, object_name: str, **filters: Any) -> pd.DataFrame:
    return filter_frame(read_auto(path, object_name=object_name), **filters)


@pytest.mark.parametrize("filters", CASES)
@pytest.mark.parametrize(
    "name", ["v2.feather", "v2_raw.feather", "x.parquet", "dict.feather", "x.csv"]
)
def test_filtered_read_matches_full_read_then_filter(
    export_dir: Path, name: str, filters: dict[str, Any]
) -> None:
    if name == "x.csv" and filters.get("date_column") == "Datetime":
        pytest.skip("CSV round-trips timestamps as text; covered by the iso case")
    out = read_frame(export_dir / name, object_name=name, **filters)
    expected = _reference(export_dir / name, name, **filters)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), expected.reset_index(drop=True), check_categorical=False
    )


@pytest.mark.parametrize(
    "filters", [c for c in CASES if c.values[0].get("date_column") != "Datetime"]
)
def test_feather_v1_falls_back_and_still_filters(export_dir: Path, filters: Any) -> None:
    """Feather V1 (the per-pair FX exports) is not an IPC file, so it cannot
    be scanned; the fallback must still return exactly the filtered rows."""

    assert (
        formats._scan_filtered(
            export_dir / "v1.feather",
            DataFormat.FEATHER,
            date_column=None,
            start=None,
            end=None,
            symbol_column=None,
            symbols=None,
        )
        is None
    )
    out = read_frame(export_dir / "v1.feather", object_name="v1.feather", **filters)
    expected = _reference(export_dir / "v1.feather", "v1.feather", **filters)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


def test_timezone_aware_timestamps_filter_by_utc_day(export_dir: Path) -> None:
    filters = {"date_column": "Datetime", "start": dt.date(2016, 1, 1), "end": dt.date(2016, 1, 3)}
    out = read_frame(export_dir / "tz.feather", object_name="tz.feather", **filters)
    expected = _reference(export_dir / "tz.feather", "tz.feather", **filters)
    assert len(out) == N_TICKERS * 3
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


def test_a_year_by_year_backfill_returns_every_row_exactly_once(export_dir: Path) -> None:
    """The workload that OOM'd: iterate the export one calendar year at a time."""

    path = export_dir / "v2.feather"
    years = range(2015, 2021)
    parts = [
        read_frame(
            path,
            object_name="v2.feather",
            date_column="date",
            start=dt.date(y, 1, 1),
            end=dt.date(y, 12, 31),
        )
        for y in years
    ]
    combined = pd.concat(parts, ignore_index=True)
    assert len(combined) == N_TICKERS * N_DAYS
    assert not combined.duplicated(["ticker", "date"]).any()
