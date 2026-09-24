"""File-format dispatch and parsing."""

from __future__ import annotations

import datetime as dt
import gzip
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.feather
import pyarrow.parquet
import pytest

import rwpytools.formats as formats
from rwpytools.errors import RwPyToolsError
from rwpytools.formats import (
    DataFormat,
    detect_format,
    max_timestamp,
    read,
    read_auto,
    read_auto_with_hints,
    read_csv_chunks,
    read_frame,
    read_head,
    register_dtype_hints,
)


def test_detect_format_feather() -> None:
    assert detect_format("R1000_ohlc_1d.feather") == "feather"


def test_detect_format_csv() -> None:
    assert detect_format("coinmetrics.csv") == "csv"


def test_detect_format_csv_gz() -> None:
    assert detect_format("ib_shortstock_2023.csv.gzip") == "csv.gz"
    assert detect_format("foo.csv.gz") == "csv.gz"


def test_detect_format_parquet() -> None:
    assert detect_format("sharadar/ticker_metadata.parquet") == "parquet"
    assert detect_format("x.parquet") is DataFormat.PARQUET


def test_detect_format_unknown_raises() -> None:
    with pytest.raises(RwPyToolsError):
        detect_format("file.xlsx")


def test_dataformat_is_str_backcompat() -> None:
    # str-subclass enum: string call-sites and equality keep working.
    assert DataFormat.FEATHER == "feather"
    assert read.__module__  # smoke: importable
    assert DataFormat("csv.gz") is DataFormat.CSV_GZ


def test_read_csv(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    df = read(p, "csv")
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_read_csv_gz(tmp_path: Path) -> None:
    p = tmp_path / "x.csv.gzip"
    with gzip.open(p, "wb") as fh:
        fh.write(b"a,b\n1,2\n")
    df = read_auto(p, object_name="x.csv.gzip")
    assert df.shape == (1, 2)


def test_read_feather(tmp_path: Path) -> None:
    p = tmp_path / "x.feather"
    pyarrow.feather.write_feather(pa.table({"a": [1, 2], "b": [3, 4]}), p)
    df = read(p, "feather")
    assert df.shape == (2, 2)


def test_read_parquet(tmp_path: Path) -> None:
    p = tmp_path / "x.parquet"
    pyarrow.parquet.write_table(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), p)
    df = read(p, "parquet")
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (3, 2)
    # also via suffix auto-detection
    assert read_auto(p, object_name="x.parquet").shape == (3, 2)


def test_arrow_reader_falls_back_when_metadata_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arrow files from BigQuery name db-dtypes extension types (e.g. ``dbdate``)
    in pandas metadata; without db-dtypes the metadata-honouring conversion
    raises TypeError. The reader must fall back to a metadata-free read.
    """

    p = tmp_path / "sectors.feather"
    table = pa.table(
        {
            "ticker": ["AAA", "BBB"],
            "listed": pa.array([dt.date(2020, 1, 1), dt.date(2021, 6, 2)], type=pa.date32()),
        }
    )
    pyarrow.feather.write_feather(table, p)

    # Simulate the db-dtypes failure: the metadata-honouring path raises,
    # the metadata-free path (ignore_metadata=True) succeeds.
    real_to_pandas = formats._to_pandas

    def flaky(tbl: object, *, ignore_metadata: bool = False):  # type: ignore[no-untyped-def]
        if not ignore_metadata:
            raise TypeError("data type 'dbdate' not understood")
        return real_to_pandas(tbl, ignore_metadata=True)

    monkeypatch.setattr(formats, "_to_pandas", flaky)

    df = read(p, "feather")
    assert list(df.columns) == ["ticker", "listed"]
    assert df.shape == (2, 2)
    # date column came back as a usable datetime, not an exotic extension dtype
    assert str(df["listed"].dtype).startswith("datetime64")


def test_max_bytes_guard_blocks_oversized_file(tmp_path: Path) -> None:
    p = tmp_path / "huge.csv"
    p.write_bytes(b"a,b\n" + b"1,2\n" * 1000)
    with pytest.raises(RwPyToolsError):
        read(p, "csv", max_bytes=20)


def test_dtype_hints_are_applied_postload(tmp_path: Path) -> None:
    """Registered dtype hints coerce existing columns, ignore new ones."""

    p = tmp_path / "macro_vix_ohlcv.csv"
    p.write_text("date,close\n2024-01-02,18\n2024-01-03,19\n")
    register_dtype_hints(
        namespace="macro",
        dataset="vix",
        schema="ohlcv-1d",
        dtypes={"close": "float64", "future_column": "object"},
    )
    df = read_auto_with_hints(
        p,
        object_name="vix_vix3m.csv",
        namespace="macro",
        dataset="vix",
        schema="ohlcv-1d",
    )
    assert str(df["close"].dtype) == "float64"
    # unknown ``future_column`` did not crash the load
    assert "future_column" not in df.columns


def test_read_auto_with_hints_unregistered_cell_passes_through(
    tmp_path: Path,
) -> None:
    """No registered hints → behaves exactly like read_auto."""

    p = tmp_path / "unmapped.csv"
    p.write_text("a,b\n1,2\n")
    df = read_auto_with_hints(
        p,
        object_name="unmapped.csv",
        namespace="crypto",
        dataset="unmapped",
        schema="snapshot",
    )
    assert df.shape == (1, 2)


def test_read_csv_chunks_yields_smaller_frames(tmp_path: Path) -> None:
    p = tmp_path / "rows.csv"
    p.write_text("a,b\n" + "\n".join(f"{i},{i * 2}" for i in range(10)) + "\n")
    chunks = list(read_csv_chunks(p, chunksize=4))
    assert len(chunks) == 3  # 4 + 4 + 2 rows
    assert sum(len(c) for c in chunks) == 10


# ---- filtered reads, heads and watermarks ------------------------------------------------


def _write_hourly(path: Path, *, tz: str | None = None) -> None:
    stamps = pd.to_datetime(
        ["2024-01-01 23:00", "2024-01-02 00:00", "2024-01-02 23:00", "2024-01-03 00:00"]
    ).as_unit("us")
    if tz:
        stamps = stamps.tz_localize(tz)
    table = pa.table(
        {
            "Ticker": ["BTC", "ETH", "BTC", "BTC"],
            "Datetime": pa.array(stamps.to_pydatetime(), pa.timestamp("us", tz=tz)),
            "Close": [1.0, 2.0, 3.0, 4.0],
        }
    )
    pyarrow.feather.write_feather(table, path)


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_read_frame_filters_timestamps_in_arrow(tmp_path: Path, tz: str | None) -> None:
    path = tmp_path / "x.feather"
    _write_hourly(path, tz=tz)
    out = read_frame(
        path,
        object_name="x.feather",
        date_column="Datetime",
        start=dt.date(2024, 1, 2),
        end=dt.date(2024, 1, 2),
        symbol_column="Ticker",
        symbols=["BTC"],
    )
    # end is inclusive of the whole day
    assert out["Close"].tolist() == [3.0]


def test_read_frame_filters_date32_and_strings(tmp_path: Path) -> None:
    path = tmp_path / "d.feather"
    table = pa.table(
        {
            "date": pa.array([dt.date(2024, 1, d) for d in (1, 2, 3)], pa.date32()),
            "iso": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    pyarrow.feather.write_feather(table, path)
    by_date = read_frame(
        path, object_name="d.feather", date_column="date", start=dt.date(2024, 1, 2)
    )
    by_text = read_frame(path, object_name="d.feather", date_column="iso", end=dt.date(2024, 1, 2))
    assert by_date["iso"].tolist() == ["2024-01-02", "2024-01-03"]
    assert by_text["iso"].tolist() == ["2024-01-01", "2024-01-02"]


def test_read_frame_filters_csv(tmp_path: Path) -> None:
    path = tmp_path / "v.csv"
    path.write_text("ticker,date,close\nVIX,2024-01-01,1\nVIX3M,2024-01-02,2\nVIX,2024-01-02,3\n")
    out = read_frame(
        path,
        object_name="v.csv",
        date_column="date",
        start=dt.date(2024, 1, 2),
        symbol_column="ticker",
        symbols=["VIX"],
    )
    assert out["close"].tolist() == [3]
    assert out["date"].tolist() == ["2024-01-02"]  # representation untouched


def test_read_frame_without_filters_is_read_auto(tmp_path: Path) -> None:
    path = tmp_path / "v.csv"
    path.write_text("a\n1\n2\n")
    assert len(read_frame(path, object_name="v.csv")) == 2


def test_max_timestamp_reads_one_column(tmp_path: Path) -> None:
    feather_path = tmp_path / "x.feather"
    _write_hourly(feather_path, tz="UTC")
    assert max_timestamp(feather_path, object_name="x.feather", column="Datetime") == pd.Timestamp(
        "2024-01-03 00:00"
    )
    csv_path = tmp_path / "v.csv"
    csv_path.write_text("date,x\n2024-01-01,1\n2024-03-05,2\n,3\n")
    assert max_timestamp(csv_path, object_name="v.csv", column="date") == pd.Timestamp("2024-03-05")
    assert max_timestamp(csv_path, object_name="v.csv", column="missing") is None


def test_read_head_samples_every_format(tmp_path: Path) -> None:
    feather_path = tmp_path / "x.feather"
    _write_hourly(feather_path)
    head = read_head(feather_path, object_name="x.feather", rows=2)
    assert list(head.columns) == ["Ticker", "Datetime", "Close"]
    assert len(head) == 2

    parquet_path = tmp_path / "x.parquet"
    pyarrow.parquet.write_table(pyarrow.feather.read_table(feather_path), parquet_path)
    assert len(read_head(parquet_path, object_name="x.parquet", rows=3)) == 3

    csv_path = tmp_path / "v.csv.gz"
    pd.DataFrame({"a": range(10)}).to_csv(csv_path, index=False, compression="gzip")
    assert len(read_head(csv_path, object_name="v.csv.gz", rows=4)) == 4


def test_read_head_handles_feather_v1(tmp_path: Path) -> None:
    """The per-pair FX exports are Feather V1, which is not an IPC file."""

    path = tmp_path / "EURUSD.feather"
    pyarrow.feather.write_feather(
        pa.table({"Date": ["2024-01-01"] * 8, "close": range(8)}), path, version=1
    )
    head = read_head(path, object_name="EURUSD.feather", rows=3)
    assert list(head.columns) == ["Date", "close"]
    assert len(head) == 3
