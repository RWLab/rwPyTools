"""File format dispatch — read cached pod files into pandas DataFrames.

Each format rw-api serves is modelled as a :class:`DataFormat` enum member
with a single registry entry (:data:`_FORMAT_SPECS`) bundling its filename
suffixes and its reader. Adding a new format = one entry there; both
:func:`detect_format` and :func:`read` are derived from it, so the suffix
detection and the reader can never drift apart.

Supported formats:

* ``feather``  — Arrow IPC (``.feather``).
* ``parquet``  — Apache Parquet (``.parquet``).
* ``csv``      — plain UTF-8 text (``.csv``).
* ``csv.gz``   — gzip-compressed CSV (``.csv.gz`` / ``.csv.gzip``).

Don't put pod-specific cleaning here — that belongs in the pod method that
called us.

Arrow extension dtypes
----------------------

Arrow files produced from BigQuery (`.to_dataframe()`) can carry pandas
metadata naming the ``db-dtypes`` extension types (``dbdate`` / ``dbtime``).
Replaying that metadata on read requires ``db-dtypes`` to be installed; we
don't want that dependency, so :func:`_arrow_to_df` falls back to reading
without the metadata (Arrow date/time logical types → ``datetime64``) when
the metadata-honouring conversion fails.

Dtype hints
-----------

:func:`register_dtype_hints` lets a pod opt in to typed dtype hints for a
``(namespace, dataset, schema)`` triple, applied post-load.

Memory guard
------------

Pass ``max_bytes`` to :func:`read` / :func:`read_auto` to bound the
worst-case in-memory footprint; oversized files raise before any read.

Filtered reads
--------------

:func:`read_frame` narrows by date range and symbol while reading. For the
Arrow formats the predicate is pushed into a batch-by-batch dataset scan,
so ``start="2024-01-01", symbols=["BTCUSDT"]`` against a 20M-row export
never holds more than the matching rows plus one record batch — the whole
export is never materialised. :func:`max_timestamp` finds an export's
watermark by reading just its date column. After each read, Arrow's
allocator is asked to return freed memory to the OS
(:func:`release_arrow_memory`), so long-running processes do not ratchet up.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import enum
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather
import pyarrow.parquet

from .errors import RwPyToolsError

log = logging.getLogger(__name__)


class DataFormat(str, enum.Enum):
    """On-disk formats rw-api serves for bulk pod files.

    Subclasses ``str`` so existing string call-sites keep working:
    ``read(path, "csv")`` and ``detect_format(x) == "feather"`` both
    behave as before.
    """

    FEATHER = "feather"
    PARQUET = "parquet"
    CSV = "csv"
    CSV_GZ = "csv.gz"


#: ``compression=`` values this module passes to :func:`pandas.read_csv`.
CsvCompression = Literal["gzip"] | None

#: Triple of (namespace, dataset, schema). Used as the key in the dtype
#: registry. Pods register their own hints in their module-level code.
CellKey = tuple[str, str, str]


_DTYPE_REGISTRY: dict[CellKey, Mapping[str, Any]] = {}


def register_dtype_hints(
    *,
    namespace: str,
    dataset: str,
    schema: str,
    dtypes: Mapping[str, Any],
) -> None:
    """Register column → dtype hints for a ``(namespace, dataset, schema)`` cell.

    Hints are applied **post-load** (as ``DataFrame.astype``) so any new
    columns the server adds pass through untouched. Registration is
    idempotent; the latest call wins.
    """

    _DTYPE_REGISTRY[(namespace, dataset, schema)] = dict(dtypes)


def get_dtype_hints(*, namespace: str, dataset: str, schema: str) -> Mapping[str, Any] | None:
    return _DTYPE_REGISTRY.get((namespace, dataset, schema))


# ---------------------------------------------------------------------------
# Arrow -> pandas (with db-dtypes-tolerant fallback)
# ---------------------------------------------------------------------------


def _to_pandas(table: Any, *, ignore_metadata: bool = False) -> pd.DataFrame:
    """Convert a pyarrow Table to a DataFrame.

    Split out as a module function so the metadata-fallback path in
    :func:`_arrow_to_df` is unit-testable.
    """

    frame: pd.DataFrame
    if ignore_metadata:
        frame = table.to_pandas(ignore_metadata=True, date_as_object=False)
    else:
        frame = table.to_pandas()
    return frame


def _arrow_to_df(table: Any) -> pd.DataFrame:
    """Materialise an Arrow table, tolerating unreadable pandas metadata.

    Files written from BigQuery carry pandas metadata naming the
    ``db-dtypes`` extension types (e.g. ``dbdate``). Reconstructing those
    needs ``db-dtypes`` installed; when it isn't, ``to_pandas()`` raises
    ``TypeError: data type 'dbdate' not understood``. We then re-read
    ignoring the metadata, which maps Arrow date/time logical types to
    ``datetime64`` — portable and dependency-free.
    """

    try:
        return _to_pandas(table)
    except TypeError:
        log.debug(
            "arrow->pandas metadata replay failed (likely a db-dtypes "
            "extension type); retrying without pandas metadata"
        )
        return _to_pandas(table, ignore_metadata=True)


def _read_feather(path: Path) -> pd.DataFrame:
    return _arrow_to_df(pyarrow.feather.read_table(path))  # type: ignore[no-untyped-call]


def _read_parquet(path: Path) -> pd.DataFrame:
    return _arrow_to_df(pyarrow.parquet.read_table(path))  # type: ignore[no-untyped-call]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_csv_gz(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")


# ---------------------------------------------------------------------------
# Format registry: one entry per format drives detection + reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FormatSpec:
    fmt: DataFormat
    suffixes: tuple[str, ...]
    reader: Callable[[Path], pd.DataFrame]


# Order matters for suffix detection: compound suffixes (``.csv.gz``) must
# be tried before their tail (``.csv``) so they aren't mis-detected.
_FORMAT_SPECS: tuple[_FormatSpec, ...] = (
    _FormatSpec(DataFormat.FEATHER, (".feather",), _read_feather),
    _FormatSpec(DataFormat.PARQUET, (".parquet",), _read_parquet),
    _FormatSpec(DataFormat.CSV_GZ, (".csv.gzip", ".csv.gz"), _read_csv_gz),
    _FormatSpec(DataFormat.CSV, (".csv",), _read_csv),
)

_BY_FORMAT: dict[DataFormat, _FormatSpec] = {spec.fmt: spec for spec in _FORMAT_SPECS}


def detect_format(object_name: str) -> DataFormat:
    """Infer the format from a GCS object key.

    ``ib_shortstock_2023.csv.gzip`` and ``coinmetrics.csv`` both occur in the
    macro pod, so we match the full suffix chain (compound suffixes first)
    rather than just the last extension.
    """

    lower = object_name.lower()
    for spec in _FORMAT_SPECS:
        if lower.endswith(spec.suffixes):
            return spec.fmt
    raise RwPyToolsError(f"Unknown file format for object {object_name!r}")


def _check_memory_guard(path: Path, *, max_bytes: int | None) -> None:
    if max_bytes is None:
        return
    try:
        size = path.stat().st_size
    except OSError:
        return  # nothing to check against; let the reader fail with its own error
    if size > max_bytes:
        raise RwPyToolsError(
            f"{path.name} is {size:,} bytes which exceeds max_bytes="
            f"{max_bytes:,}; refusing to read into memory."
        )


def read(path: Path, fmt: DataFormat | str, *, max_bytes: int | None = None) -> pd.DataFrame:
    """Read ``path`` into a DataFrame using the named format.

    ``fmt`` accepts a :class:`DataFormat` or its string value. ``max_bytes``
    is an optional upper bound on the file size; reading larger files raises
    :class:`rwpytools.errors.RwPyToolsError` without touching the data.
    """

    try:
        spec = _BY_FORMAT[DataFormat(fmt)]
    except ValueError:
        raise RwPyToolsError(f"Unsupported format {fmt!r}") from None
    _check_memory_guard(path, max_bytes=max_bytes)
    return spec.reader(path)


def read_auto(path: Path, *, object_name: str, max_bytes: int | None = None) -> pd.DataFrame:
    """Read ``path`` choosing the format from ``object_name``'s suffix."""

    return read(path, detect_format(object_name), max_bytes=max_bytes)


def read_auto_with_hints(
    path: Path,
    *,
    object_name: str,
    namespace: str,
    dataset: str,
    schema: str,
    max_bytes: int | None = None,
) -> pd.DataFrame:
    """:func:`read_auto` plus post-load dtype application from the registry.

    Columns named in the registered hint are coerced; unknown columns pass
    through untouched (forwards-compatible). When no hint is registered this
    is exactly equivalent to :func:`read_auto`.
    """

    frame = read_auto(path, object_name=object_name, max_bytes=max_bytes)
    hints = get_dtype_hints(namespace=namespace, dataset=dataset, schema=schema)
    if not hints:
        return frame
    overlap = {col: dtype for col, dtype in hints.items() if col in frame.columns}
    if not overlap:
        return frame
    return frame.astype(overlap)


# ---------------------------------------------------------------------------
# Filtered reads and watermarks
# ---------------------------------------------------------------------------


def to_timestamps(values: pd.Series) -> pd.Series:
    """Normalise a date-like column to naive ``datetime64`` for comparisons.

    Exports disagree on representation — ``datetime64`` (Arrow timestamps),
    ``datetime.date`` objects (Arrow ``date32``) and ISO strings (CSV) all
    occur — and tz-aware columns are converted to naive UTC. Unparseable
    values become ``NaT``.
    """

    if isinstance(values.dtype, pd.DatetimeTZDtype):
        naive: pd.Series = values.dt.tz_convert("UTC").dt.tz_localize(None)
        return naive
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return values
    converted = pd.to_datetime(values, errors="coerce", format="ISO8601", utc=True)
    return converted.dt.tz_localize(None)


def filter_frame(
    frame: pd.DataFrame,
    *,
    date_column: str | None = None,
    start: _dt.date | None = None,
    end: _dt.date | None = None,
    symbol_column: str | None = None,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Narrow a DataFrame to ``[start, end]`` (inclusive days) and ``symbols``.

    A missing column disables that filter rather than raising, so a filter
    never silently empties a frame whose schema moved.
    """

    out = frame
    if symbols and symbol_column and symbol_column in out.columns:
        out = out[out[symbol_column].isin(list(symbols))]
    if (start is not None or end is not None) and date_column and date_column in out.columns:
        stamps = to_timestamps(out[date_column])
        mask = pd.Series(True, index=out.index)
        if start is not None:
            mask &= stamps >= pd.Timestamp(start)
        if end is not None:
            mask &= stamps < pd.Timestamp(end + _dt.timedelta(days=1))
        out = out[mask]
    if out is frame:
        return frame
    return out.reset_index(drop=True)


def _arrow_bound(column_type: Any, day: _dt.date) -> Any:
    """An Arrow scalar comparable with a column of ``column_type`` at ``day``
    00:00, or ``None`` when the type is not one we can compare natively."""

    if pa.types.is_timestamp(column_type):
        stamp = pd.Timestamp(day)
        if column_type.tz is not None:
            stamp = stamp.tz_localize(column_type.tz)
        return pa.scalar(stamp.to_pydatetime(), type=column_type)
    if pa.types.is_date(column_type):
        return pa.scalar(day, type=column_type)
    if pa.types.is_string(column_type) or pa.types.is_large_string(column_type):
        # ISO-8601 text sorts chronologically.
        return pa.scalar(day.isoformat(), type=column_type)
    return None


def _filter_expression(
    schema: Any,
    *,
    date_column: str | None,
    start: _dt.date | None,
    end: _dt.date | None,
    symbol_column: str | None,
    symbols: Sequence[str] | None,
) -> tuple[Any, bool]:
    """Translate the date / symbol filters into an Arrow dataset expression.

    Returns ``(expression, complete)``. ``expression`` is ``None`` when no
    filter applies to this schema. ``complete`` is False when some filter
    could not be expressed (e.g. a dictionary-encoded symbol column or a
    date column in an unusual type); the caller then finishes the job in
    pandas, so the result is always fully filtered.
    """

    complete = True
    expression: Any = None

    def _and(clause: Any) -> None:
        nonlocal expression
        expression = clause if expression is None else (expression & clause)

    names = set(schema.names)
    if symbols and symbol_column and symbol_column in names:
        column_type = schema.field(symbol_column).type
        if pa.types.is_dictionary(column_type):
            complete = False
        else:
            try:
                value_set = pa.array(list(symbols)).cast(column_type)
                _and(pc.field(symbol_column).isin(value_set))
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
                complete = False

    if (start is not None or end is not None) and date_column and date_column in names:
        column_type = schema.field(date_column).type
        low = _arrow_bound(column_type, start) if start is not None else None
        high = _arrow_bound(column_type, end + _dt.timedelta(days=1)) if end is not None else None
        if (start is not None and low is None) or (end is not None and high is None):
            complete = False
        else:
            if low is not None:
                _and(pc.field(date_column) >= low)
            if high is not None:
                _and(pc.field(date_column) < high)

    return expression, complete


def _scan_filtered(
    path: Path,
    fmt: DataFormat,
    *,
    date_column: str | None,
    start: _dt.date | None,
    end: _dt.date | None,
    symbol_column: str | None,
    symbols: Sequence[str] | None,
    memory_pool: Any = None,
) -> tuple[Any, bool] | None:
    """Scan an Arrow-format export with the filter pushed into the scan.

    The file is read record batch by record batch and each batch is
    filtered as it is decoded, so peak memory is the *matching* rows plus
    one batch — never the whole export. That is what keeps a narrow
    ``start`` / ``end`` cheap on a multi-GB file, on the first read and on
    every different window after it.

    Returns ``(table, complete)``, or ``None`` when the file cannot be
    scanned as a dataset (Feather V1, which the per-pair FX exports use, is
    not an Arrow IPC file) and the caller must fall back to a full read.
    """

    try:
        dataset = ds.dataset(str(path), format="ipc" if fmt is DataFormat.FEATHER else "parquet")
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, OSError):
        return None
    expression, complete = _filter_expression(
        dataset.schema,
        date_column=date_column,
        start=start,
        end=end,
        symbol_column=symbol_column,
        symbols=symbols,
    )
    try:
        table = dataset.to_table(filter=expression, memory_pool=memory_pool)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        # An expression the scanner rejects for this file: read it all and
        # let pandas filter instead.
        table = dataset.to_table(memory_pool=memory_pool)
        complete = False
    return table, complete


def release_arrow_memory() -> None:
    """Return Arrow's freed-but-cached memory to the operating system.

    Arrow's allocator (jemalloc / mimalloc) keeps freed memory for reuse, so
    after reading a large export a process's RSS stays near its peak even
    once every table is gone — and a long-running worker that reads export
    after export ratchets upward until it hits its memory limit. Called
    after every read of a cached export.
    """

    with contextlib.suppress(AttributeError, NotImplementedError):
        pa.default_memory_pool().release_unused()


def _read_arrow_table(path: Path, fmt: DataFormat, columns: Sequence[str] | None = None) -> Any:
    cols = list(columns) if columns is not None else None
    if fmt is DataFormat.FEATHER:
        return pyarrow.feather.read_table(path, columns=cols)  # type: ignore[no-untyped-call]
    return pyarrow.parquet.read_table(path, columns=cols)  # type: ignore[no-untyped-call]


def read_frame(
    path: Path,
    *,
    object_name: str,
    date_column: str | None = None,
    start: _dt.date | None = None,
    end: _dt.date | None = None,
    symbol_column: str | None = None,
    symbols: Sequence[str] | None = None,
    max_bytes: int | None = None,
) -> pd.DataFrame:
    """Read a cached export, narrowed to ``[start, end]`` and ``symbols``.

    With no filters this is :func:`read_auto`. ``end`` is inclusive of the
    whole day, so an intraday column keeps every bar on ``end``.

    Feather (Arrow IPC) and parquet exports are scanned with the filter
    pushed down (see :func:`_scan_filtered`), so memory scales with the
    rows returned rather than with the size of the export.
    """

    fmt = detect_format(object_name)
    filtering = bool(symbols) or start is not None or end is not None
    try:
        if not filtering:
            return read(path, fmt, max_bytes=max_bytes)
        _check_memory_guard(path, max_bytes=max_bytes)
        kwargs: dict[str, Any] = {
            "date_column": date_column,
            "start": start,
            "end": end,
            "symbol_column": symbol_column,
            "symbols": symbols,
        }
        if fmt in (DataFormat.FEATHER, DataFormat.PARQUET):
            scanned = _scan_filtered(path, fmt, **kwargs)
            if scanned is None:  # Feather V1: no batch index, read it whole
                table = _read_arrow_table(path, fmt)
                complete = False
            else:
                table, complete = scanned
            frame = _arrow_to_df(table)
            del table
            return frame if complete else filter_frame(frame, **kwargs)
        return filter_frame(_BY_FORMAT[fmt].reader(path), **kwargs)
    finally:
        release_arrow_memory()


def max_timestamp(path: Path, *, object_name: str, column: str) -> pd.Timestamp | None:
    """The latest value of ``column`` in a cached export, as a naive
    ``pandas.Timestamp`` — the export's watermark. ``None`` when the column
    is missing or holds no parseable dates.

    Reads only that one column (Arrow formats) or parses only it (CSV).
    """

    fmt = detect_format(object_name)
    if fmt in (DataFormat.FEATHER, DataFormat.PARQUET):
        try:
            table = _read_arrow_table(path, fmt, columns=[column])
        except (KeyError, pa.ArrowInvalid):
            return None
        series = _arrow_to_df(table)[column]
        del table
        release_arrow_memory()
    else:
        compression: CsvCompression = "gzip" if fmt is DataFormat.CSV_GZ else None
        try:
            series = pd.read_csv(path, usecols=[column], compression=compression)[column]
        except ValueError:  # usecols names a missing column
            return None
    return latest_timestamp(series)


def read_head(path: Path, *, object_name: str, rows: int = 5) -> pd.DataFrame:
    """The first few rows of a cached export — its schema, with sample values.

    Used to conform live rows to an export's column order and dtypes when
    the bulk half of a request is empty. Cheap for every format: Arrow
    files read one record batch, CSVs parse ``rows`` lines.
    """

    fmt = detect_format(object_name)
    if fmt is DataFormat.FEATHER:
        try:
            with pa.memory_map(str(path)) as source:
                reader = pa.ipc.open_file(source)
                if reader.num_record_batches == 0:
                    return _arrow_to_df(reader.schema.empty_table())
                batch = reader.get_batch(0).slice(0, rows)
                return _arrow_to_df(pa.Table.from_batches([batch]))
        except pa.ArrowInvalid:
            # Feather V1 (e.g. the per-pair FX exports) is not an Arrow IPC
            # file; it has no batch index, so read it whole. V1 files are small.
            table = pyarrow.feather.read_table(path)  # type: ignore[no-untyped-call]
            return _arrow_to_df(table.slice(0, rows))
    if fmt is DataFormat.PARQUET:
        parquet = pyarrow.parquet.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=rows):
            return _arrow_to_df(pa.Table.from_batches([batch]))
        return _arrow_to_df(parquet.schema_arrow.empty_table())
    compression: CsvCompression = "gzip" if fmt is DataFormat.CSV_GZ else None
    return pd.read_csv(path, nrows=rows, compression=compression)


def latest_timestamp(values: pd.Series) -> pd.Timestamp | None:
    """Max of a date-like column as a naive Timestamp, or ``None``."""

    stamps = to_timestamps(values)
    latest = stamps.max()
    if latest is None or pd.isna(latest):
        return None
    return pd.Timestamp(latest)


def read_csv_chunks(
    path: Path, *, chunksize: int, compression: CsvCompression = None
) -> Iterator[pd.DataFrame]:
    """Yield chunks of a (possibly gzipped) CSV.

    Use when the file is too large to materialise as a single DataFrame.
    ``compression="gzip"`` is required for ``.csv.gz`` / ``.csv.gzip``
    files; ``None`` for plain CSV.
    """

    chunked: Iterator[pd.DataFrame] = pd.read_csv(
        path, chunksize=chunksize, compression=compression
    )
    return chunked


__all__ = [
    "DataFormat",
    "detect_format",
    "filter_frame",
    "get_dtype_hints",
    "latest_timestamp",
    "max_timestamp",
    "read",
    "read_auto",
    "read_auto_with_hints",
    "read_csv_chunks",
    "read_frame",
    "read_head",
    "register_dtype_hints",
    "release_arrow_memory",
    "to_timestamps",
]
