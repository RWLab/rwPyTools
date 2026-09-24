"""Filesystem cache for downloaded pod objects.

Layout::

    <cache_dir>/
        <bucket>/
            <object_path>           # the file itself, named exactly as in GCS
            <object_path>.meta.json # size, fetched_at, cell, watermark

The meta sidecar records which rw-api cell (``namespace`` / ``dataset`` /
``schema`` / ``symbol``) the object came from and, once read, the export's
watermark (its last date). That lets the client find a cached export — and
decide how much live data to top it up with — **without calling rw-api**:
issuing a signed URL charges the object's full size against the caller's
bandwidth cap, even when the bytes are already on disk.
``object_path`` may contain ``/`` and the cache preserves that structure
(e.g. ``feather/Daily/EURUSD.feather``).

Eviction is LRU-by-fetched_at over total size. Eviction runs at insertion
time only, never as a background task — keeps the model simple and
predictable.

Thread safety
-------------

A single :class:`FilesystemCache` instance is safe for use from many
threads inside the same process; every mutating method acquires the
instance lock. Multi-process callers (e.g. two separate Python
interpreters pointed at the same cache root) are *not* coordinated and
can still race on eviction — use one populator if the cache is shared.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from .errors import CacheError

log = logging.getLogger(__name__)


_META_SUFFIX = ".meta.json"

#: Sidecar JSON format version. Bump when CacheEntryMeta gains/loses
#: *required* fields; old sidecars are then evicted on read instead of
#: crashing the client. Optional fields (``cell``, ``watermark``) are added
#: without a bump, so an upgrade never throws away gigabytes of cache.
CACHE_META_SCHEMA_VERSION = 1

#: ``(namespace, dataset, schema, symbol)`` — the rw-api cell an object serves.
Cell = tuple[str, str, str, str | None]


@dataclass(frozen=True)
class CacheEntryMeta:
    """Sidecar metadata stored alongside each cached object.

    ``schema_version`` carries the on-disk format version. Old entries
    written before versioning was introduced parse as v0 and are
    treated as corrupt (evicted on read).
    """

    bucket: str
    object_name: str
    size: int
    fetched_at: float
    schema_version: int = CACHE_META_SCHEMA_VERSION
    cell: Cell | None = None
    #: Column the watermark was computed over, and its ISO value.
    watermark_column: str | None = None
    watermark: str | None = None

    def to_json(self) -> str:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "bucket": self.bucket,
            "object": self.object_name,
            "size": self.size,
            "fetched_at": self.fetched_at,
        }
        if self.cell is not None:
            namespace, dataset, schema, symbol = self.cell
            payload["cell"] = {
                "namespace": namespace,
                "dataset": dataset,
                "schema": schema,
                "symbol": symbol,
            }
        if self.watermark_column is not None:
            payload["watermark"] = {"column": self.watermark_column, "value": self.watermark}
        return json.dumps(payload, sort_keys=True)

    def age(self, now: float | None = None) -> float:
        """Seconds since the object was fetched (or last re-validated)."""

        return (time.time() if now is None else now) - self.fetched_at

    @classmethod
    def from_json(cls, raw: str) -> CacheEntryMeta:
        try:
            data = json.loads(raw)
            version = int(data.get("schema_version", 0))
            if version != CACHE_META_SCHEMA_VERSION:
                raise CacheError(
                    f"cache meta schema_version mismatch: expected "
                    f"{CACHE_META_SCHEMA_VERSION}, got {version}"
                )
            cell: Cell | None = None
            raw_cell = data.get("cell")
            if isinstance(raw_cell, dict):
                symbol = raw_cell.get("symbol")
                cell = (
                    str(raw_cell["namespace"]),
                    str(raw_cell["dataset"]),
                    str(raw_cell["schema"]),
                    None if symbol is None else str(symbol),
                )
            watermark_column: str | None = None
            watermark: str | None = None
            raw_wm = data.get("watermark")
            if isinstance(raw_wm, dict) and raw_wm.get("column"):
                watermark_column = str(raw_wm["column"])
                value = raw_wm.get("value")
                watermark = None if value is None else str(value)
            return cls(
                bucket=data["bucket"],
                object_name=data["object"],
                size=int(data["size"]),
                fetched_at=float(data["fetched_at"]),
                schema_version=version,
                cell=cell,
                watermark_column=watermark_column,
                watermark=watermark,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise CacheError(f"corrupt cache meta: {exc}") from exc


class FilesystemCache:
    """Thread-safe filesystem-backed cache with LRU-by-fetched_at eviction.

    All public methods acquire ``self._lock``. The lock is non-reentrant
    so internal helpers that need to call into one another use the
    ``_<name>_locked`` variants (caller already holds the lock).
    """

    def __init__(self, root: Path, *, max_bytes: int) -> None:
        self._root = root
        self._max_bytes = max_bytes
        self._lock = threading.RLock()
        # Lazily-initialised running total of file bytes. Refreshed from
        # disk on first access; thereafter updated incrementally by
        # store_meta / evict / reserve_and_commit. ``None`` means "not yet
        # computed".
        self._total_bytes_cache: int | None = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    # ---- path helpers ------------------------------------------------------

    def path_for(self, bucket: str, object_name: str) -> Path:
        # Defensive: prevent path traversal. ``object_name`` comes from the
        # API, which validates it server-side, but check again here.
        if ".." in object_name.split("/"):
            raise CacheError(f"object name must not contain '..': {object_name!r}")
        return self._root / bucket / object_name

    def meta_path_for(self, bucket: str, object_name: str) -> Path:
        obj = self.path_for(bucket, object_name)
        return obj.with_name(obj.name + _META_SUFFIX)

    # ---- read paths --------------------------------------------------------

    def lookup(self, bucket: str, object_name: str) -> CacheEntryMeta | None:
        """Return cached meta if both the object file and its sidecar exist.

        Corrupt sidecars (bad JSON, missing fields, schema_version
        mismatch) are evicted in place so the caller observes a normal
        cache miss.
        """

        with self._lock:
            obj = self.path_for(bucket, object_name)
            meta = self.meta_path_for(bucket, object_name)
            if not obj.is_file() or not meta.is_file():
                return None
            try:
                return CacheEntryMeta.from_json(meta.read_text())
            except CacheError:
                log.warning("Discarding cache entry with corrupt meta: %s", obj)
                self._evict_locked(bucket, object_name)
                return None

    def lookup_cell(self, cell: Cell) -> CacheEntryMeta | None:
        """Find the cached export for an rw-api cell, without any network call.

        Returns ``None`` when no sidecar names ``cell`` or its object file
        is gone. When several objects claim the cell (the server renamed
        the export) the most recently fetched wins.
        """

        with self._lock:
            best: CacheEntryMeta | None = None
            for entry in self._enumerate_locked():
                if entry.cell != cell:
                    continue
                if not self.path_for(entry.bucket, entry.object_name).is_file():
                    continue
                if best is None or entry.fetched_at > best.fetched_at:
                    best = entry
            return best

    def entries(self) -> list[CacheEntryMeta]:
        """Every cached object's metadata, oldest fetch first."""

        with self._lock:
            return sorted(self._enumerate_locked(), key=lambda e: e.fetched_at)

    def clear(self) -> int:
        """Evict every cached object. Returns the number of objects removed."""

        with self._lock:
            entries = self._enumerate_locked()
            for entry in entries:
                self._evict_locked(entry.bucket, entry.object_name)
            self._total_bytes_cache = None
            return len(entries)

    def update_meta(
        self,
        meta: CacheEntryMeta,
        *,
        fetched_at: float | None = None,
        cell: Cell | None = None,
        watermark_column: str | None = None,
        watermark: str | None = None,
    ) -> CacheEntryMeta:
        """Rewrite an existing sidecar with the given fields replaced (atomically).

        Used to record a computed watermark and to bump ``fetched_at`` when
        rw-api confirms a cached object is still current. Fields left as
        ``None`` keep their value. The object file and the size accounting
        are untouched.
        """

        updated = replace(
            meta,
            fetched_at=meta.fetched_at if fetched_at is None else fetched_at,
            cell=meta.cell if cell is None else cell,
            watermark_column=(
                meta.watermark_column if watermark_column is None else watermark_column
            ),
            watermark=meta.watermark if watermark_column is None else watermark,
        )
        meta_path = self.meta_path_for(meta.bucket, meta.object_name)
        with self._lock:
            tmp = meta_path.with_name(meta_path.name + ".tmp")
            tmp.write_text(updated.to_json())
            os.replace(tmp, meta_path)
        return updated

    def total_bytes(self) -> int:
        """Return total object-file bytes under management (excluding sidecars)."""

        with self._lock:
            if self._total_bytes_cache is None:
                self._total_bytes_cache = self._compute_total_bytes_locked()
            return self._total_bytes_cache

    # ---- write paths -------------------------------------------------------

    def store_meta(
        self,
        *,
        bucket: str,
        object_name: str,
        size: int,
        fetched_at: float | None = None,
        cell: Cell | None = None,
    ) -> CacheEntryMeta:
        """Write the sidecar after the object file is in place.

        Sidecar is written via a tempfile + atomic rename, so a crash
        mid-write can never produce a half-written JSON file.
        """

        meta = CacheEntryMeta(
            bucket=bucket,
            object_name=object_name,
            size=size,
            fetched_at=fetched_at if fetched_at is not None else time.time(),
            cell=cell,
        )
        meta_path = self.meta_path_for(bucket, object_name)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(meta.to_json())
        os.replace(tmp, meta_path)
        with self._lock:
            # Bumping the in-memory counter is cheaper than rescanning the
            # tree; ``_total_bytes_cache`` is None on first call which
            # forces a recompute the next time someone asks.
            if self._total_bytes_cache is not None:
                self._total_bytes_cache += size
        return meta

    def evict(self, bucket: str, object_name: str) -> None:
        """Public eviction wrapper; safe to call when not holding the lock."""

        with self._lock:
            self._evict_locked(bucket, object_name)

    def _evict_locked(self, bucket: str, object_name: str) -> None:
        obj_path = self.path_for(bucket, object_name)
        meta_path = self.meta_path_for(bucket, object_name)
        # Capture the on-disk size *before* unlinking so the in-memory
        # counter stays consistent even if the file shrinks between
        # store and evict.
        try:
            size_to_subtract = obj_path.stat().st_size
        except OSError:
            size_to_subtract = 0
        for p in (obj_path, meta_path):
            try:
                p.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Failed to evict cache entry %s: %s", p, exc)
        if self._total_bytes_cache is not None:
            self._total_bytes_cache = max(0, self._total_bytes_cache - size_to_subtract)

    # ---- atomic reservation -----------------------------------------------

    @contextlib.contextmanager
    def reserve(
        self, *, bucket: str, object_name: str, size: int, cell: Cell | None = None
    ) -> Iterator[Path]:
        """Reserve cache space for an incoming download.

        Yields the *temp* path the caller should write to. On normal
        exit the temp file is atomically renamed to the destination and
        the sidecar is written; on exception the temp file is removed
        and no sidecar is left behind. The cache lock is *not* held
        across the yield (downloads can take minutes); concurrent
        callers serialise only on the eviction + sidecar commit phases.

        Use it like::

            with cache.reserve(bucket=b, object_name=o, size=n) as tmp:
                http.stream(url, tmp)
            # sidecar already committed here

        On exception inside the ``with`` block the partial temp file is
        removed before the exception propagates.
        """

        with self._lock:
            self._make_room_for_locked(size)

        target = self.path_for(bucket, object_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        # If a previous run crashed mid-download, scrub the stale .part
        # so we don't append to it or rename it without overwriting.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)

        success = False
        try:
            yield tmp
            # Replacing an existing object (a refresh): take its bytes out of
            # the running total before store_meta adds the new size, or the
            # budget would count the object twice.
            try:
                replaced_size = target.stat().st_size
            except OSError:
                replaced_size = 0
            os.replace(tmp, target)
            if replaced_size:
                with self._lock:
                    if self._total_bytes_cache is not None:
                        self._total_bytes_cache = max(0, self._total_bytes_cache - replaced_size)
            try:
                self.store_meta(bucket=bucket, object_name=object_name, size=size, cell=cell)
                success = True
            except OSError as exc:
                # Sidecar write failed: object file would be invisible on
                # next lookup() (no sidecar -> miss). Evict the orphan so
                # callers don't see "partial" cache state.
                with contextlib.suppress(OSError):
                    target.unlink(missing_ok=True)
                raise CacheError(
                    f"failed to write cache sidecar for {bucket}/{object_name}"
                ) from exc
        finally:
            if not success:
                # Clean up either the .part (download failure) or after
                # we've already moved tmp to target and the sidecar
                # failed (the target has just been removed above).
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)

    # ---- eviction ----------------------------------------------------------

    def make_room_for(self, incoming_bytes: int) -> None:
        """Evict LRU-by-fetched_at entries until ``current+incoming <= max_bytes``."""

        with self._lock:
            self._make_room_for_locked(incoming_bytes)

    def _make_room_for_locked(self, incoming_bytes: int) -> None:
        if self._max_bytes == 0:
            return
        current = (
            self._total_bytes_cache
            if self._total_bytes_cache is not None
            else self._compute_total_bytes_locked()
        )
        self._total_bytes_cache = current
        target = self._max_bytes - incoming_bytes
        if current <= target:
            return

        entries = self._enumerate_locked()
        entries.sort(key=lambda e: e.fetched_at)

        for entry in entries:
            if current <= target:
                break
            obj_path = self.path_for(entry.bucket, entry.object_name)
            try:
                size = obj_path.stat().st_size
            except OSError:
                size = entry.size
            self._evict_locked(entry.bucket, entry.object_name)
            current -= size
        self._total_bytes_cache = max(0, current)

    # ---- internals ---------------------------------------------------------

    def _compute_total_bytes_locked(self) -> int:
        total = 0
        if not self._root.is_dir():
            return 0
        for p in self._root.rglob("*"):
            if p.is_file() and not p.name.endswith(_META_SUFFIX):
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
        return total

    def _enumerate_locked(self) -> list[CacheEntryMeta]:
        out: list[CacheEntryMeta] = []
        if not self._root.is_dir():
            return out
        for meta_file in self._root.rglob(f"*{_META_SUFFIX}"):
            try:
                out.append(CacheEntryMeta.from_json(meta_file.read_text()))
            except (OSError, CacheError):
                continue
        return out


__all__ = ["CACHE_META_SCHEMA_VERSION", "CacheEntryMeta", "Cell", "FilesystemCache"]
