"""Filesystem cache: lookup, eviction, sidecar handling."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from rwpytools.cache import (
    CACHE_META_SCHEMA_VERSION,
    CacheEntryMeta,
    FilesystemCache,
)
from rwpytools.errors import CacheError


def test_lookup_miss_when_cache_empty(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=1024)
    assert cache.lookup("bucket", "obj.csv") is None


def test_store_and_lookup_round_trip(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=1024)
    obj_path = cache.path_for("bucket", "obj.csv")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_bytes(b"hello")
    cache.store_meta(bucket="bucket", object_name="obj.csv", size=5)

    meta = cache.lookup("bucket", "obj.csv")
    assert meta is not None
    assert meta.size == 5
    assert meta.bucket == "bucket"
    assert meta.object_name == "obj.csv"


def test_corrupt_meta_is_evicted(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=1024)
    obj_path = cache.path_for("b", "o")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_bytes(b"x")
    meta_path = cache.meta_path_for("b", "o")
    meta_path.write_text("not-json")

    assert cache.lookup("b", "o") is None
    assert not meta_path.exists()
    assert not obj_path.exists()


def test_path_for_rejects_traversal(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=1024)
    with pytest.raises(CacheError):
        cache.path_for("b", "../../etc/passwd")


def test_total_bytes_excludes_meta_files(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=1024)
    obj_path = cache.path_for("b", "o")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_bytes(b"a" * 100)
    cache.store_meta(bucket="b", object_name="o", size=100)
    assert cache.total_bytes() == 100


def test_make_room_evicts_lru_when_over_budget(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=200)

    for i, age in enumerate([100.0, 200.0, 300.0]):
        obj = cache.path_for("b", f"o{i}.csv")
        obj.parent.mkdir(parents=True, exist_ok=True)
        obj.write_bytes(b"x" * 80)
        cache.store_meta(bucket="b", object_name=f"o{i}.csv", size=80, fetched_at=age)

    # Total = 240. Adding 80 more bytes (target=200, so we need <=120 cached).
    cache.make_room_for(80)
    remaining = {p.name for p in (tmp_path / "b").iterdir() if not p.name.endswith(".meta.json")}
    # The two oldest (o0, o1) should be gone; o2 (newest at fetched_at=300) survives.
    assert remaining == {"o2.csv"}


def test_make_room_noop_when_max_bytes_zero(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=0)
    obj = cache.path_for("b", "huge")
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(b"x" * 1024)
    cache.store_meta(bucket="b", object_name="huge", size=1024)
    cache.make_room_for(99999)
    assert obj.exists()


def test_meta_round_trips_through_json() -> None:
    meta = CacheEntryMeta(bucket="b", object_name="o", size=10, fetched_at=time.time())
    parsed = CacheEntryMeta.from_json(meta.to_json())
    assert parsed == meta


def test_meta_includes_schema_version() -> None:
    meta = CacheEntryMeta(bucket="b", object_name="o", size=1, fetched_at=1.0)
    decoded = json.loads(meta.to_json())
    assert decoded["schema_version"] == CACHE_META_SCHEMA_VERSION


def test_meta_rejects_unknown_schema_version() -> None:
    rogue = json.dumps(
        {
            "schema_version": CACHE_META_SCHEMA_VERSION + 99,
            "bucket": "b",
            "object": "o",
            "size": 1,
            "fetched_at": 1.0,
        }
    )
    with pytest.raises(CacheError):
        CacheEntryMeta.from_json(rogue)


def test_legacy_sidecar_treated_as_corrupt(tmp_path: Path) -> None:
    """Old sidecars (no schema_version field) parse as v0 and are evicted."""

    cache = FilesystemCache(tmp_path, max_bytes=1024)
    obj = cache.path_for("b", "o")
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(b"x")
    legacy = json.dumps({"bucket": "b", "object": "o", "size": 1, "fetched_at": 1.0})
    cache.meta_path_for("b", "o").write_text(legacy)

    assert cache.lookup("b", "o") is None
    assert not obj.exists()


def test_reserve_commits_atomically(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=1024)
    with cache.reserve(bucket="b", object_name="o.csv", size=5) as tmp:
        assert tmp.name.endswith(".part")
        tmp.write_bytes(b"hello")

    obj = cache.path_for("b", "o.csv")
    assert obj.read_bytes() == b"hello"
    meta = cache.lookup("b", "o.csv")
    assert meta is not None
    assert meta.size == 5


def test_reserve_cleans_up_on_exception(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=1024)

    def _trigger() -> None:
        with cache.reserve(bucket="b", object_name="o.csv", size=5) as tmp:
            tmp.write_bytes(b"partial")
            raise RuntimeError("download failed")

    with pytest.raises(RuntimeError):
        _trigger()

    obj = cache.path_for("b", "o.csv")
    meta = cache.meta_path_for("b", "o.csv")
    assert not obj.exists(), "object file must not be committed on failure"
    assert not meta.exists(), "sidecar must not exist for an aborted download"
    # And no .part file should linger.
    assert list((tmp_path / "b").iterdir()) == []


def test_reserve_scrubs_stale_part_file(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=1024)
    obj = cache.path_for("b", "o.csv")
    obj.parent.mkdir(parents=True, exist_ok=True)
    (obj.parent / "o.csv.part").write_bytes(b"leftover-from-crash")

    with cache.reserve(bucket="b", object_name="o.csv", size=5) as tmp:
        # Stale .part must have been removed before our handle is yielded;
        # we overwrite cleanly, not append.
        assert not tmp.exists() or tmp.read_bytes() == b""
        tmp.write_bytes(b"fresh")

    assert obj.read_bytes() == b"fresh"


def test_evict_updates_total_bytes(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=10_000)
    obj = cache.path_for("b", "o")
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(b"x" * 100)
    cache.store_meta(bucket="b", object_name="o", size=100)
    assert cache.total_bytes() == 100

    cache.evict("b", "o")
    assert cache.total_bytes() == 0


def test_total_bytes_cached_between_calls(tmp_path: Path) -> None:
    """A second call doesn't re-walk the tree; the answer is incremental."""

    cache = FilesystemCache(tmp_path, max_bytes=10_000)
    obj = cache.path_for("b", "o")
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(b"x" * 50)
    cache.store_meta(bucket="b", object_name="o", size=50)

    first = cache.total_bytes()
    # Mutate disk *outside* the cache API. If total_bytes() rescanned
    # the tree every call it would notice; the in-memory counter
    # should keep reporting the API-tracked value.
    obj.write_bytes(b"y" * 5_000)
    second = cache.total_bytes()
    assert first == second == 50


def test_concurrent_make_room_is_safe(tmp_path: Path) -> None:
    """Two threads racing on make_room_for don't double-evict."""

    cache = FilesystemCache(tmp_path, max_bytes=500)
    for i in range(5):
        p = cache.path_for("b", f"o{i}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"a" * 100)
        cache.store_meta(bucket="b", object_name=f"o{i}", size=100, fetched_at=float(i))

    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            cache.make_room_for(200)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert cache.total_bytes() <= 500


# ---- cell index, watermark, and maintenance ----------------------------------------


def _put(
    cache: FilesystemCache, obj: str, payload: bytes, cell: tuple[str, str, str, str | None] | None
) -> None:
    with cache.reserve(bucket="b", object_name=obj, size=len(payload), cell=cell) as tmp:
        tmp.write_bytes(payload)


def test_lookup_cell_finds_a_cached_export_without_its_object_name(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=0)
    _put(cache, "fx/EURUSD.feather", b"eur", ("fx", "pairs", "ohlcv-1d", "EURUSD"))
    _put(cache, "fx/GBPUSD.feather", b"gbp", ("fx", "pairs", "ohlcv-1d", "GBPUSD"))

    hit = cache.lookup_cell(("fx", "pairs", "ohlcv-1d", "EURUSD"))
    assert hit is not None and hit.object_name == "fx/EURUSD.feather"
    assert cache.lookup_cell(("fx", "pairs", "ohlcv-1d", "USDJPY")) is None

    # a sidecar whose object vanished is not a hit
    cache.path_for("b", "fx/EURUSD.feather").unlink()
    assert cache.lookup_cell(("fx", "pairs", "ohlcv-1d", "EURUSD")) is None


def test_sidecars_without_cell_or_watermark_still_load(tmp_path: Path) -> None:
    """Sidecars written before the optional fields existed must not be
    evicted — that would throw away multi-GB exports on upgrade."""

    cache = FilesystemCache(tmp_path, max_bytes=0)
    _put(cache, "x.csv", b"abc", None)
    meta_path = cache.meta_path_for("b", "x.csv")
    meta_path.write_text(
        '{"bucket": "b", "fetched_at": 1.0, "object": "x.csv", "schema_version": 1, "size": 3}'
    )
    meta = cache.lookup("b", "x.csv")
    assert meta is not None
    assert meta.cell is None and meta.watermark is None


def test_update_meta_records_watermark_and_touch(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=0)
    _put(cache, "x.csv", b"abc", ("macro", "vix", "ohlcv-1d", None))
    meta = cache.lookup("b", "x.csv")
    assert meta is not None
    cache.update_meta(meta, watermark_column="date", watermark="2026-09-20T00:00:00")
    cache.update_meta(cache.lookup("b", "x.csv"), fetched_at=123.0)  # type: ignore[arg-type]
    again = cache.lookup("b", "x.csv")
    assert again is not None
    assert (again.watermark_column, again.watermark) == ("date", "2026-09-20T00:00:00")
    assert again.fetched_at == 123.0
    assert again.cell == ("macro", "vix", "ohlcv-1d", None)


def test_refreshing_an_object_does_not_double_count_it(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=100)
    _put(cache, "x.csv", b"a" * 10, None)
    assert cache.total_bytes() == 10
    _put(cache, "x.csv", b"a" * 12, None)
    assert cache.total_bytes() == 12


def test_entries_and_clear(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path, max_bytes=0)
    _put(cache, "a.csv", b"a", None)
    _put(cache, "b.csv", b"bb", None)
    assert [e.object_name for e in cache.entries()] == ["a.csv", "b.csv"]
    assert cache.clear() == 2
    assert cache.entries() == []
    assert cache.total_bytes() == 0
