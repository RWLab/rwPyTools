"""The on-disk cache, and why it protects your bandwidth cap.

rw-api charges a bulk export's full size against your key's bandwidth cap
each time it issues a download URL. The client therefore downloads an
export once, and afterwards reads it from disk without calling rw-api at
all — topping it up from the live API instead.

This example uses a throwaway cache directory so it starts cold.

Run::

    RWPYTOOLS_API_KEY=... python examples/cache_and_bandwidth.py
"""

from __future__ import annotations

import datetime as dt
import tempfile
import time

import rwpytools


def main() -> None:
    with (
        tempfile.TemporaryDirectory() as cache_dir,
        rwpytools.Client(cache_dir=cache_dir) as client,
    ):
        started = time.monotonic()
        first = client.get("vix")
        print(f"first call ({time.monotonic() - started:.1f}s): {first.describe_route()}")

        started = time.monotonic()
        second = client.get("vix")
        api_called = [obj.api_called for obj in second.bulk_objects]
        print(
            f"second call ({time.monotonic() - started:.1f}s): "
            f"served_from_cache={second.served_from_cache}, "
            f"rw-api called for the export: {api_called}"
        )

        # Make sure an export is on disk without reading it.
        fetched = client.download("fx_daily", symbol="EURUSD")
        print(f"\ndownloaded {fetched.object_name}: {fetched.size:,} bytes at {fetched.path}")

        print(f"\ncache holds {client.cache.total_bytes():,} bytes:")
        for entry in client.cache.entries():
            fetched_at = dt.datetime.fromtimestamp(entry.fetched_at).isoformat(timespec="seconds")
            print(
                f"  {entry.bucket}/{entry.object_name}: {entry.size:,} bytes, fetched {fetched_at}"
            )

        # force_refresh re-downloads (and is charged again) — use sparingly.
        refreshed = client.get("vix", source="bulk", force_refresh=True)
        print(f"\nforce_refresh: served_from_cache={refreshed.served_from_cache}")

        print(f"\ncleared {client.cache.clear()} cached objects")


if __name__ == "__main__":
    main()
