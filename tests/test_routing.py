"""The routing matrix. Pure functions — no HTTP, no filesystem, no clock."""

from __future__ import annotations

import datetime as dt

import pytest

from rwpytools.errors import RoutingError
from rwpytools.routing import (
    BulkAction,
    BulkState,
    DateRange,
    Source,
    as_date,
    plan_route,
)

TODAY = dt.date(2026, 9, 24)
WATERMARK = dt.date(2026, 9, 20)
CACHED = BulkState(cached=True, watermark=WATERMARK)


def _plan(**kw):  # type: ignore[no-untyped-def]
    defaults = {
        "dataset": "demo",
        "has_bulk": True,
        "has_live": True,
        "today": TODAY,
        "live_only_window_days": 30,
        "live_diff_max_days": 30,
    }
    defaults.update(kw)
    return plan_route(**defaults)


# ---- auto: bulk + live ----------------------------------------------------------


def test_straddle_splits_at_the_watermark_day() -> None:
    route = _plan(start=dt.date(2024, 1, 1), bulk=CACHED)
    assert route.sources == ("bulk", "live")
    assert route.bulk_range == DateRange(dt.date(2024, 1, 1), WATERMARK)
    # Live starts ON the watermark: intraday exports can end mid-day.
    assert route.live_range == DateRange(WATERMARK, None)
    assert route.split_at == WATERMARK
    assert route.bulk_action is BulkAction.CACHE
    assert not route.needs_download


def test_range_before_watermark_is_all_bulk() -> None:
    route = _plan(start=dt.date(2024, 1, 1), end=dt.date(2025, 1, 1), bulk=CACHED)
    assert route.sources == ("bulk",)
    assert route.bulk_action is BulkAction.CACHE


def test_range_after_watermark_is_all_live() -> None:
    route = _plan(start=dt.date(2026, 9, 22), bulk=CACHED)
    assert route.sources == ("live",)
    assert route.bulk_action is None


def test_nothing_cached_downloads() -> None:
    route = _plan(start=dt.date(2020, 1, 1))
    assert route.sources == ("bulk",)
    assert route.bulk_action is BulkAction.DOWNLOAD
    assert route.needs_download


def test_nothing_cached_recent_range_goes_live_without_download() -> None:
    route = _plan(start=TODAY - dt.timedelta(days=10))
    assert route.sources == ("live",)
    assert "instead of downloading" in route.reason


def test_nothing_cached_unbounded_start_downloads() -> None:
    assert _plan().bulk_action is BulkAction.DOWNLOAD


def test_old_watermark_past_ttl_refreshes() -> None:
    old = BulkState(cached=True, watermark=dt.date(2026, 1, 1), expired=True)
    route = _plan(bulk=old)
    assert route.bulk_action is BulkAction.REFRESH


def test_old_watermark_within_ttl_is_topped_up_not_refreshed() -> None:
    """An export that is itself behind must not be re-issued (and re-charged)
    on every call — the live diff covers the gap until the TTL lapses."""

    old = BulkState(cached=True, watermark=dt.date(2026, 1, 1), expired=False)
    route = _plan(bulk=old)
    assert route.sources == ("bulk", "live")
    assert route.bulk_action is BulkAction.CACHE
    assert "days behind" in route.reason


def test_old_watermark_irrelevant_when_range_ends_before_it() -> None:
    old = BulkState(cached=True, watermark=dt.date(2026, 1, 1), expired=True)
    route = _plan(end=dt.date(2025, 6, 1), bulk=old)
    assert route.bulk_action is BulkAction.CACHE


def test_force_refresh_refreshes() -> None:
    route = _plan(bulk=CACHED, force_refresh=True)
    assert route.bulk_action is BulkAction.REFRESH


# ---- auto: single-source datasets ----------------------------------------------------


def test_bulk_only_dataset_uses_cache_until_expired() -> None:
    assert _plan(has_live=False, bulk=CACHED).bulk_action is BulkAction.CACHE
    expired = BulkState(cached=True, watermark=WATERMARK, expired=True)
    assert _plan(has_live=False, bulk=expired).bulk_action is BulkAction.REFRESH
    assert _plan(has_live=False).bulk_action is BulkAction.DOWNLOAD


def test_live_only_dataset_is_live() -> None:
    route = _plan(has_bulk=False)
    assert route.sources == ("live",)


def test_dataset_with_neither_source_is_an_error() -> None:
    with pytest.raises(RoutingError):
        _plan(has_bulk=False, has_live=False)


# ---- forced ------------------------------------------------------------------------------


def test_forced_bulk_ignores_live() -> None:
    route = _plan(start=dt.date(2024, 1, 1), bulk=CACHED, source="bulk")
    assert route.sources == ("bulk",)
    assert route.forced
    assert route.bulk_action is BulkAction.CACHE


def test_forced_bulk_downloads_when_missing() -> None:
    assert _plan(source=Source.BULK).bulk_action is BulkAction.DOWNLOAD


def test_forced_live_ignores_cache() -> None:
    route = _plan(start=dt.date(2024, 1, 1), bulk=CACHED, source="LIVE")
    assert route.sources == ("live",)
    assert route.live_range == DateRange(dt.date(2024, 1, 1), None)


def test_forcing_a_missing_source_is_an_error() -> None:
    with pytest.raises(RoutingError, match="no bulk export"):
        _plan(has_bulk=False, source="bulk")
    with pytest.raises(RoutingError, match="no live endpoint"):
        _plan(has_live=False, source="live")


def test_bad_source_is_an_error() -> None:
    with pytest.raises(RoutingError, match="source must be one of"):
        _plan(source="cache")


def test_empty_range_is_an_error() -> None:
    with pytest.raises(RoutingError, match="empty"):
        _plan(start=dt.date(2025, 1, 2), end=dt.date(2025, 1, 1))


# ---- presentation --------------------------------------------------------------------------


def test_describe_and_json_carry_the_decision() -> None:
    route = _plan(start=dt.date(2024, 1, 1), bulk=CACHED)
    text = route.describe()
    assert "bulk[2024-01-01 .. 2026-09-20] (cache)" in text
    assert "live[2026-09-20 .. now]" in text
    payload = route.to_json()
    assert payload["sources"] == ["bulk", "live"]
    assert payload["watermark"] == "2026-09-20"
    assert payload["bulk_action"] == "cache"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("2024-01-05", dt.date(2024, 1, 5)),
        ("2024-01-05T13:00:00Z", dt.date(2024, 1, 5)),
        ("2024-01-05 13:00:00", dt.date(2024, 1, 5)),
        (dt.datetime(2024, 1, 5, 13), dt.date(2024, 1, 5)),
        (dt.date(2024, 1, 5), dt.date(2024, 1, 5)),
    ],
)
def test_as_date(value: object, expected: dt.date | None) -> None:
    assert as_date(value) == expected


def test_as_date_rejects_garbage() -> None:
    with pytest.raises(RoutingError):
        as_date("yesterday")
    with pytest.raises(RoutingError):
        as_date(20240105)


def test_cached_export_without_a_watermark_is_served_from_cache() -> None:
    """Snapshots (no date column) cannot be diffed; they must not be
    re-planned as a download on every call."""

    snapshot = BulkState(cached=True, watermark=None)
    route = _plan(bulk=snapshot)
    assert route.sources == ("bulk",)
    assert route.bulk_action is BulkAction.CACHE
    expired = BulkState(cached=True, watermark=None, expired=True)
    assert _plan(bulk=expired).bulk_action is BulkAction.REFRESH
