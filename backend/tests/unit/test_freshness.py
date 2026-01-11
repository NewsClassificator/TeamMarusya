import sys
from datetime import date
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[3] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from components.freshness import assess_freshness  # type: ignore E402


def test_freshness_today():
    today = date(2025, 12, 22)
    res = assess_freshness("2025-12-22", today=today)
    assert res.status == "today"
    assert res.age_days == 0
    assert res.reference_date == today


def test_freshness_yesterday():
    today = date(2025, 12, 22)
    res = assess_freshness("2025-12-21", today=today)
    assert res.status == "yesterday"
    assert res.age_days == 1


def test_freshness_recent_and_stale():
    today = date(2025, 12, 22)
    res_recent = assess_freshness("2025-12-18", today=today)
    assert res_recent.status == "recent"
    res_stale = assess_freshness("2025-12-10", today=today)
    assert res_stale.status == "stale"


def test_freshness_unknown_missing_or_future():
    today = date(2025, 12, 22)
    missing = assess_freshness("", today=today)
    assert missing.status == "unknown"
    future = assess_freshness("2025-12-30", today=today)
    assert future.status == "unknown"
