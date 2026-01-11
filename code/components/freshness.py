from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional, Sequence


@dataclass
class FreshnessResult:
    status: str
    age_days: Optional[int]
    reference_date: date
    source_date: Optional[date]
    message: str


_DATE_PATTERNS: Sequence[str] = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d.%m.%Y",
    "%d.%m.%Y %H:%M",
    "%H:%M %d.%m.%Y",
)


def _parse_date(value: str) -> Optional[datetime]:
    """
    Try to parse a date string in several common formats.
    Returns naive UTC datetime on success, None otherwise.
    """
    value = value.strip()
    if not value:
        return None

    # Fast path: ISO 8601 with optional timezone
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        pass

    for pattern in _DATE_PATTERNS:
        try:
            dt = datetime.strptime(value, pattern)
            return dt
        except ValueError:
            continue

    return None


def assess_freshness(date_str: Optional[str], *, today: Optional[date] = None) -> FreshnessResult:
    """
    Compare provided publish date with the current calendar date and return a freshness label.

    Labels:
    - 0 days: "today"
    - 1 day: "yesterday"
    - 2-7 days: "recent"
    - >7 days: "stale"
    - missing/unparseable/future: "unknown"
    """
    if today is None:
        today = datetime.now(timezone.utc).date()

    if not date_str:
        return FreshnessResult(
            status="unknown",
            age_days=None,
            reference_date=today,
            source_date=None,
            message="Дата не указана",
        )

    parsed = _parse_date(date_str)
    if parsed is None:
        return FreshnessResult(
            status="unknown",
            age_days=None,
            reference_date=today,
            source_date=None,
            message="Не удалось распарсить дату",
        )

    source_date = parsed.date()
    delta = (today - source_date).days
    if delta < 0:
        return FreshnessResult(
            status="unknown",
            age_days=None,
            reference_date=today,
            source_date=source_date,
            message="Дата в будущем",
        )

    if delta == 0:
        status = "today"
        message = "Опубликовано сегодня"
    elif delta == 1:
        status = "yesterday"
        message = "Опубликовано вчера"
    elif delta <= 7:
        status = "recent"
        message = f"Опубликовано {delta} дн. назад"
    else:
        status = "stale"
        message = f"Опубликовано {delta} дн. назад"

    return FreshnessResult(
        status=status,
        age_days=delta,
        reference_date=today,
        source_date=source_date,
        message=message,
    )


def main() -> None:
    """
    CLI usage:

    uv run python -m components.freshness "2025-12-17T10:00:00"
    """
    import sys

    if len(sys.argv) < 2:
        print("Использование: python -m components.freshness \"<дата>\"")
        sys.exit(1)

    date_str = " ".join(sys.argv[1:])
    try:
        result = assess_freshness(date_str)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    print(
        f"Дата новости: {result.source_date.isoformat()} "
        f"({result.days_diff} дн. назад) — {result.label}"
    )


if __name__ == "__main__":
    main()
