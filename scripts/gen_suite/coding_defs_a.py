"""Coding-suite task definitions, batch A (tasks 1-6)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodingTaskDef:
    id: str
    title: str
    category: str
    difficulty: int
    description: str
    target_file: str
    buggy_files: dict[str, str]
    tests: dict[str, str] = field(default_factory=dict)
    solution: dict[str, str] = field(default_factory=dict)


T1 = CodingTaskDef(
    id="binary-search-boundary",
    title="Fix off-by-one bugs in boundary binary search",
    category="algorithms",
    difficulty=2,
    target_file="search_bounds.py",
    description=(
        "find_rightmost returns wrong indices for repeated values, which also "
        "breaks count_occurrences."
    ),
    buggy_files={
        "search_bounds.py": '''"""Binary search helpers for sorted lists."""


def find_leftmost(values: list[int], target: int) -> int:
    """Return the index of the first element equal to target, or -1."""
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(values) and values[lo] == target:
        return lo
    return -1


def find_rightmost(values: list[int], target: int) -> int:
    """Return the index of the last element equal to target, or -1."""
    lo, hi = 0, len(values) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    if lo > 0 and values[lo - 1] == target:
        return lo + 1
    return -1


def count_occurrences(values: list[int], target: int) -> int:
    """Count occurrences of target using the two boundary searches."""
    left = find_leftmost(values, target)
    if left == -1:
        return 0
    right = find_rightmost(values, target)
    return right - left + 1
''',
    },
    tests={
        "tests/test_search_bounds.py": '''from search_bounds import count_occurrences, find_leftmost, find_rightmost


def test_leftmost_basic():
    assert find_leftmost([1, 2, 2, 3], 2) == 1


def test_rightmost_basic():
    assert find_rightmost([1, 2, 2, 3], 2) == 2


def test_missing_value():
    assert find_leftmost([1, 3, 5], 4) == -1
    assert find_rightmost([1, 3, 5], 0) == -1


def test_single_element():
    assert find_leftmost([7], 7) == 0
    assert find_rightmost([7], 7) == 0


def test_all_same():
    assert find_leftmost([5, 5, 5, 5], 5) == 0
    assert find_rightmost([5, 5, 5, 5], 5) == 3


def test_count():
    assert count_occurrences([1, 2, 2, 2, 3], 2) == 3
    assert count_occurrences([1, 2, 3], 9) == 0


def test_empty():
    assert count_occurrences([], 1) == 0
''',
    },
    solution={
        "right_branch": '''def find_rightmost(values, target):
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if idx >= 0 and values[idx] == target:
        return idx
    return -1
''',
    },
)

T2 = CodingTaskDef(
    id="async-rate-limiter-race",
    title="Fix async race condition in token-bucket limiter",
    category="concurrency",
    difficulty=4,
    target_file="limiter.py",
    description=(
        "Concurrent callers can both pass the emptiness check before either "
        "decrements, admitting more requests than the capacity allows."
    ),
    buggy_files={
        "limiter.py": '''"""Async token-bucket rate limiter."""

import time


class TokenBucketLimiter:
    """Allows at most ``capacity`` acquisitions per ``refill_period`` seconds."""

    def __init__(self, capacity: int, refill_period: float = 1.0) -> None:
        self.capacity = capacity
        self.refill_period = refill_period
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed >= self.refill_period:
            self._tokens = float(self.capacity)
            self._last_refill = now

    async def acquire(self) -> None:
        """Consume one token; raise RuntimeError when the bucket is empty.

        BUG: the check-then-decrement sequence interleaves between tasks.
        """
        self._refill()
        if self._tokens < 1:
            raise RuntimeError("rate limit exceeded")
        await _yield_point()
        self._tokens -= 1

    @property
    def available_tokens(self) -> float:
        return self._tokens


async def _yield_point() -> None:
    import asyncio

    await asyncio.sleep(0)
''',
    },
    tests={
        "tests/test_limiter.py": '''import asyncio

import pytest

from limiter import TokenBucketLimiter


@pytest.mark.asyncio
async def test_admits_capacity_requests_only():
    limiter = TokenBucketLimiter(capacity=5, refill_period=3600)

    async def worker():
        await limiter.acquire()

    results = await asyncio.gather(*[worker() for _ in range(20)], return_exceptions=True)
    admitted = sum(1 for r in results if not isinstance(r, BaseException))
    assert admitted == 5


@pytest.mark.asyncio
async def test_rejects_when_empty():
    limiter = TokenBucketLimiter(capacity=1, refill_period=3600)
    await limiter.acquire()
    with pytest.raises(RuntimeError):
        await limiter.acquire()


@pytest.mark.asyncio
async def test_refills_after_period():
    limiter = TokenBucketLimiter(capacity=2, refill_period=0.05)
    await limiter.acquire()
    await limiter.acquire()
    await asyncio.sleep(0.08)
    await limiter.acquire()
    assert limiter.available_tokens == 1.0
''',
    },
)

T3 = CodingTaskDef(
    id="http-client-refactor",
    title="Refactor transport without breaking the public interface",
    category="api-design",
    difficulty=3,
    target_file="weather_client.py",
    description=(
        "Route all public methods through one shared `_request` helper while "
        "keeping every documented public behavior identical, including the "
        "request_count accounting."
    ),
    buggy_files={
        "weather_client.py": '''"""Tiny weather lookup client.

Public interface (must remain stable):

- ``current(city)`` returns a dict with keys ``city`` and ``temp_c``.
- ``forecast(city, days)`` returns a list of dicts with ``day`` and
  ``temp_c``, of length ``days``; raises ValueError when days < 1.
- ``is_available`` is a bool property.
"""


class WeatherClient:
    def __init__(self, base_url: str = "https://api.example.com", timeout: float = 10.0) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.request_count = 0

    def current(self, city: str) -> dict:
        self.request_count += 1
        payload = {"kind": "current", "city": city.title(), "temp_c": 21}
        return {"city": payload["city"], "temp_c": payload["temp_c"]}

    def forecast(self, city: str, days: int) -> list[dict]:
        if days < 1:
            raise ValueError("days must be >= 1")
        self.request_count += days
        rows = [{"kind": "forecast", "day": i + 1, "temp_c": 18 + i} for i in range(days)]
        return [{"day": r["day"], "temp_c": r["temp_c"]} for r in rows]

    @property
    def is_available(self) -> bool:
        return bool(self.base_url)
''',
    },
    tests={
        "tests/test_weather_client.py": '''import inspect

from weather_client import WeatherClient


def test_current_shape():
    c = WeatherClient()
    out = c.current("berlin")
    assert out == {"city": "Berlin", "temp_c": 21}


def test_forecast_length_and_validation():
    c = WeatherClient()
    assert len(c.forecast("oslo", 3)) == 3
    try:
        c.forecast("oslo", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for days=0")


def test_request_counting_via_shared_helper():
    c = WeatherClient()
    c.current("paris")
    c.forecast("paris", 4)
    assert c.request_count == 5


def test_availability_property():
    assert WeatherClient().is_available is True


def test_public_api_uses_shared_transport():
    members = inspect.getmembers(WeatherClient, inspect.isfunction)
    assert any(name.startswith("_") and name != "__init__" for name, _ in members), (
        "public methods must delegate to a private helper"
    )
''',
    },
)

T4 = CodingTaskDef(
    id="pagination-window",
    title="Fix pagination window slicing off-by-one",
    category="off-by-one",
    difficulty=2,
    target_file="paginate.py",
    description="page_window drops the last item of every page.",
    buggy_files={
        "paginate.py": '''"""Helpers for offset-based pagination."""


def total_pages(item_count: int, page_size: int) -> int:
    """Number of pages needed for item_count items."""
    if item_count < 0 or page_size < 1:
        raise ValueError("invalid arguments")
    return (item_count + page_size - 1) // page_size


def page_window(items: list, page: int, page_size: int) -> list:
    """Return the 0-based ``page`` slice; IndexError when out of range."""
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    last = total_pages(len(items), page_size)
    if page < 0 or page >= last:
        raise IndexError("page out of range")
    start = page * page_size
    return items[start : start + page_size - 1]


def paginate_all(items: list, page_size: int) -> list[list]:
    """Split items into consecutive pages."""
    return [page_window(items, p, page_size) for p in range(total_pages(len(items), page_size))]
''',
    },
    tests={
        "tests/test_paginate.py": '''import pytest

from paginate import page_window, paginate_all, total_pages


def test_total_pages():
    assert total_pages(10, 3) == 4
    assert total_pages(0, 5) == 0
    assert total_pages(9, 3) == 3


def test_window_no_overlap():
    pages = paginate_all(list(range(10)), 3)
    flat = [x for p in pages for x in p]
    assert flat == list(range(10))


def test_window_exact_page():
    assert page_window(list(range(6)), 1, 3) == [3, 4, 5]


def test_out_of_range():
    with pytest.raises(IndexError):
        page_window([1, 2], 2, 2)


def test_bad_page_size():
    with pytest.raises(ValueError):
        page_window([1], 0, 0)


def test_empty_items():
    assert paginate_all([], 4) == []
''',
    },
)

T5 = CodingTaskDef(
    id="log-line-parser-regex",
    title="Fix regex log-line parser",
    category="regex",
    difficulty=3,
    target_file="logparse.py",
    description=(
        "The component pattern swallows colons so messages like "
        "'connection lost: timeout' are truncated; TRACE lines are wrongly accepted."
    ),
    buggy_files={
        "logparse.py": r'''"""Parse structured log lines:

    LEVEL [2026-01-02T03:04:05Z] component: message text here
"""

import re
from dataclasses import dataclass

LINE_RE = re.compile(
    r"^(?P<level>DEBUG|INFO|WARNING|ERROR|TRACE) \[(?P<ts>[^]]+)\] (?P<component>[\w-]+): (?P<message>.*)$"
)


@dataclass(frozen=True)
class LogLine:
    level: str
    timestamp: str
    component: str
    message: str


def parse_line(line: str) -> LogLine | None:
    """Parse one line; return None when malformed."""
    m = LINE_RE.match(line.strip())
    if m is None:
        return None
    return LogLine(
        level=m.group("level"),
        timestamp=m.group("ts"),
        component=m.group("component"),
        message=m.group("message").strip(),
    )


def parse_lines(text: str) -> list[LogLine]:
    """Parse all well-formed lines, skipping blanks and malformed ones."""
    out = []
    for raw in text.splitlines():
        parsed = parse_line(raw)
        if parsed is not None:
            out.append(parsed)
    return out
''',
    },
    tests={
        "tests/test_logparse.py": '''from logparse import parse_line, parse_lines


SAMPLE = (
    "INFO [2026-01-02T03:04:05Z] web: request completed path=/health status=200\\n"
    "ERROR [2026-01-02T03:04:06Z] db-worker: connection lost: timeout after 30s\\n"
    "garbage line\\n"
    "\\n"
    "DEBUG [2026-01-02T03:04:07Z] scheduler: tick"
)


def test_message_with_colons_and_spaces():
    row = parse_line("ERROR [2026-01-02T03:04:06Z] db: connection lost: timeout after 30s")
    assert row is not None
    assert row.message == "connection lost: timeout after 30s"
    assert row.component == "db"


def test_malformed_rejected():
    assert parse_line("INFO missing-brackets web: hi") is None
    assert parse_line("TRACE [2026-01-01T00:00:00Z] web: nope") is None
    assert parse_line("") is None


def test_multi_line_parse():
    rows = parse_lines(SAMPLE)
    assert len(rows) == 3
    assert rows[0].level == "INFO"
    assert rows[1].level == "ERROR"
    assert rows[2].component == "scheduler"


def test_timestamp_preserved():
    rows = parse_lines(SAMPLE)
    assert rows[0].timestamp == "2026-01-02T03:04:05Z"
''',
    },
)

T6 = CodingTaskDef(
    id="dataclass-to-pydantic",
    title="Complete dataclass-to-pydantic v2 migration",
    category="migration",
    difficulty=4,
    target_file="models_user.py",
    description=(
        "Migrate User/AdminUser to pydantic v2 models preserving the "
        "documented behavior contract (email validation, display_name "
        "default, independent scopes defaults, from_row)."
    ),
    buggy_files={
        "models_user.py": '''"""User domain models. Behavior contract:

- ``User.email`` must contain "@" or construction fails with ValidationError.
- ``User.display_name`` defaults to the local part of the email.
- ``AdminUser`` inherits User fields plus ``scopes`` (default empty set).
- ``from_row`` builds a User from a DB tuple ``(id, email)``.
"""

from dataclasses import dataclass, field


@dataclass
class User:
    id: int
    email: str
    display_name: str = ""

    def __post_init__(self) -> None:
        if "@" not in self.email:
            raise ValueError("invalid email")
        if not self.display_name:
            self.display_name = self.email.split("@")[0]

    @classmethod
    def from_row(cls, row: tuple) -> "User":
        uid, email = row
        return cls(id=uid, email=email)


@dataclass
class AdminUser(User):
    scopes: set[str] = field(default_factory=set)


def validate_email(email: str) -> bool:
    return "@" in email
''',
    },
    tests={
        "tests/test_models_user.py": '''import pydantic
import pytest

from models_user import AdminUser, User, validate_email


@pytest.mark.parametrize("bad", ["nope", ""])
def test_email_validation(bad):
    with pytest.raises(pydantic.ValidationError):
        User(id=1, email=bad)


def test_display_name_default():
    u = User(id=1, email="ada@example.com")
    assert u.display_name == "ada"


def test_display_name_explicit():
    u = User(id=1, email="ada@example.com", display_name="Ada L.")
    assert u.display_name == "Ada L."


def test_admin_scopes_default_independent():
    a = AdminUser(id=2, email="root@example.com")
    b = AdminUser(id=3, email="alt@example.com")
    assert a.scopes == set()
    a.scopes.add("write")
    assert b.scopes == set()


def test_from_row():
    u = User.from_row((9, "g@example.com"))
    assert (u.id, u.email) == (9, "g@example.com")


def test_validate_email():
    assert validate_email("x@y") is True
    assert validate_email("xy") is False


def test_is_pydantic_model():
    assert issubclass(User, pydantic.BaseModel)
''',
    },
)

TASKS_A = [T1, T2, T3, T4, T5, T6]
