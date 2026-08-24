"""Coding-suite task definitions, batch B (tasks 7-12)."""

from __future__ import annotations

from scripts.gen_suite.coding_defs_a import CodingTaskDef

T7 = CodingTaskDef(
    id="cli-argparse-fix",
    title="Fix CLI argument handling",
    category="cli",
    difficulty=2,
    target_file="csvtool.py",
    description=(
        "The CLI's --limit flag parses as a string, and summarize() ignores "
        "the limit entirely; fix flags and honoring of the limit."
    ),
    buggy_files={
        "csvtool.py": '''"""CSV column statistics CLI.

Usage: python csvtool.py FILE --column NAME [--delimiter CH] [--limit N]
"""

import argparse
import csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="csvtool")
    p.add_argument("file", help="input CSV file")
    p.add_argument("--column", required=True, help="column to summarize")
    p.add_argument("--delimiter", default=",", help="field delimiter")
    p.add_argument("--limit", type=str, default=None, help="max rows")
    return p


def summarize(path: str, column: str, delimiter: str = ",", limit=None) -> dict:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        values = []
        for row in reader:
            raw = (row.get(column) or "").strip()
            if raw:
                values.append(float(raw))
    if not values:
        raise ValueError(f"no numeric values in column {column!r}")
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = summarize(args.file, args.column, delimiter=args.delimiter, limit=args.limit)
    print(
        f"count={stats['count']} min={stats['min']} "
        f"max={stats['max']} mean={stats['mean']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
    },
    tests={
        "tests/test_csvtool.py": '''import pytest

from csvtool import build_parser, main, summarize


@pytest.fixture
def sample_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("name,score\\na,10\\nb,20\\nc,30\\n", encoding="utf-8")
    return str(p)


def test_defaults(sample_csv):
    args = build_parser().parse_args([sample_csv, "--column", "score"])
    assert args.delimiter == ","
    assert args.limit is None


def test_limit_type_and_effect(sample_csv):
    args = build_parser().parse_args([sample_csv, "--column", "score", "--limit", "2"])
    assert isinstance(args.limit, int)
    stats = summarize(args.file, args.column, delimiter=args.delimiter, limit=args.limit)
    assert stats["count"] == 2
    assert stats["mean"] == 15.0


def test_limit_zero_means_empty_error(sample_csv):
    with pytest.raises(ValueError):
        summarize(sample_csv, "score", limit=0)


def test_required_column(sample_csv):
    with pytest.raises(SystemExit):
        build_parser().parse_args([sample_csv])


def test_main_output(sample_csv, capsys):
    rc = main([sample_csv, "--column", "score"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "count=3" in out and "mean=20.0000" in out


def test_semicolon_delimiter(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_text("name;score\\na;5\\nb;15\\n", encoding="utf-8")
    stats = summarize(str(p), "score", delimiter=";")
    assert stats["mean"] == 10
''',
    },
)

T8 = CodingTaskDef(
    id="memoize-decorator-state",
    title="Fix shared-state bug in memoize decorator",
    category="decorators",
    difficulty=3,
    target_file="memoize.py",
    description=(
        "memoize stores results in one module-global cache so different "
        "decorated functions collide; caches must be per-function."
    ),
    buggy_files={
        "memoize.py": '''"""Memoization decorator with per-function statistics."""

import functools

_SHARED_CACHE: dict[tuple, object] = {}


def memoize(func):
    """Cache results by call arguments; expose .cache_info()."""
    local_cache: dict[tuple, object] = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key in _SHARED_CACHE:
            wrapper.hits += 1
            return _SHARED_CACHE[key]
        value = func(*args, **kwargs)
        _SHARED_CACHE[key] = value
        wrapper.misses += 1
        return value

    wrapper.hits = 0
    wrapper.misses = 0

    def cache_info():
        return {"hits": wrapper.hits, "misses": wrapper.misses, "currsize": len(local_cache)}

    wrapper.cache_info = cache_info
    return wrapper


class CallCounter:
    """Counts calls to a function; resets via reset()."""

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0

    def reset(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)
''',
    },
    tests={
        "tests/test_memoize.py": '''from memoize import CallCounter, memoize


@memoize
def add(a, b):
    return a + b


@memoize
def mul(a, b):
    return a * b


def test_caches_are_per_function():
    assert add(2, 3) == 5
    assert mul(2, 3) == 6
    info_add = add.cache_info()
    assert info_add["hits"] == 0 and info_add["misses"] == 1
    assert add(2, 3) == 5
    assert add.cache_info()["hits"] == 1


def test_no_cross_function_pollution():
    add(4, 4)
    assert mul(4, 4) == 16


def test_kwargs_order_irrelevant():
    add(a=1, b=2)
    before = add.cache_info()["misses"]
    assert add(b=2, a=1) == 3
    assert add.cache_info()["misses"] == before


def test_call_counter_reset():
    calls = CallCounter(lambda x: x * 2)
    calls(1)
    calls(2)
    assert calls.count == 2
    calls.reset()
    assert calls.count == 0
''',
    },
)

T9 = CodingTaskDef(
    id="generator-streaming-memory",
    title="Fix memory blowup in streaming generator pipeline",
    category="generators",
    difficulty=3,
    target_file="stream_stats.py",
    description=(
        "running_mean materializes the whole iterable into a list; it must be "
        "a lazy generator that also supports early termination via islice."
    ),
    buggy_files={
        "stream_stats.py": '''"""Streaming statistics over large iterables."""


def running_mean(iterable):
    """Yield the running mean after each item.

    Must run in O(1) memory per step and work on infinite iterators.
    """
    items = list(iterable)
    means = []
    total = 0.0
    for i, value in enumerate(items, start=1):
        total += float(value)
        means.append(total / i)
    return means


def chunked(iterable, size):
    """Yield lists of up to `size` consecutive items."""
    if size < 1:
        raise ValueError("size must be >= 1")
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
''',
    },
    tests={
        "tests/test_stream_stats.py": '''from itertools import count, islice

import pytest

from stream_stats import chunked, running_mean


def test_running_mean_values():
    assert list(running_mean([1, 2, 3, 4])) == [1.0, 1.5, 2.0, 2.5]


def test_running_mean_is_lazy_generator():
    gen = running_mean([1, 2])
    assert hasattr(gen, "__next__"), "running_mean must return a lazy generator"


def test_running_mean_accepts_infinite_input():
    bounded = islice(count(), 5)
    first3 = list(islice(running_mean(bounded), 3))
    assert first3 == [0.0, 0.5, 1.0]


def test_running_mean_empty():
    assert list(running_mean([])) == []


def test_chunked_basic():
    chunks = list(chunked(range(10), 4))
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_chunked_exact_multiple():
    assert list(chunked([1, 2], 2)) == [[1, 2]]


def test_chunked_bad_size():
    with pytest.raises(ValueError):
        list(chunked([1], 0))
''',
    },
)

T10 = CodingTaskDef(
    id="timezone-aware-datetime",
    title="Fix naive/aware datetime comparison bug",
    category="datetime",
    difficulty=4,
    target_file="shifts.py",
    description=(
        "is_within_shift crashes comparing naive and aware datetimes when the "
        "shift boundaries carry an offset but `at` is naive, or vice versa; "
        "all inputs must be normalized to UTC before comparison."
    ),
    buggy_files={
        "shifts.py": '''"""Shift-window membership checks."""

from datetime import datetime, timedelta, timezone


def parse_utc(value: str) -> datetime:
    """Parse an ISO timestamp into an aware UTC datetime.

    Naive inputs are assumed to already be UTC.
    """
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc)


def is_within_shift(at: str | datetime, start: str | datetime, minutes: int) -> bool:
    """True when ``at`` falls in [start, start + minutes).

    BUG: mixes naive/aware comparisons and ignores the shift length.
    """
    at_dt = at if isinstance(at, datetime) else parse_utc(at)
    start_dt = start if isinstance(start, datetime) else parse_utc(start)
    end_dt = start_dt + timedelta(minutes=minutes)
    return start_dt <= at_dt <= end_dt
''',
    },
    tests={
        "tests/test_shifts.py": '''import pytest

from shifts import is_within_shift, parse_utc


def test_parse_utc_naive_assumed_utc():
    dt = parse_utc("2026-01-02T03:04:05")
    assert dt.utcoffset() is not None


def test_parse_utc_converts_offset():
    dt = parse_utc("2026-01-02T05:04:05+02:00")
    assert dt.hour == 3


def test_within_shift_basic():
    assert is_within_shift("2026-01-02T00:30:00Z", "2026-01-02T00:00:00Z", 60)


def test_after_shift_end():
    assert not is_within_shift("2026-01-02T01:00:00+00:00", "2026-01-02T00:00:00Z", 60)


def test_offset_inputs_normalized():
    assert is_within_shift("2026-01-02T02:30:00+02:00", "2026-01-02T00:00:00Z", 60)


def test_naive_vs_aware_mix_does_not_crash():
    try:
        out = is_within_shift("2026-01-02T00:30:00", "2026-01-02T01:00:00+02:00", 120)
    except TypeError:
        raise AssertionError("naive/aware comparison crashed") from None
    assert out is True
''',
    },
)

T11 = CodingTaskDef(
    id="json-schema-validator",
    title="Fix JSON schema subset validator",
    category="validation",
    difficulty=5,
    target_file="schemacheck.py",
    description=(
        "The mini validator mishandles nested objects, required lists, enum, "
        "and additionalProperties=false. Fix all documented cases."
    ),
    buggy_files={
        "schemacheck.py": r'''"""Validator for a small JSON-schema subset:

- type: object | array | string | integer | number | boolean | null
- properties / required / additionalProperties (object)
- items (array), enum (any type)

Known issues reported by users: several constraints appear to be ignored.
"""


def validate(value, schema, path="$"):
    """Return a list of human-readable error strings (empty when valid)."""
    errors: list[str] = []
    expected_type = schema.get("type")

    if expected_type == "null":
        if value is not None:
            errors.append(f"{path}: expected null")
        return errors
    if expected_type == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean")
        return errors
    if expected_type == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string")
        return errors
    if expected_type == "integer":
        if not isinstance(value, int):
            errors.append(f"{path}: expected integer")
        return errors
    if expected_type == "number":
        if not isinstance(value, int | float):
            errors.append(f"{path}: expected number")
        return errors
    if expected_type == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array")
            return errors
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                errors.extend(validate(item, item_schema, f"{path}[{i}]"))
        return errors
    if expected_type == "object":
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object")
            return errors
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                errors.extend(validate(value[key], sub, f"{path}.{key}"))
        return errors

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        errors.append(f"{path}: {value!r} is not one of {enum!r}")
    return errors
''',
    },
    tests={
        "tests/test_schemacheck.py": '''from schemacheck import validate


PERSON = {
    "type": "object",
    "required": ["name", "age"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "role": {"enum": ["user", "admin"]},
    },
}


def test_valid_person():
    errs = validate({"name": "ada", "age": 36, "role": "admin"}, PERSON)
    assert errs == []


def test_missing_required():
    errs = validate({"name": "ada"}, PERSON)
    assert any("missing required property 'age'" in e for e in errs)


def test_unexpected_property():
    errs = validate({"name": "a", "age": 1, "nickname": "x"}, PERSON)
    assert any("unexpected property 'nickname'" in e for e in errs)


def test_enum_rejection():
    errs = validate({"name": "a", "age": 1, "role": "root"}, PERSON)
    assert any("not one of" in e for e in errs)


def test_bool_is_not_integer():
    errs = validate({"name": "a", "age": True}, PERSON)
    assert any("expected integer" in e for e in errs)


def test_nested_array_items():
    schema = {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    errs = validate({"tags": ["ok", 5]}, schema)
    assert any("expected string" in e for e in errs)


def test_paths_are_reported():
    errs = validate({"tags": ["ok", 5]}, {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}})
    assert "$.tags[1]" in errs[0]


def test_null_and_number():
    assert validate(None, {"type": "null"}) == []
    assert validate(3.5, {"type": "number"}) == []
    assert validate(3, {"type": "number"}) == []
''',
    },
)

T12 = CodingTaskDef(
    id="thread-pool-deadlock",
    title="Eliminate deadlock from nested queue handoff",
    category="concurrency",
    difficulty=5,
    target_file="pipeline.py",
    description=(
        "Pipeline.submit waits forever when the pool is saturated because "
        "workers block on the same queue their callbacks feed; restructure "
        "so submit never blocks on worker completion."
    ),
    buggy_files={
        "pipeline.py": '''"""Two-stage pipeline with a result registry."""

import threading
from collections.abc import Callable


class Pipeline:
    """Runs callables on background threads; collect() gathers results."""

    def __init__(self, workers: int = 4) -> None:
        self.workers = workers
        self._tasks: list[Callable[[], None]] = []
        self._results: dict[int, object] = {}
        self._lock = threading.Lock()
        self._done = threading.Event()

    def _worker(self, idx: int) -> None:
        # BUG: worker holds nothing useful, but submit() below blocks the
        # main thread while workers wait on tasks that are never appended.
        while True:
            task = None
            with self._lock:
                if self._tasks:
                    task = self._tasks.pop(0)
            if task is None:
                if self._done.is_set():
                    return
                continue
            result = task()

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._worker, args=(i,), daemon=True)
            for i in range(self.workers)
        ]
        for t in self._threads:
            t.start()

    def submit(self, fn: Callable[[], object], task_id: int, timeout: float = 0.5) -> bool:
        """Queue fn; must never block indefinitely even when workers are busy."""
        done_here = threading.Event()

        def wrapped() -> None:
            value = fn()
            with self._lock:
                self._results[task_id] = value
            done_here.set()

        with self._lock:
            self._tasks.append(wrapped)
            # BUG: waits for completion while holding the registry lock, so
            # the worker cannot pop the task or record its result.
            finished = done_here.wait(timeout)
        return finished

    def collect(self) -> dict:
        with self._lock:
            return dict(self._results)

    def shutdown(self) -> None:
        self._done.set()
''',
    },
    tests={
        "tests/test_pipeline.py": '''import time

from pipeline import Pipeline


def test_submit_returns_without_waiting_for_completion():
    p = Pipeline(workers=2)
    p.start()
    started = time.monotonic()
    ok = p.submit(lambda: time.sleep(0.2) or "slow", task_id=1, timeout=0.05)
    elapsed = time.monotonic() - started
    assert elapsed < 0.19, "submit blocked until the task finished"
    p.shutdown()


def test_results_eventually_available():
    p = Pipeline(workers=2)
    p.start()
    p.submit(lambda: 10, task_id=1)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and 1 not in p.collect():
        time.sleep(0.01)
    assert p.collect()[1] == 10
    p.shutdown()


def test_many_tasks_no_deadlock():
    p = Pipeline(workers=2)
    p.start()
    for i in range(50):
        assert p.submit(lambda v=i: v, task_id=i, timeout=1.0)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and len(p.collect()) < 50:
        time.sleep(0.01)
    assert len(p.collect()) == 50
    p.shutdown()


def test_collect_snapshot_is_copy():
    p = Pipeline(workers=1)
    p.start()
    snap = p.collect()
    snap["x"] = 1
    assert "x" not in p.collect()
    p.shutdown()
''',
    },
)

TASKS_B = [T7, T8, T9, T10, T11, T12]
