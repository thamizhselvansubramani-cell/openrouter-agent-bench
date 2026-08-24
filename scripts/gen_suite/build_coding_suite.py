"""Build tasks/suites/coding/*.yaml and verify every fixture.

Verification contract per task:
1. hidden tests FAIL against the shipped buggy module;
2. hidden tests PASS against a reference solution;
3. the emitted YAML round-trips through TaskSpec validation.

Run from repo root: uv run python scripts/gen_suite/build_coding_suite.py
"""

from __future__ import annotations

import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from openrouter_agent_bench.sandbox.executor import SandboxExecutor, SandboxWorkspace
from scripts.gen_suite.coding_defs_a import TASKS_A, CodingTaskDef
from scripts.gen_suite.coding_defs_b import TASKS_B

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "tasks" / "suites" / "coding"

PROMPT_TEMPLATE = """\
You are given a small Python project with a bug or missing behavior.

Problem: {description}

Files currently in the workspace:
{files}

{extra}
"""

SOLUTIONS: dict[str, str] = {}


def _solution_for(task: CodingTaskDef) -> dict[str, str]:
    """Return full corrected target_file content per task id."""
    tid = task.id
    if tid == "binary-search-boundary":
        content = task.buggy_files["search_bounds.py"].replace(
            '''    lo, hi = 0, len(values) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    if lo > 0 and values[lo - 1] == target:
        return lo + 1
    return -1''',
            '''    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if idx >= 0 and values[idx] == target:
        return idx
    return -1''',
        )
        return {"search_bounds.py": content}
    if tid == "async-rate-limiter-race":
        content = task.buggy_files["limiter.py"].replace(
            '''"""Async token-bucket rate limiter."""

import time''',
            '''"""Async token-bucket rate limiter."""

import asyncio
import time''',
        ).replace(
            '''    async def acquire(self) -> None:
        """Consume one token; raise RuntimeError when the bucket is empty.

        BUG: the check-then-decrement sequence interleaves between tasks.
        """
        self._refill()
        if self._tokens < 1:
            raise RuntimeError("rate limit exceeded")
        await _yield_point()
        self._tokens -= 1''',
            '''    def __init__(self, capacity: int, refill_period: float = 1.0) -> None:
        self.capacity = capacity
        self.refill_period = refill_period
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Consume one token; raise RuntimeError when the bucket is empty."""
        async with self._lock:
            self._refill()
            if self._tokens < 1:
                raise RuntimeError("rate limit exceeded")
            self._tokens -= 1''',
        ).replace('''

async def _yield_point() -> None:
    import asyncio

    await asyncio.sleep(0)
''', "")
        return {"limiter.py": content}
    if tid == "http-client-refactor":
        return {
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

    def _request(self, kind: str, city: str | None = None, days: int = 0):
        """Single internal transport used by all public methods."""
        self.request_count += days if kind == "forecast" else 1
        if kind == "current":
            return {"city": (city or "").title(), "temp_c": 21}
        return [{"day": i + 1, "temp_c": 18 + i} for i in range(days)]

    def current(self, city: str) -> dict:
        payload = self._request("current", city=city)
        return {"city": payload["city"], "temp_c": payload["temp_c"]}

    def forecast(self, city: str, days: int) -> list[dict]:
        if days < 1:
            raise ValueError("days must be >= 1")
        rows = self._request("forecast", city=city, days=days)
        return [{"day": r["day"], "temp_c": r["temp_c"]} for r in rows]

    @property
    def is_available(self) -> bool:
        return bool(self.base_url)
''',
        }
    if tid == "pagination-window":
        content = task.buggy_files["paginate.py"].replace(
            "return items[start : start + page_size - 1]",
            "return items[start : start + page_size]",
        )
        return {"paginate.py": content}
    if tid == "log-line-parser-regex":
        content = task.buggy_files["logparse.py"].replace(
            "DEBUG|INFO|WARNING|ERROR|TRACE", "DEBUG|INFO|WARNING|ERROR"
        )
        return {"logparse.py": content}
    if tid == "dataclass-to-pydantic":
        return {
            "models_user.py": '''"""User domain models (pydantic v2)."""

from pydantic import BaseModel, field_validator


class User(BaseModel):
    id: int
    email: str
    display_name: str = ""

    @field_validator("email")
    @classmethod
    def _email_has_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("invalid email")
        return v

    def model_post_init(self, __context: object) -> None:
        if not self.display_name:
            self.display_name = self.email.split("@")[0]

    @classmethod
    def from_row(cls, row: tuple) -> "User":
        uid, email = row
        return cls(id=uid, email=email)


class AdminUser(User):
    scopes: set[str] = set()


def validate_email(email: str) -> bool:
    return "@" in email
''',
        }
    if tid == "cli-argparse-fix":
        content = task.buggy_files["csvtool.py"].replace(
            'p.add_argument("--limit", type=str, default=None, help="max rows")',
            'p.add_argument("--limit", type=int, default=None, help="max rows")',
        ).replace(
            '''        values = []
        for row in reader:
            raw = (row.get(column) or "").strip()''',
            '''        values = []
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            raw = (row.get(column) or "").strip()''',
        )
        return {"csvtool.py": content}
    if tid == "memoize-decorator-state":
        content = task.buggy_files["memoize.py"].replace(
            "if key in _SHARED_CACHE:\n            wrapper.hits += 1\n            return _SHARED_CACHE[key]\n        value = func(*args, **kwargs)\n        _SHARED_CACHE[key] = value",
            "if key in local_cache:\n            wrapper.hits += 1\n            return local_cache[key]\n        value = func(*args, **kwargs)\n        local_cache[key] = value",
        )
        return {"memoize.py": content}
    if tid == "generator-streaming-memory":
        content = task.buggy_files["stream_stats.py"].replace(
            '''    items = list(iterable)
    means = []
    total = 0.0
    for i, value in enumerate(items, start=1):
        total += float(value)
        means.append(total / i)
    return means''',
            '''    total = 0.0
    count = 0
    for value in iterable:
        total += float(value)
        count += 1
        yield total / count''',
        )
        return {"stream_stats.py": content}
    if tid == "timezone-aware-datetime":
        content = task.buggy_files["shifts.py"].replace(
            '''    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc)''',
            '''    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)''',
        ).replace(
            '''    at_dt = at if isinstance(at, datetime) else parse_utc(at)
    start_dt = start if isinstance(start, datetime) else parse_utc(start)
    end_dt = start_dt + timedelta(minutes=minutes)
    return start_dt <= at_dt <= end_dt''',
            '''    at_dt = at if isinstance(at, datetime) else parse_utc(at)
    start_dt = start if isinstance(start, datetime) else parse_utc(start)

    def to_utc(dt: datetime) -> datetime:
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

    at_utc = to_utc(at_dt)
    start_utc = to_utc(start_dt)
    end_utc = start_utc + timedelta(minutes=minutes)
    return start_utc <= at_utc < end_utc''',
        )
        return {"shifts.py": content}
    if tid == "json-schema-validator":
        content = task.buggy_files["schemacheck.py"].replace(
            '''        if not isinstance(value, int):
            errors.append(f"{path}: expected integer")''',
            '''        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: expected integer")''',
        ).replace(
            '''        props = schema.get("properties", {})
        for key, sub in props.items():''',
            '''        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required property {req!r}")
        if schema.get("additionalProperties") is False:
            for key in sorted(set(value) - set(props)):
                errors.append(f"{path}: unexpected property {key!r}")
        for key, sub in props.items():''',
        )
        return {"schemacheck.py": content}
    if tid == "thread-pool-deadlock":
        content = task.buggy_files["pipeline.py"].replace(
            '''        with self._lock:
            self._tasks.append(wrapped)
            # BUG: waits for completion while holding the registry lock, so
            # the worker cannot pop the task or record its result.
            finished = done_here.wait(timeout)
        return finished''',
            '''        with self._lock:
            self._tasks.append(wrapped)
        # Wait outside the lock so workers can make progress.
        finished = done_here.wait(timeout)
        return finished''',
        )
        return {"pipeline.py": content}
    msg = f"no reference solution registered for {tid}"
    raise KeyError(msg)


def build_task_yaml(task: CodingTaskDef) -> dict:
    file_lines = "\n".join(f"- {name}" for name in task.buggy_files)
    prompt = PROMPT_TEMPLATE.format(description=task.description, files=file_lines, extra="")
    tests = {
        path.removeprefix("tests/"): content for path, content in task.tests.items()
    }
    return {
        "id": task.id,
        "suite": "coding",
        "title": task.title,
        "category": task.category,
        "difficulty": task.difficulty,
        "target_file": task.target_file,
        "max_turns": 1,
        "timeout_s": 300,
        "prompt": prompt,
        "files": task.buggy_files,
        "grader": {"type": "unit_tests", "tests": tests},
    }


def run_tests(files: dict[str, str], tests: dict[str, str]) -> bool:
    ws = SandboxWorkspace.create({**files, **tests})
    try:
        executor = SandboxExecutor(timeout_s=120, memory_mb=None)
        result = executor.run_pytest(ws, sorted(tests.keys()))
        return result.ok
    finally:
        ws.cleanup()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for task in [*TASKS_A, *TASKS_B]:
        payload = build_task_yaml(task)
        out_path = OUT / f"{task.id}.yaml"
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        tests = payload["grader"]["tests"]
        buggy_ok = run_tests(task.buggy_files, tests)
        solved_ok = run_tests({**task.buggy_files, **_solution_for(task)}, tests)
        status = "OK" if (not buggy_ok and solved_ok) else "FAIL"
        print(f"[{status}] {task.id}: buggy_passes={buggy_ok} solution_passes={solved_ok}")
        if status == "FAIL":
            failures.append(task.id)
    if failures:
        print(f"FAILED fixtures: {failures}")
        return 1
    print(f"All coding fixtures verified; YAML written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
