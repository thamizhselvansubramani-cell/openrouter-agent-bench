"""Tests for result persistence: round-tripping, provenance, and migration."""

from __future__ import annotations

import pathlib
import sqlite3

from openrouter_agent_bench.agent.runner import AttemptResult, RunConfig
from openrouter_agent_bench.storage.storage import ResultStore


def _attempt(**over: object) -> AttemptResult:
    base = {
        "task_id": "t1",
        "suite": "coding",
        "model": "vendor/model",
        "passed": True,
        "score": 1.0,
        "answer": "```python\nx = 1\n```",
        "served_model": "vendor/model-quantized",
        "provider": "SomeProvider",
        "generation_id": "gen-123",
        "finish_reason": "stop",
        "detail": "1 passed",
    }
    base.update(over)
    return AttemptResult(**base)  # type: ignore[arg-type]


def _columns(db: pathlib.Path, table: str) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_raw_completion_survives_a_round_trip(tmp_path: pathlib.Path) -> None:
    """The reply must be recoverable, or an attempt cannot be re-graded."""
    db = tmp_path / "r.db"
    store = ResultStore(db)
    run_id = store.start_run(model="vendor/model", suite="coding")
    store.record(run_id, _attempt())

    (row,) = store.attempts(run_id)
    assert row.answer == "```python\nx = 1\n```"
    assert row.served_model == "vendor/model-quantized"
    assert row.provider == "SomeProvider"
    assert row.generation_id == "gen-123"
    assert row.finish_reason == "stop"


def test_detail_is_not_truncated(tmp_path: pathlib.Path) -> None:
    store = ResultStore(tmp_path / "r.db")
    run_id = store.start_run(model="m", suite="coding")
    long_detail = "traceback line\n" * 500
    store.record(run_id, _attempt(detail=long_detail))
    (row,) = store.attempts(run_id)
    assert row.detail == long_detail


def test_provenance_is_persisted_on_the_run(tmp_path: pathlib.Path) -> None:
    store = ResultStore(tmp_path / "r.db")
    run_id = store.start_run(
        model="m",
        suite="coding",
        config=RunConfig(repeats=3),
        provenance={
            "harness_version": "9.9.9",
            "git_sha": "abc123-dirty",
            "suite_hash": "deadbeef",
            "python_version": "3.12.0",
            "platform": "Linux-x-y",
            "unknown_key": "ignored",
        },
    )
    run = next(r for r in store.runs() if r.id == run_id)
    assert run.harness_version == "9.9.9"
    assert run.git_sha == "abc123-dirty"
    assert run.suite_hash == "deadbeef"


def test_record_many_persists_provenance(tmp_path: pathlib.Path) -> None:
    store = ResultStore(tmp_path / "r.db")
    run_id = store.start_run(model="m", suite="coding")
    store.record_many(run_id, [_attempt(task_id="a"), _attempt(task_id="b")])
    rows = store.attempts(run_id)
    assert {r.task_id for r in rows} == {"a", "b"}
    assert all(r.generation_id == "gen-123" for r in rows)


def test_migration_adds_columns_and_keeps_existing_rows(tmp_path: pathlib.Path) -> None:
    """A database written before these columns existed must still open."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE benchrun (
            id INTEGER PRIMARY KEY, label TEXT, model TEXT, suite TEXT,
            created_at TEXT, config_json TEXT
        );
        CREATE TABLE attemptrow (
            id INTEGER PRIMARY KEY, run_id INTEGER, task_id TEXT, suite TEXT,
            model TEXT, attempt INTEGER, passed BOOLEAN, score FLOAT,
            fault TEXT, latency_s FLOAT, cost FLOAT, prompt_tokens INTEGER,
            completion_tokens INTEGER, total_tokens INTEGER, detail TEXT, error TEXT
        );
        INSERT INTO benchrun VALUES (1, 'old', 'vendor/m', 'coding', '2020-01-01', '{}');
        INSERT INTO attemptrow VALUES
            (1, 1, 'legacy-task', 'coding', 'vendor/m', 1, 1, 1.0,
             NULL, 1.5, 0.0, 10, 20, 30, 'ok', NULL);
        """
    )
    conn.commit()
    conn.close()

    assert "answer" not in _columns(db, "attemptrow")

    store = ResultStore(db)  # opening runs the migration

    assert {"answer", "served_model", "provider", "generation_id"} <= _columns(
        db, "attemptrow"
    )
    assert {"git_sha", "suite_hash", "harness_version"} <= _columns(db, "benchrun")

    (row,) = store.attempts(1)
    assert row.task_id == "legacy-task"
    assert row.passed is True
    # non-nullable column backfilled with the model default, not NULL
    assert row.answer == ""
    assert row.served_model is None


def test_migration_is_idempotent(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "r.db"
    store = ResultStore(db)
    run_id = store.start_run(model="m", suite="coding")
    store.record(run_id, _attempt())
    reopened = ResultStore(db)
    assert len(reopened.attempts(run_id)) == 1
