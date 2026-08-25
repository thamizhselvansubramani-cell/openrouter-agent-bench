"""Tests for cross-run aggregation and the comparison figure."""

from __future__ import annotations

import pathlib

from openrouter_agent_bench.agent.runner import AttemptResult
from openrouter_agent_bench.reporting.compare import (
    aggregate,
    plot_comparison,
    write_comparison,
)
from openrouter_agent_bench.storage.storage import ResultStore


def _db(
    tmp_path: pathlib.Path,
    name: str,
    model: str,
    attempts: list[AttemptResult],
) -> pathlib.Path:
    path = tmp_path / f"{name}.db"
    store = ResultStore(path)
    run_id = store.start_run(model=model, suite="coding")
    store.record_many(run_id, attempts)
    return path


def _a(task: str, *, model: str, passed: bool = False, **over: object) -> AttemptResult:
    base: dict[str, object] = {
        "task_id": task,
        "suite": "coding",
        "model": model,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "latency_s": 2.0,
    }
    base.update(over)
    return AttemptResult(**base)  # type: ignore[arg-type]


def test_aggregate_combines_multiple_databases(tmp_path: pathlib.Path) -> None:
    a = _db(tmp_path, "a", "m/a", [_a("t1", model="m/a", passed=True), _a("t2", model="m/a")])
    b = _db(tmp_path, "b", "m/b", [_a("t1", model="m/b"), _a("t2", model="m/b")])

    data = aggregate([a, b])

    assert data["total_tasks"] == 2
    assert [m["model"] for m in data["models"]] == ["m/a", "m/b"]
    assert data["models"][0]["pass_rate"] == 0.5
    assert data["models"][1]["pass_rate"] == 0.0


def test_faulted_attempts_are_excluded_from_pass_rate_but_counted(
    tmp_path: pathlib.Path,
) -> None:
    """A 429 is an infrastructure outcome, not a wrong answer."""
    db = _db(
        tmp_path,
        "a",
        "m/a",
        [
            _a("t1", model="m/a", passed=True),
            _a("t2", model="m/a", fault="api_error"),
        ],
    )
    (model,) = aggregate([db])["models"]
    assert model["graded"] == 1
    assert model["faults"] == 1
    assert model["pass_rate"] == 1.0
    assert model["fault_kinds"] == ["api_error"]


def test_coverage_is_reported_against_the_union_of_tasks(tmp_path: pathlib.Path) -> None:
    """A model that skipped tasks must not be marked complete."""
    a = _db(tmp_path, "a", "m/a", [_a("t1", model="m/a"), _a("t2", model="m/a")])
    b = _db(tmp_path, "b", "m/b", [_a("t1", model="m/b")])

    data = aggregate([a, b])
    by_id = {m["model"]: m for m in data["models"]}

    assert by_id["m/a"]["complete"] is True
    assert by_id["m/b"]["complete"] is False
    assert by_id["m/b"]["tasks_seen"] == 1


def test_a_model_with_only_faults_has_no_pass_rate(tmp_path: pathlib.Path) -> None:
    db = _db(tmp_path, "a", "m/a", [_a("t1", model="m/a", fault="api_error")])
    (model,) = aggregate([db])["models"]
    assert model["pass_rate"] is None
    assert model["mean_latency_s"] is None


def test_unrated_models_sort_last(tmp_path: pathlib.Path) -> None:
    a = _db(tmp_path, "a", "m/a", [_a("t1", model="m/a", fault="api_error")])
    b = _db(tmp_path, "b", "m/b", [_a("t1", model="m/b", passed=True)])
    assert [m["model"] for m in aggregate([a, b])["models"]] == ["m/b", "m/a"]


def test_routed_endpoints_record_every_served_model(tmp_path: pathlib.Path) -> None:
    """A router's score is only interpretable alongside what actually served it."""
    db = _db(
        tmp_path,
        "a",
        "openrouter/free",
        [
            _a("t1", model="openrouter/free", served_model="vendor/x"),
            _a("t2", model="openrouter/free", served_model="vendor/y"),
        ],
    )
    (model,) = aggregate([db])["models"]
    assert model["served_models"] == ["vendor/x", "vendor/y"]


def test_task_difficulty_is_aggregated_across_models(tmp_path: pathlib.Path) -> None:
    a = _db(tmp_path, "a", "m/a", [_a("hard", model="m/a"), _a("easy", model="m/a", passed=True)])
    b = _db(tmp_path, "b", "m/b", [_a("hard", model="m/b"), _a("easy", model="m/b", passed=True)])
    tasks = aggregate([a, b])["tasks"]
    assert tasks["hard"]["pass_rate"] == 0.0
    assert tasks["easy"]["pass_rate"] == 1.0
    # hardest first
    assert next(iter(tasks)) == "hard"


def test_aggregate_on_empty_database(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "empty.db"
    ResultStore(db)
    data = aggregate([db])
    assert data["models"] == []
    assert data["total_tasks"] == 0


def test_plot_and_json_are_written(tmp_path: pathlib.Path) -> None:
    a = _db(
        tmp_path,
        "a",
        "m/a",
        [_a("t1", model="m/a", passed=True), _a("t2", model="m/a", fault="api_error")],
    )
    b = _db(tmp_path, "b", "m/b", [_a("t1", model="m/b"), _a("t2", model="m/b")])
    data = aggregate([a, b])

    png = plot_comparison(data, tmp_path / "out" / "fig.png", title="T", subtitle="S")
    assert png.is_file()
    assert png.stat().st_size > 1000

    js = write_comparison(data, tmp_path / "out" / "data.json")
    assert js.is_file()
    assert '"total_tasks": 2' in js.read_text(encoding="utf-8")
