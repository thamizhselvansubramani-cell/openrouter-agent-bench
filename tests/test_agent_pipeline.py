"""Tests for the benchmark runner, storage, reporting, and CLI wiring."""

from __future__ import annotations

import pathlib

import pytest
from typer.testing import CliRunner

from openrouter_agent_bench.agent.runner import (
    AttemptResult,
    RunConfig,
    run_attempt,
    run_task,
)
from openrouter_agent_bench.client.schemas import CompletionResponse, Usage
from openrouter_agent_bench.reporting.report import (
    plot_pass_rates,
    render_markdown,
    summarize_by_model,
    summarize_by_task,
    write_markdown_report,
)
from openrouter_agent_bench.storage.storage import ResultStore
from openrouter_agent_bench.tasks.schema import (
    ExactMatchGrader,
    TaskSpec,
    UnitTestGrader,
)


class FakeClient:
    """Stand-in for OpenRouterClient returning a scripted reply."""

    def __init__(self, content: str, *, raise_exc: Exception | None = None) -> None:
        self.content = content
        self.raise_exc = raise_exc
        self.calls: list[str] = []

    async def chat_completion(
        self,
        *,
        model: str,
        messages: object,
        temperature: object = 0.0,
        max_tokens: object = None,
        seed: object = None,
    ) -> CompletionResponse:
        self.calls.append(model)
        if self.raise_exc is not None:
            raise self.raise_exc
        return CompletionResponse(
            model=model,
            content=self.content,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.001),
            latency_s=0.25,
        )


def _exact_task() -> TaskSpec:
    return TaskSpec(
        id="capital",
        suite="coding",
        title="Capital",
        category="qa",
        difficulty=1,
        prompt="What is the capital of France?",
        grader=ExactMatchGrader(expected="paris"),
    )


def _unit_task() -> TaskSpec:
    return TaskSpec(
        id="add-one",
        suite="coding",
        title="Add one",
        category="algorithms",
        difficulty=1,
        prompt="Implement add_one.",
        target_file="mod.py",
        timeout_s=60,
        files={"mod.py": "def add_one(x):\n    return x\n"},
        grader=UnitTestGrader(
            tests={
                "test_mod.py": (
                    "from mod import add_one\n\n\n"
                    "def test_add_one():\n    assert add_one(1) == 2\n"
                )
            }
        ),
    )


async def test_run_attempt_exact_match_pass() -> None:
    client = FakeClient("The answer is PARIS.")
    result = await run_attempt(client, _exact_task(), "stealth/ox-alpha")
    assert result.passed is True
    assert result.score == 1.0
    assert result.total_tokens == 15
    assert result.cost == pytest.approx(0.001)
    assert result.fault is None


async def test_run_attempt_api_error_is_fault() -> None:
    client = FakeClient("", raise_exc=RuntimeError("boom"))
    result = await run_attempt(client, _exact_task(), "m")
    assert result.passed is False
    assert result.fault == "api_error"
    assert result.error is not None and "boom" in result.error


async def test_run_task_repeats() -> None:
    client = FakeClient("paris")
    results = await run_task(client, _exact_task(), "m", RunConfig(repeats=3))
    assert [r.attempt for r in results] == [1, 2, 3]
    assert all(r.passed for r in results)


async def test_run_attempt_unit_tests_pass() -> None:
    answer = "### FILE: mod.py\n```python\ndef add_one(x):\n    return x + 1\n```\n"
    client = FakeClient(answer)
    result = await run_attempt(client, _unit_task(), "m", RunConfig(sandbox_timeout_s=60))
    assert result.passed is True
    assert result.score == 1.0


def _sample_results() -> list[AttemptResult]:
    return [
        AttemptResult(task_id="t1", suite="coding", model="m1", passed=True, score=1.0,
                      total_tokens=100, cost=0.01, latency_s=1.0),
        AttemptResult(task_id="t2", suite="coding", model="m1", passed=False, score=0.0,
                      total_tokens=50, cost=0.005, latency_s=2.0),
        AttemptResult(task_id="t1", suite="coding", model="m2", passed=True, score=1.0,
                      total_tokens=80, cost=0.0, latency_s=0.5, fault=None),
    ]


def test_summaries() -> None:
    results = _sample_results()
    by_model = {s.key: s for s in summarize_by_model(results)}
    assert by_model["m1"].attempts == 2
    assert by_model["m1"].passed == 1
    assert by_model["m1"].pass_rate == pytest.approx(0.5)
    assert by_model["m2"].pass_rate == pytest.approx(1.0)
    by_task = {s.key: s for s in summarize_by_task(results)}
    assert by_task["t1"].attempts == 2


def test_render_and_write_report(tmp_path: pathlib.Path) -> None:
    results = _sample_results()
    md = render_markdown(summarize_by_model(results), title="By model")
    assert "| Group |" in md
    out = write_markdown_report(tmp_path / "r.md", results)
    assert out.exists()
    assert "By model" in out.read_text(encoding="utf-8")


def test_plot_pass_rates(tmp_path: pathlib.Path) -> None:
    png = plot_pass_rates(summarize_by_task(_sample_results()), tmp_path / "p.png")
    assert png.exists() and png.stat().st_size > 0


def test_result_store_roundtrip(tmp_path: pathlib.Path) -> None:
    store = ResultStore(tmp_path / "db.sqlite")
    run_id = store.start_run(model="m1", suite="coding", label="test",
                             config=RunConfig(repeats=2))
    for result in _sample_results():
        store.record(run_id, result)
    attempts = store.attempts(run_id)
    assert len(attempts) == 3
    assert store.latest_run() is not None
    assert store.latest_run().id == run_id  # type: ignore[union-attr]


def test_cli_validate_and_suites() -> None:
    from openrouter_agent_bench.cli.app import app

    runner = CliRunner()
    res_validate = runner.invoke(app, ["validate"])
    assert res_validate.exit_code == 0, res_validate.output
    assert "valid" in res_validate.output
    res_suites = runner.invoke(app, ["suites", "--suite", "coding"])
    assert res_suites.exit_code == 0, res_suites.output
    assert "coding" in res_suites.output


def test_cli_models() -> None:
    from openrouter_agent_bench.cli.app import app

    runner = CliRunner()
    res = runner.invoke(app, ["models"])
    assert res.exit_code == 0, res.output


def test_cli_run_rejects_unknown_model() -> None:
    from openrouter_agent_bench.cli.app import app

    runner = CliRunner()
    res = runner.invoke(app, ["run", "coding", "--model", "nope/does-not-exist"])
    assert res.exit_code == 1
