"""Benchmark runner: execute tasks against a model and grade the replies.

The runner is deliberately transport-agnostic: it depends only on the small
async surface (:meth:`chat_completion`) exposed by
:class:`~openrouter_agent_bench.client.api.OpenRouterClient`, so tests can
inject a fake client without touching the network.

For ``unit_tests`` tasks a fresh :class:`SandboxWorkspace` is created and
seeded with both the task fixtures (``task.files``) and the hidden test files
(``grader.tests``) before grading; the submitted files are written over the
fixtures by the grader itself.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field

from openrouter_agent_bench.client.schemas import ChatMessage, CompletionResponse
from openrouter_agent_bench.evaluation.graders import GradeResult, build_prompt, grade_answer
from openrouter_agent_bench.sandbox.executor import SandboxExecutor, SandboxWorkspace
from openrouter_agent_bench.tasks.schema import TaskSpec, UnitTestGrader


class SupportsChatCompletion(Protocol):
    """Minimal async client surface the runner depends on."""

    async def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float | None = ...,
        max_tokens: int | None = ...,
        seed: int | None = ...,
    ) -> CompletionResponse: ...


@dataclass(frozen=True)
class RunConfig:
    """Knobs controlling a benchmark run."""

    temperature: float | None = 0.0
    max_tokens: int | None = None
    seed: int | None = None
    #: How many times to attempt each task (pass@k style repetition).
    repeats: int = 1
    #: Wall-clock cap for each sandboxed grading subprocess.
    sandbox_timeout_s: float = 60.0
    sandbox_memory_mb: int | None = 512


class AttemptResult(BaseModel):
    """The outcome of a single task attempt against one model."""

    task_id: str
    suite: str
    model: str
    #: 1-based attempt index within a repeated run.
    attempt: int = 1
    passed: bool = False
    score: float = 0.0
    detail: str = ""
    #: Operational fault tag (``api_error``, ``timeout``...) or ``None``.
    fault: str | None = None
    latency_s: float = 0.0
    cost: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    answer: str = ""
    error: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


ResultCallback = Callable[[AttemptResult], None]


def _grade(task: TaskSpec, answer: str, config: RunConfig) -> GradeResult:
    """Grade ``answer`` for ``task``, provisioning a sandbox when required."""
    grader = task.grader
    if isinstance(grader, UnitTestGrader):
        seed_files = {**task.files, **grader.tests}
        workspace = SandboxWorkspace.create(seed_files)
        sandbox = SandboxExecutor(
            timeout_s=config.sandbox_timeout_s,
            memory_mb=config.sandbox_memory_mb,
        )
        try:
            return grade_answer(task, answer, sandbox=sandbox, workspace=workspace)
        finally:
            workspace.cleanup()
    return grade_answer(task, answer)


async def run_attempt(
    client: SupportsChatCompletion,
    task: TaskSpec,
    model: str,
    config: RunConfig | None = None,
    *,
    attempt: int = 1,
) -> AttemptResult:
    """Run a single attempt of ``task`` against ``model`` and grade it."""
    cfg = config or RunConfig()
    prompt = build_prompt(task)
    messages = [ChatMessage(role="user", content=prompt)]
    base = AttemptResult(task_id=task.id, suite=task.suite, model=model, attempt=attempt)

    try:
        response = await client.chat_completion(
            model=model,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )
    except Exception as exc:
        return base.model_copy(
            update={"fault": "api_error", "error": f"{type(exc).__name__}: {exc}"}
        )

    answer = response.content or ""
    usage = response.usage
    try:
        grade = _grade(task, answer, cfg)
    except Exception as exc:
        return base.model_copy(
            update={
                "fault": "grader_error",
                "error": f"{type(exc).__name__}: {exc}",
                "answer": answer,
                "latency_s": usage_latency(response),
                "cost": usage.cost,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        )

    return base.model_copy(
        update={
            "passed": grade.passed,
            "score": grade.score,
            "detail": grade.detail,
            "fault": grade.fault,
            "latency_s": usage_latency(response),
            "cost": usage.cost,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "answer": answer,
        }
    )


def usage_latency(response: CompletionResponse) -> float:
    """Latency in seconds recorded on a completion response."""
    return round(response.latency_s, 4)


async def run_task(
    client: SupportsChatCompletion,
    task: TaskSpec,
    model: str,
    config: RunConfig | None = None,
    *,
    on_result: ResultCallback | None = None,
) -> list[AttemptResult]:
    """Run ``config.repeats`` attempts of a single task."""
    cfg = config or RunConfig()
    results: list[AttemptResult] = []
    for i in range(1, max(cfg.repeats, 1) + 1):
        result = await run_attempt(client, task, model, cfg, attempt=i)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


async def run_tasks(
    client: SupportsChatCompletion,
    tasks: Iterable[TaskSpec],
    model: str,
    config: RunConfig | None = None,
    *,
    on_result: ResultCallback | None = None,
) -> list[AttemptResult]:
    """Run every task in ``tasks`` sequentially, flattening all attempts."""
    cfg = config or RunConfig()
    results: list[AttemptResult] = []
    for task in tasks:
        results.extend(await run_task(client, task, model, cfg, on_result=on_result))
    return results


# Kept for callers that prefer to pass a bare coroutine factory.
RunTasks = Callable[..., Awaitable[list[AttemptResult]]]


__all__ = [
    "AttemptResult",
    "ResultCallback",
    "RunConfig",
    "SupportsChatCompletion",
    "run_attempt",
    "run_task",
    "run_tasks",
]
