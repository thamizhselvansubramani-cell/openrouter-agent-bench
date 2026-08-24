"""Durable storage for benchmark runs backed by SQLite via SQLModel.

A *run* groups the attempts produced by one invocation of ``bench run``
(one model over a set of tasks). Individual :class:`AttemptRow` records mirror
:class:`~openrouter_agent_bench.agent.runner.AttemptResult` so results survive
process exit and can be re-reported later.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Field, Session, SQLModel, create_engine, select

from openrouter_agent_bench.agent.runner import AttemptResult, RunConfig

DEFAULT_DB_PATH = "bench_results.db"


class BenchRun(SQLModel, table=True):
    """One benchmark invocation (a model over a batch of tasks)."""

    id: int | None = Field(default=None, primary_key=True)
    label: str = ""
    model: str
    suite: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    #: JSON-serialized :class:`RunConfig`.
    config_json: str = "{}"


class AttemptRow(SQLModel, table=True):
    """A single graded attempt belonging to a :class:`BenchRun`."""

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="benchrun.id", index=True)
    task_id: str = Field(index=True)
    suite: str = ""
    model: str = ""
    attempt: int = 1
    passed: bool = False
    score: float = 0.0
    fault: str | None = None
    latency_s: float = 0.0
    cost: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    detail: str = ""
    error: str | None = None


def _config_to_json(config: RunConfig | dict[str, Any] | None) -> str:
    if config is None:
        return "{}"
    if is_dataclass(config) and not isinstance(config, type):
        return json.dumps(asdict(config))
    if isinstance(config, dict):
        return json.dumps(config)
    return "{}"


class ResultStore:
    """Thin persistence facade over a SQLite database."""

    def __init__(self, path: str | pathlib.Path = DEFAULT_DB_PATH) -> None:
        self.path = pathlib.Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self.path}")
        SQLModel.metadata.create_all(self._engine)

    def start_run(
        self,
        *,
        model: str,
        suite: str = "",
        label: str = "",
        config: RunConfig | dict[str, Any] | None = None,
    ) -> int:
        """Create a new run row and return its id."""
        run = BenchRun(
            model=model,
            suite=suite,
            label=label,
            config_json=_config_to_json(config),
        )
        with Session(self._engine) as session:
            session.add(run)
            session.commit()
            session.refresh(run)
            assert run.id is not None
            return run.id

    def record(self, run_id: int, result: AttemptResult) -> None:
        """Persist a single attempt result under ``run_id``."""
        row = AttemptRow(
            run_id=run_id,
            task_id=result.task_id,
            suite=result.suite,
            model=result.model,
            attempt=result.attempt,
            passed=result.passed,
            score=result.score,
            fault=result.fault,
            latency_s=result.latency_s,
            cost=result.cost,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            detail=result.detail[:2000],
            error=result.error,
        )
        with Session(self._engine) as session:
            session.add(row)
            session.commit()

    def record_many(self, run_id: int, results: Iterable[AttemptResult]) -> None:
        """Persist several attempt results in one transaction."""
        with Session(self._engine) as session:
            for result in results:
                session.add(
                    AttemptRow(
                        run_id=run_id,
                        task_id=result.task_id,
                        suite=result.suite,
                        model=result.model,
                        attempt=result.attempt,
                        passed=result.passed,
                        score=result.score,
                        fault=result.fault,
                        latency_s=result.latency_s,
                        cost=result.cost,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.total_tokens,
                        detail=result.detail[:2000],
                        error=result.error,
                    )
                )
            session.commit()

    def runs(self) -> list[BenchRun]:
        """All recorded runs, newest first."""
        with Session(self._engine) as session:
            rows = session.exec(select(BenchRun)).all()
        return sorted(rows, key=lambda r: (r.created_at, r.id or 0), reverse=True)

    def latest_run(self) -> BenchRun | None:
        runs = self.runs()
        return runs[0] if runs else None

    def attempts(self, run_id: int) -> list[AttemptRow]:
        """All attempts belonging to ``run_id``."""
        with Session(self._engine) as session:
            statement = select(AttemptRow).where(AttemptRow.run_id == run_id)
            return list(session.exec(statement).all())


__all__ = [
    "DEFAULT_DB_PATH",
    "AttemptRow",
    "BenchRun",
    "ResultStore",
]
