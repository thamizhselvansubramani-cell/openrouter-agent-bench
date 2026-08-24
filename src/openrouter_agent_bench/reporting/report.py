"""Aggregate benchmark attempts into summaries, tables, and plots.

Consumes either in-memory :class:`AttemptResult` objects (fresh from a run) or
persisted :class:`AttemptRow` rows (loaded from storage); both expose the same
attribute names, so the aggregation functions accept a small structural
protocol rather than a concrete type.
"""

from __future__ import annotations

import pathlib
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class AttemptLike(Protocol):
    """Structural view shared by ``AttemptResult`` and ``AttemptRow``."""

    task_id: str
    suite: str
    model: str
    passed: bool
    score: float
    fault: str | None
    latency_s: float
    cost: float | None
    total_tokens: int


@dataclass(frozen=True)
class GroupSummary:
    """Aggregated statistics for one grouping key (model, suite, or task)."""

    key: str
    attempts: int
    passed: int
    pass_rate: float
    mean_score: float
    total_cost: float
    mean_latency_s: float
    total_tokens: int
    faults: int

    def as_row(self) -> list[str]:
        """Render as a fixed-order list of display cells."""
        return [
            self.key,
            str(self.attempts),
            f"{self.passed}/{self.attempts}",
            f"{self.pass_rate * 100:.1f}%",
            f"{self.mean_score:.3f}",
            f"${self.total_cost:.4f}",
            f"{self.mean_latency_s:.2f}s",
            f"{self.total_tokens:,}",
            str(self.faults),
        ]


TABLE_HEADERS = [
    "Group",
    "N",
    "Passed",
    "Pass rate",
    "Mean score",
    "Cost",
    "Mean latency",
    "Tokens",
    "Faults",
]


def _summarize_group(key: str, items: list[AttemptLike]) -> GroupSummary:
    n = len(items)
    passed = sum(1 for a in items if a.passed)
    scores = [a.score for a in items]
    latencies = [a.latency_s for a in items]
    return GroupSummary(
        key=key,
        attempts=n,
        passed=passed,
        pass_rate=passed / n if n else 0.0,
        mean_score=statistics.fmean(scores) if scores else 0.0,
        total_cost=round(sum(a.cost or 0.0 for a in items), 6),
        mean_latency_s=statistics.fmean(latencies) if latencies else 0.0,
        total_tokens=sum(a.total_tokens for a in items),
        faults=sum(1 for a in items if a.fault),
    )


def _group_by(
    attempts: Iterable[AttemptLike], key: str
) -> list[GroupSummary]:
    buckets: dict[str, list[AttemptLike]] = {}
    for attempt in attempts:
        bucket_key = str(getattr(attempt, key))
        buckets.setdefault(bucket_key, []).append(attempt)
    return [_summarize_group(k, buckets[k]) for k in sorted(buckets)]


def summarize_by_model(attempts: Iterable[AttemptLike]) -> list[GroupSummary]:
    """One summary row per model."""
    return _group_by(attempts, "model")


def summarize_by_suite(attempts: Iterable[AttemptLike]) -> list[GroupSummary]:
    """One summary row per suite."""
    return _group_by(attempts, "suite")


def summarize_by_task(attempts: Iterable[AttemptLike]) -> list[GroupSummary]:
    """One summary row per task id."""
    return _group_by(attempts, "task_id")


def render_markdown(summaries: list[GroupSummary], *, title: str = "Results") -> str:
    """Render summaries as a GitHub-flavored Markdown table."""
    lines = [f"## {title}", "", "| " + " | ".join(TABLE_HEADERS) + " |"]
    lines.append("| " + " | ".join("---" for _ in TABLE_HEADERS) + " |")
    for summary in summaries:
        lines.append("| " + " | ".join(summary.as_row()) + " |")
    return "\n".join(lines) + "\n"


def render_text(summaries: list[GroupSummary]) -> str:
    """Render summaries as a monospace-aligned plain-text table."""
    rows = [TABLE_HEADERS, *[s.as_row() for s in summaries]]
    widths = [max(len(row[i]) for row in rows) for i in range(len(TABLE_HEADERS))]
    out = []
    for r, row in enumerate(rows):
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if r == 0:
            out.append("  ".join("-" * widths[i] for i in range(len(widths))))
    return "\n".join(out) + "\n"


def write_markdown_report(
    path: str | pathlib.Path,
    attempts: Iterable[AttemptLike],
    *,
    title: str = "openrouter-agent-bench report",
) -> pathlib.Path:
    """Write a multi-section Markdown report grouped by model, suite, and task."""
    items = list(attempts)
    sections = [
        f"# {title}",
        "",
        render_markdown(summarize_by_model(items), title="By model"),
        render_markdown(summarize_by_suite(items), title="By suite"),
        render_markdown(summarize_by_task(items), title="By task"),
    ]
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    return out


def plot_pass_rates(
    summaries: list[GroupSummary],
    path: str | pathlib.Path,
    *,
    title: str = "Pass rate by group",
) -> pathlib.Path:
    """Render a horizontal bar chart of pass rates to ``path`` (PNG).

    Uses the non-interactive ``Agg`` backend so it works headless.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    labels = [s.key for s in summaries]
    values = [s.pass_rate * 100 for s in summaries]
    height = max(1.5, 0.5 * len(labels) + 1)
    fig, ax = plt.subplots(figsize=(8, height))
    positions = range(len(labels))
    ax.barh(list(positions), values, color="#4c8bf5")
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pass rate (%)")
    ax.set_xlim(0, 100)
    ax.set_title(title)
    ax.invert_yaxis()
    for pos, value in zip(positions, values, strict=True):
        ax.text(min(value + 1, 96), pos, f"{value:.0f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


__all__ = [
    "TABLE_HEADERS",
    "AttemptLike",
    "GroupSummary",
    "plot_pass_rates",
    "render_markdown",
    "render_text",
    "summarize_by_model",
    "summarize_by_suite",
    "summarize_by_task",
    "write_markdown_report",
]
