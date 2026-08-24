"""Reporting: aggregate benchmark attempts into summaries, tables, and plots."""

from openrouter_agent_bench.reporting.report import (
    GroupSummary,
    plot_pass_rates,
    render_markdown,
    render_text,
    summarize_by_model,
    summarize_by_suite,
    summarize_by_task,
    write_markdown_report,
)

__all__ = [
    "GroupSummary",
    "plot_pass_rates",
    "render_markdown",
    "render_text",
    "summarize_by_model",
    "summarize_by_suite",
    "summarize_by_task",
    "write_markdown_report",
]
