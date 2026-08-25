"""Cross-run model comparison: aggregate several result databases into one view.

``bench report`` summarizes a single run. Comparing models means reading many
runs -- often written to separate databases so concurrent writers never
contend -- and lining them up. :func:`aggregate` produces a plain dict
(JSON-serializable, so it can be published as the data behind a figure) and
:func:`plot_comparison` renders it.

Every figure in the project is produced from a committed database by this
module, so any published number can be traced back to a command.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable, Sequence
from typing import Any

from openrouter_agent_bench.storage.storage import ResultStore

__all__ = [
    "aggregate",
    "plot_comparison",
    "write_comparison",
]

# Categorical palette. The slot *ordering* is the colour-blind-safety
# mechanism, not cosmetic: these hues clear CVD and normal-vision separation
# floors in this order, so suites must be assigned in sequence.
_SURFACE = "#fcfcfb"
_PAGE = "#f9f9f7"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASE = "#c3c2b7"
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def _rate(passed: int, n: int) -> float | None:
    return (passed / n) if n else None


def aggregate(db_paths: Iterable[str | pathlib.Path]) -> dict[str, Any]:
    """Combine every run in ``db_paths`` into a comparison structure.

    Attempts carrying a ``fault`` are excluded from pass rates -- an HTTP 429
    is an infrastructure outcome, not a wrong answer -- but they are counted
    and reported separately so a run's coverage is never silently overstated.
    """
    rows = []
    for path in db_paths:
        store = ResultStore(path)
        for run in store.runs():
            if run.id is not None:
                rows.extend(store.attempts(run.id))

    suites = sorted({r.suite for r in rows if r.suite})
    total_tasks = len({r.task_id for r in rows})

    models: list[dict[str, Any]] = []
    for model in sorted({r.model for r in rows}):
        mine = [r for r in rows if r.model == model]
        graded = [r for r in mine if not r.fault]
        passed = sum(1 for r in graded if r.passed)
        per_suite = {}
        for suite in suites:
            sub = [r for r in graded if r.suite == suite]
            hits = sum(1 for r in sub if r.passed)
            per_suite[suite] = {
                "attempts": len(sub),
                "passed": hits,
                "pass_rate": _rate(hits, len(sub)),
            }
        seen = len({r.task_id for r in graded})
        models.append(
            {
                "model": model,
                "attempts_total": len(mine),
                "graded": len(graded),
                "faults": len(mine) - len(graded),
                "fault_kinds": sorted({r.fault for r in mine if r.fault}),
                "passed": passed,
                "pass_rate": _rate(passed, len(graded)),
                "tasks_seen": seen,
                "complete": seen == total_tasks,
                "mean_latency_s": (
                    sum(r.latency_s for r in graded) / len(graded) if graded else None
                ),
                "total_tokens": sum(r.total_tokens for r in graded),
                "total_cost": sum((r.cost or 0.0) for r in graded),
                # Routed endpoints resolve to a different model per call, so
                # record what actually served the request; without this a score
                # against a router is not attributable to anything.
                "served_models": sorted({r.served_model for r in mine if r.served_model}),
                "providers": sorted({r.provider for r in mine if r.provider}),
                "per_suite": per_suite,
            }
        )
    models.sort(key=lambda m: (m["pass_rate"] is None, -(m["pass_rate"] or 0.0)))

    tasks: dict[str, Any] = {}
    for row in (r for r in rows if not r.fault):
        entry = tasks.setdefault(row.task_id, {"suite": row.suite, "n": 0, "passed": 0})
        entry["n"] += 1
        entry["passed"] += int(row.passed)
    for entry in tasks.values():
        entry["pass_rate"] = entry["passed"] / entry["n"]

    return {
        "suites": suites,
        "total_tasks": total_tasks,
        "graded_attempts": sum(m["graded"] for m in models),
        "faulted_attempts": sum(m["faults"] for m in models),
        "total_cost": sum(m["total_cost"] for m in models),
        "models": models,
        "tasks": dict(sorted(tasks.items(), key=lambda kv: kv[1]["pass_rate"])),
    }


def _short(model_id: str) -> str:
    """Compact display label, keeping bare-router ids like ``openrouter/free``."""
    if "/" not in model_id or model_id.endswith("/free"):
        return model_id
    return model_id.split("/")[-1].replace(":free", "")


def _style(ax: Any, *, grid: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(_BASE)
    ax.spines["bottom"].set_color(_BASE)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis="x", color=_GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)


def plot_comparison(
    data: dict[str, Any],
    path: str | pathlib.Path,
    *,
    title: str = "Model comparison",
    subtitle: str = "",
) -> pathlib.Path:
    """Render the four-panel comparison figure to ``path`` (PNG).

    Incomplete runs are hatched and labelled with their coverage, and sorted
    below the complete ones, because a pass rate over a subset of tasks is not
    comparable to one over all of them and should not head the chart.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "figure.facecolor": _PAGE,
            "axes.facecolor": _SURFACE,
            "text.color": _INK,
            "axes.labelcolor": _INK2,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "axes.edgecolor": _BASE,
        }
    )

    suites: Sequence[str] = data["suites"]
    colour = {s: _SERIES[i % len(_SERIES)] for i, s in enumerate(suites)}
    total = data["total_tasks"]
    models = data["models"]
    rated = [m for m in models if m["pass_rate"] is not None]
    unrated = [m for m in models if m["pass_rate"] is None]

    fig = plt.figure(figsize=(15.5, 10.5))
    gs = fig.add_gridspec(
        2, 2, hspace=0.42, wspace=0.30, top=0.85, bottom=0.10, left=0.155, right=0.975
    )
    fig.suptitle(
        title, x=0.055, y=0.965, ha="left", fontsize=19, fontweight="bold", color=_INK
    )
    if subtitle:
        fig.text(0.055, 0.925, subtitle, ha="left", fontsize=10.5, color=_INK2)
    fig.text(
        0.055,
        0.898,
        f"Striped bars are incomplete runs, labelled with tasks graded of {total}; "
        "their rates are not comparable to the solid bars.",
        ha="left",
        fontsize=9.5,
        color=_MUTED,
        style="italic",
    )

    def coverage(m: dict[str, Any]) -> str:
        return "" if m["complete"] else f"  ·{m['tasks_seen']}/{total}"

    # -- overall pass rate; complete runs sort last (drawn at the top)
    ax = fig.add_subplot(gs[0, 0])
    order = unrated + sorted(rated, key=lambda m: (m["complete"], m["pass_rate"]))
    for y, m in enumerate(order):
        if m["pass_rate"] is None:
            ax.text(
                1.0,
                y,
                "no graded attempts",
                va="center",
                fontsize=9,
                color=_MUTED,
                style="italic",
            )
            continue
        pct = m["pass_rate"] * 100
        ax.barh(
            y,
            pct,
            height=0.62,
            color=_SERIES[0],
            zorder=3,
            hatch="" if m["complete"] else "////",
            edgecolor=_SURFACE,
            linewidth=0,
        )
        ax.text(
            pct + 1.4,
            y,
            f"{pct:.0f}%   ({m['passed']}/{m['graded']})",
            va="center",
            fontsize=10,
            color=_INK,
            zorder=4,
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [_short(m["model"]) + coverage(m) for m in order], fontsize=10, color=_INK
    )
    ax.set_xlim(0, max((m["pass_rate"] * 100 for m in rated), default=100) * 1.55 or 100)
    ax.set_xlabel("Pass rate (%)", fontsize=10)
    ax.set_title(
        "Overall pass rate", loc="left", fontsize=13, fontweight="bold", color=_INK, pad=10
    )
    _style(ax)

    # -- pass rate by suite
    ax = fig.add_subplot(gs[0, 1])
    grp = [m for m in reversed(order) if m["pass_rate"] is not None]
    n_s = max(len(suites), 1)
    bar_h = 0.72 / n_s
    for gi, m in enumerate(grp):
        for si, suite in enumerate(suites):
            cell = m["per_suite"][suite]
            ypos = gi + (si - (n_s - 1) / 2) * bar_h
            if cell["pass_rate"] is None:
                ax.text(
                    1.5, ypos, "no data", va="center", fontsize=8, color=_MUTED,
                    style="italic",
                )
                continue
            val = cell["pass_rate"] * 100
            ax.barh(
                ypos,
                val,
                height=bar_h * 0.86,
                color=colour[suite],
                edgecolor=_SURFACE,
                linewidth=1.2,
                zorder=3,
            )
            ax.text(
                val + 1.5,
                ypos,
                f"{val:.0f}%  ({cell['passed']}/{cell['attempts']})",
                va="center",
                fontsize=8.5,
                color=_INK,
                zorder=4,
            )
    ax.set_yticks(range(len(grp)))
    ax.set_yticklabels([_short(m["model"]) for m in grp], fontsize=10, color=_INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Pass rate within suite (%)", fontsize=10)
    ax.set_title(
        "Pass rate by suite", loc="left", fontsize=13, fontweight="bold", color=_INK, pad=10
    )
    ax.legend(
        handles=[mpatches.Patch(color=colour[s], label=s) for s in suites],
        loc="lower right",
        frameon=False,
        fontsize=9.5,
        labelcolor=_INK2,
    )
    _style(ax)

    # -- mean latency
    ax = fig.add_subplot(gs[1, 0])
    lat = sorted(
        (m for m in rated if m["mean_latency_s"]), key=lambda m: m["mean_latency_s"]
    )
    for y, m in enumerate(lat):
        ax.barh(
            y,
            m["mean_latency_s"],
            height=0.62,
            color=_SERIES[0],
            zorder=3,
            hatch="" if m["complete"] else "////",
            edgecolor=_SURFACE,
            linewidth=0,
        )
        ax.text(
            m["mean_latency_s"] + 0.8,
            y,
            f"{m['mean_latency_s']:.1f}s",
            va="center",
            fontsize=10,
            color=_INK,
            zorder=4,
        )
    ax.set_yticks(range(len(lat)))
    ax.set_yticklabels(
        [_short(m["model"]) + coverage(m) for m in lat], fontsize=10, color=_INK
    )
    ax.set_xlim(0, max((m["mean_latency_s"] for m in lat), default=1) * 1.22)
    ax.set_xlabel("Mean latency per task (s)", fontsize=10)
    ax.set_title("Mean latency", loc="left", fontsize=13, fontweight="bold", color=_INK, pad=10)
    _style(ax)

    # -- per-task difficulty
    ax = fig.add_subplot(gs[1, 1])
    items = sorted(data["tasks"].items(), key=lambda kv: (kv[1]["pass_rate"], kv[0]))
    for y, (_tid, t) in enumerate(items):
        val = t["pass_rate"] * 100
        ax.barh(
            y,
            val,
            height=0.66,
            color=colour.get(t["suite"], _SERIES[0]),
            edgecolor=_SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            val + 1.5,
            y,
            f"{t['passed']}/{t['n']}",
            va="center",
            fontsize=8.5,
            color=_INK,
            zorder=4,
        )
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([tid for tid, _ in items], fontsize=8.2, color=_INK)
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of attempts that passed (%)", fontsize=10)
    ax.set_title("Task difficulty", loc="left", fontsize=13, fontweight="bold", color=_INK, pad=10)
    ax.legend(
        handles=[mpatches.Patch(color=colour[s], label=s) for s in suites],
        loc="lower right",
        frameon=False,
        fontsize=9,
        labelcolor=_INK2,
    )
    _style(ax)

    fig.text(
        0.055,
        0.035,
        f"{data['graded_attempts']} graded attempts, {data['faulted_attempts']} faults "
        f"· total cost ${data['total_cost']:.4f}",
        ha="left",
        fontsize=9,
        color=_MUTED,
    )
    fig.savefig(out, dpi=150, facecolor=_PAGE)
    plt.close(fig)
    return out


def write_comparison(data: dict[str, Any], path: str | pathlib.Path) -> pathlib.Path:
    """Write the aggregate structure as indented JSON."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out
