"""``bench`` command-line interface.

Subcommands:
- ``bench models``   — list the model catalog.
- ``bench suites``   — list loaded suites and their tasks.
- ``bench validate`` — validate every suite/task file.
- ``bench run``      — run a suite (or single task) against a model, store
  results, and print a summary.
- ``bench report``   — re-report a stored run (table, Markdown, and/or plot).
- ``bench web``      — launch the FastAPI dashboard.
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import typer
from rich.console import Console
from rich.table import Table

from openrouter_agent_bench.agent.runner import AttemptResult, RunConfig, run_task
from openrouter_agent_bench.client.api import OpenRouterClient
from openrouter_agent_bench.config import get_settings, load_env_file
from openrouter_agent_bench.evaluation.graders import build_prompt
from openrouter_agent_bench.evaluation.regrade import (
    POLICIES,
    regrade_attempt,
    summarize_regrade,
)
from openrouter_agent_bench.models.registry import ModelRegistry, ModelSpec
from openrouter_agent_bench.provenance import run_provenance
from openrouter_agent_bench.reporting.compare import (
    aggregate,
    plot_comparison,
    write_comparison,
)
from openrouter_agent_bench.reporting.report import (
    TABLE_HEADERS,
    GroupSummary,
    plot_pass_rates,
    summarize_by_model,
    summarize_by_suite,
    summarize_by_task,
    write_markdown_report,
)
from openrouter_agent_bench.storage.storage import DEFAULT_DB_PATH, ResultStore
from openrouter_agent_bench.tasks.loader import (
    TaskValidationError,
    default_suites_root,
    load_suites,
)
from openrouter_agent_bench.tasks.schema import TaskSpec, UnitTestGrader

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="openrouter-agent-bench: benchmark OpenRouter models.",
)
console = Console()


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _load_registry() -> ModelRegistry:
    settings = get_settings()
    path = settings.models_file or _repo_root() / "models.yaml"
    return ModelRegistry.load(path)


def _suites_root() -> pathlib.Path:
    return get_settings().suites_dir or default_suites_root(_repo_root())


def _pricing_map(registry: ModelRegistry) -> dict[str, tuple[float, float]]:
    return {
        spec.id: (spec.pricing.prompt_per_million, spec.pricing.completion_per_million)
        for spec in registry.all()
    }


def _render_summary(summaries: list[GroupSummary], title: str) -> None:
    table = Table(title=title, header_style="bold cyan")
    for header in TABLE_HEADERS:
        table.add_column(header)
    for summary in summaries:
        table.add_row(*summary.as_row())
    console.print(table)


@app.command()
def models(
    all_models: bool = typer.Option(
        False, "--all", help="Include paid models (default: free models only)."
    ),
) -> None:
    """List the registered model catalog."""
    load_env_file(_repo_root())
    try:
        registry = _load_registry()
    except (OSError, ValueError) as exc:
        console.print(f"[red]Failed to load model catalog:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    specs = registry.all() if all_models else registry.free()
    table = Table(title="Models", header_style="bold cyan")
    for col in ("ID", "Name", "Context", "Max out", "Vision", "Tools", "Free"):
        table.add_column(col)
    for spec in specs:
        table.add_row(
            spec.id,
            spec.display_name,
            f"{spec.context_window:,}",
            f"{spec.max_output_tokens:,}",
            "yes" if spec.supports_vision else "no",
            "yes" if spec.supports_tools else "no",
            "[green]FREE[/green]" if spec.is_free else "paid",
        )
    console.print(table)


@app.command()
def suites(
    suite: str | None = typer.Option(None, "--suite", "-s", help="Only this suite."),
) -> None:
    """List loaded suites and their tasks."""
    load_env_file(_repo_root())
    try:
        loaded = load_suites(_suites_root(), names=[suite] if suite else None)
    except TaskValidationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    for name, s in loaded.items():
        table = Table(title=f"{name} ({len(s.tasks)} tasks)", header_style="bold cyan")
        for col in ("Task", "Category", "Difficulty", "Grader"):
            table.add_column(col)
        for task in s.tasks:
            table.add_row(
                task.id, task.category, "★" * task.difficulty, task.grader.type
            )
        console.print(table)


@app.command()
def validate() -> None:
    """Validate every suite and task file."""
    load_env_file(_repo_root())
    try:
        loaded = load_suites(_suites_root())
    except TaskValidationError as exc:
        console.print(f"[red]Validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    total_tasks = sum(len(s.tasks) for s in loaded.values())
    console.print(
        f"[green]OK[/green] — {len(loaded)} suite(s), {total_tasks} task(s) valid."
    )


def _select_tasks(
    suite: str, task_id: str | None, limit: int | None
) -> list[TaskSpec]:
    loaded = load_suites(_suites_root(), names=[suite])
    tasks = loaded[suite].tasks
    if task_id:
        tasks = [t for t in tasks if t.id == task_id]
        if not tasks:
            raise TaskValidationError(f"task {task_id!r} not found in suite {suite!r}")
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def _resolve_model(registry: ModelRegistry, model: str) -> ModelSpec:
    settings = get_settings()
    if not registry.has(model):
        console.print(f"[red]Unknown model:[/red] {model}")
        console.print("Run [bold]bench models --all[/bold] to see available ids.")
        raise typer.Exit(code=1)
    spec = registry.get(model)
    if settings.free_models_only and not spec.is_free:
        console.print(
            f"[red]{model} is a paid model[/red] and OAB_FREE_MODELS_ONLY is set. "
            "Set OAB_FREE_MODELS_ONLY=0 to allow paid models."
        )
        raise typer.Exit(code=1)
    return spec


@app.command()
def run(
    suite: str = typer.Argument(..., help="Suite name to run (e.g. coding)."),
    model: str = typer.Option(..., "--model", "-m", help="Model id from the catalog."),
    task: str | None = typer.Option(None, "--task", "-t", help="Run one task id."),
    repeats: int = typer.Option(1, "--repeats", "-n", min=1, help="Attempts per task."),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Max output tokens."),
    limit: int | None = typer.Option(None, "--limit", help="Only the first N tasks."),
    db: str = typer.Option(DEFAULT_DB_PATH, "--db", help="SQLite results database path."),
    label: str = typer.Option("", "--label", help="Optional label for this run."),
    no_store: bool = typer.Option(False, "--no-store", help="Do not persist results."),
) -> None:
    """Run a suite (or single task) against a model and grade the replies."""
    load_env_file(_repo_root())
    settings = get_settings()
    if not settings.openrouter_api_key:
        console.print("[red]OPENROUTER_API_KEY is not set.[/red] Add it to your .env.")
        raise typer.Exit(code=1)

    try:
        registry = _load_registry()
    except (OSError, ValueError) as exc:
        console.print(f"[red]Failed to load model catalog:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _resolve_model(registry, model)

    try:
        tasks = _select_tasks(suite, task, limit)
    except TaskValidationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not tasks:
        console.print("[yellow]No tasks selected.[/yellow]")
        raise typer.Exit(code=0)

    config = RunConfig(
        temperature=temperature, max_tokens=max_tokens, repeats=repeats
    )
    store: ResultStore | None = None
    run_id: int | None = None
    if not no_store:
        store = ResultStore(db)
        run_id = store.start_run(
            model=model,
            suite=suite,
            label=label,
            config=config,
            provenance=run_provenance(
                task_ids_and_prompts=[(spec.id, build_prompt(spec)) for spec in tasks],
                repo_root=_repo_root(),
            ),
        )

    results = asyncio.run(
        _run_all(
            registry=registry,
            settings_api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            tasks=tasks,
            model=model,
            config=config,
            store=store,
            run_id=run_id,
        )
    )

    console.print()
    _render_summary(summarize_by_model(results), "Summary by model")
    _render_summary(summarize_by_task(results), "Summary by task")
    if store is not None and run_id is not None:
        console.print(f"[dim]Stored run #{run_id} in {db}[/dim]")


async def _run_all(
    *,
    registry: ModelRegistry,
    settings_api_key: str,
    base_url: str,
    tasks: list[TaskSpec],
    model: str,
    config: RunConfig,
    store: ResultStore | None,
    run_id: int | None,
) -> list[AttemptResult]:
    results: list[AttemptResult] = []
    client = OpenRouterClient(
        api_key=settings_api_key,
        base_url=base_url,
        pricing=_pricing_map(registry),
    )

    def _on_result(result: AttemptResult) -> None:
        mark = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        if result.fault:
            mark = f"[yellow]{result.fault}[/yellow]"
        console.print(
            f"  {mark} {result.task_id} "
            f"(score={result.score:.2f}, {result.latency_s:.1f}s)"
        )
        if store is not None and run_id is not None:
            store.record(run_id, result)

    try:
        for task in tasks:
            console.print(f"[bold]{task.id}[/bold] → {model}")
            attempts = await run_task(client, task, model, config, on_result=_on_result)
            results.extend(attempts)
    finally:
        await client.aclose()
    return results


@app.command()
def report(
    run_id: int | None = typer.Option(
        None, "--run", help="Run id to report (default: latest)."
    ),
    db: str = typer.Option(DEFAULT_DB_PATH, "--db", help="SQLite results database path."),
    markdown: str | None = typer.Option(
        None, "--markdown", help="Write a Markdown report to this path."
    ),
    plot: str | None = typer.Option(
        None, "--plot", help="Write a pass-rate bar chart PNG to this path."
    ),
) -> None:
    """Re-report a stored run as a table, Markdown, and/or plot."""
    store = ResultStore(db)
    if run_id is None:
        latest = store.latest_run()
        if latest is None:
            console.print("[yellow]No runs stored yet.[/yellow]")
            raise typer.Exit(code=0)
        run_id = latest.id
    assert run_id is not None
    attempts = store.attempts(run_id)
    if not attempts:
        console.print(f"[yellow]Run #{run_id} has no attempts.[/yellow]")
        raise typer.Exit(code=0)

    _render_summary(summarize_by_model(attempts), f"Run #{run_id} — by model")
    _render_summary(summarize_by_suite(attempts), f"Run #{run_id} — by suite")
    _render_summary(summarize_by_task(attempts), f"Run #{run_id} — by task")

    if markdown:
        out = write_markdown_report(markdown, attempts, title=f"Run #{run_id}")
        console.print(f"[green]Markdown report written:[/green] {out}")
    if plot:
        out = plot_pass_rates(
            summarize_by_task(attempts), plot, title=f"Run #{run_id} pass rate"
        )
        console.print(f"[green]Plot written:[/green] {out}")


@app.command()
def compare(
    dbs: list[str] = typer.Argument(
        ..., help="Result databases to combine (one per model is typical)."
    ),
    out_json: str = typer.Option(
        "reports/results.json", "--out-json", help="Where to write aggregate JSON."
    ),
    out_plot: str = typer.Option(
        "reports/model-comparison.png", "--out-plot", help="Where to write the figure."
    ),
    title: str = typer.Option("Model comparison", "--title", help="Figure title."),
    subtitle: str = typer.Option("", "--subtitle", help="Figure subtitle."),
) -> None:
    """Aggregate several result databases into one comparison table and figure.

    Regenerates the published data and chart from stored results, so every
    reported number traces back to this command.
    """
    missing = [d for d in dbs if not pathlib.Path(d).is_file()]
    if missing:
        console.print(f"[red]No such database:[/red] {', '.join(missing)}")
        raise typer.Exit(code=1)

    data = aggregate(dbs)
    if not data["models"]:
        console.print("[yellow]No attempts found in the given databases.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="Model comparison", header_style="bold cyan")
    for header in ("Model", "Graded", "Passed", "Pass rate", "Mean latency", "Faults", "Coverage"):
        table.add_column(header, justify="right" if header != "Model" else "left")
    total = data["total_tasks"]
    for m in data["models"]:
        rate = "n/a" if m["pass_rate"] is None else f"{m['pass_rate'] * 100:.1f}%"
        lat = "n/a" if m["mean_latency_s"] is None else f"{m['mean_latency_s']:.2f}s"
        table.add_row(
            m["model"],
            str(m["graded"]),
            f"{m['passed']}/{m['graded']}",
            rate,
            lat,
            str(m["faults"]),
            "complete" if m["complete"] else f"{m['tasks_seen']}/{total}",
        )
    console.print(table)

    incomplete = [m for m in data["models"] if not m["complete"]]
    if incomplete:
        console.print(
            f"[yellow]{len(incomplete)} of {len(data['models'])} models did not cover "
            f"all {total} tasks; their pass rates are not comparable.[/yellow]"
        )
    routed = [m for m in data["models"] if len(m["served_models"]) > 1]
    for m in routed:
        console.print(
            f"[yellow]{m['model']} was served by {len(m['served_models'])} different "
            f"models[/yellow] ({', '.join(m['served_models'])}); its score is not "
            "attributable to a single model."
        )

    console.print(f"[green]Aggregate written:[/green] {write_comparison(data, out_json)}")
    console.print(
        "[green]Figure written:[/green] "
        f"{plot_comparison(data, out_plot, title=title, subtitle=subtitle)}"
    )


@app.command()
def regrade(
    dbs: list[str] = typer.Argument(
        ..., help="Result databases holding stored completions."
    ),
    policies: str = typer.Option(
        ",".join(POLICIES), "--policies", help="Comma-separated extraction policies."
    ),
    suite: str = typer.Option("coding", "--suite", help="Suite to regrade."),
    out_json: str | None = typer.Option(
        None, "--out-json", help="Write the full regrade report here."
    ),
) -> None:
    """Re-score stored completions under alternative extraction policies.

    Measures how much of a unit_tests score reflects submission formatting
    rather than correctness. Makes no API calls: it replays the stored replies
    against the hidden tests, so it is free to run and repeatable.
    """
    wanted = [p.strip() for p in policies.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in POLICIES]
    if unknown:
        console.print(f"[red]Unknown policy:[/red] {', '.join(unknown)}")
        console.print(f"Available: {', '.join(POLICIES)}")
        raise typer.Exit(code=1)

    missing = [d for d in dbs if not pathlib.Path(d).is_file()]
    if missing:
        console.print(f"[red]No such database:[/red] {', '.join(missing)}")
        raise typer.Exit(code=1)

    try:
        loaded = load_suites(_suites_root(), names=[suite])
    except TaskValidationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    tasks = {t.id: t for t in loaded[suite].tasks if isinstance(t.grader, UnitTestGrader)}
    if not tasks:
        console.print(f"[yellow]Suite {suite} has no unit_tests tasks.[/yellow]")
        raise typer.Exit(code=0)

    # Collect stored completions. Attempts predating raw-answer persistence
    # cannot be replayed, so report them rather than counting them as zeros.
    corpus: list[tuple[str, str, str]] = []
    without_answer = 0
    for db in dbs:
        store = ResultStore(db)
        for run in store.runs():
            if run.id is None:
                continue
            for row in store.attempts(run.id):
                if row.task_id not in tasks or row.fault == "api_error":
                    continue
                if not row.answer:
                    without_answer += 1
                    continue
                corpus.append((row.task_id, row.model, row.answer))

    if without_answer:
        console.print(
            f"[yellow]{without_answer} stored attempt(s) have no saved reply[/yellow] "
            "and cannot be regraded (recorded before raw completions were kept)."
        )
    if not corpus:
        console.print("[yellow]Nothing to regrade.[/yellow]")
        raise typer.Exit(code=0)

    console.print(
        f"Regrading {len(corpus)} completion(s) under {len(wanted)} policies "
        f"({len(corpus) * len(wanted)} sandboxed test runs, no API calls)."
    )
    outcomes = []
    for policy in wanted:
        for task_id, model, answer in corpus:
            outcomes.append(
                regrade_attempt(tasks[task_id], answer, policy, model=model)
            )
        done = [o for o in outcomes if o.policy == policy]
        console.print(
            f"  {policy:15} {sum(1 for o in done if o.passed)}/{len(done)} passed"
        )

    report = summarize_regrade(outcomes, wanted)
    table = Table(title=f"Extraction-policy ablation ({suite})", header_style="bold cyan")
    for header in ("Policy", "Passed", "Pass rate", "Delta", "Changed", "Recovered", "Broken"):
        table.add_column(header, justify="left" if header == "Policy" else "right")
    for policy in wanted:
        stats = report["policies"].get(policy)
        if not stats:
            continue
        delta = stats["delta_vs_baseline"]
        table.add_row(
            policy,
            f"{stats['passed']}/{stats['attempts']}",
            f"{(stats['pass_rate'] or 0) * 100:.1f}%",
            "baseline" if policy == report["baseline"] else f"{delta * 100:+.1f} pts",
            str(stats["submissions_changed"]),
            str(len(stats["recovered"])),
            str(len(stats["broken"])),
        )
    console.print(table)

    for policy in wanted:
        stats = report["policies"].get(policy) or {}
        for label, key in (("recovered by", "recovered"), ("broken by", "broken")):
            if stats.get(key):
                console.print(f"[dim]{label} {policy}:[/dim] {', '.join(stats[key])}")

    if out_json:
        out = pathlib.Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"[green]Report written:[/green] {out}")


@app.command()
def web() -> None:
    """Launch the FastAPI web dashboard."""
    from openrouter_agent_bench.server.app import run as run_server

    run_server()


def main() -> None:
    """Console-script entry point for ``bench``."""
    app()


if __name__ == "__main__":
    main()
