"""Run provenance: what code, what tasks, what environment produced a result.

A benchmark number is only reproducible if you know which harness revision
produced it, which task definitions were in play, and on what interpreter.
:func:`run_provenance` collects that into a flat dict suitable for persisting
alongside a run.
"""

from __future__ import annotations

import hashlib
import pathlib
import platform
import subprocess
import sys
from collections.abc import Iterable

from openrouter_agent_bench import __version__

__all__ = [
    "git_sha",
    "run_provenance",
    "suite_hash",
]


def git_sha(repo_root: str | pathlib.Path | None = None) -> str | None:
    """Current git commit of the harness, or ``None`` outside a work tree.

    Appends ``"-dirty"`` when the work tree has uncommitted changes, so a
    result produced from edited code is never mistaken for a clean revision.
    """
    cwd = str(repo_root) if repo_root else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not sha:
        return None
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return sha
    return f"{sha}-dirty" if dirty else sha


def suite_hash(task_ids_and_prompts: Iterable[tuple[str, str]]) -> str:
    """Stable digest over the tasks used in a run.

    Hashes ``(task_id, prompt)`` pairs in sorted order, so the digest changes
    whenever a task is added, removed, or reworded — which is exactly when a
    previous run's numbers stop being comparable.
    """
    digest = hashlib.sha256()
    for task_id, prompt in sorted(task_ids_and_prompts):
        digest.update(task_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def run_provenance(
    *,
    task_ids_and_prompts: Iterable[tuple[str, str]] | None = None,
    repo_root: str | pathlib.Path | None = None,
) -> dict[str, str | None]:
    """Collect the provenance fields persisted on a :class:`BenchRun`."""
    return {
        "harness_version": __version__,
        "git_sha": git_sha(repo_root),
        "suite_hash": suite_hash(task_ids_and_prompts) if task_ids_and_prompts else None,
        "python_version": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
    }
