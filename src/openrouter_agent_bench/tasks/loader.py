"""Task suite loading and validation.

Suites live in ``tasks/suites/<suite_name>/<task_id>.yaml`` (one task per
file). Every file is validated with pydantic at load time; malformed files
raise :class:`TaskValidationError` listing the offending path and reason.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any

import yaml

from openrouter_agent_bench.tasks.schema import SuiteManifest, TaskSpec

SUITES_DIR_ENV = "OAB_SUITES_DIR"


class TaskValidationError(Exception):
    """Raised when a suite or task file fails validation."""


@dataclass(frozen=True)
class LoadedSuite:
    """A suite directory with its parsed tasks."""

    name: str
    manifest: SuiteManifest
    tasks: list[TaskSpec]
    #: True when the directory declared itself via an explicit ``suite.yaml``.
    has_manifest: bool = False


def _parse_task_file(path: pathlib.Path) -> TaskSpec:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"{path}: invalid YAML ({exc})"
        raise TaskValidationError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path}: expected a mapping at top level"
        raise TaskValidationError(msg)
    try:
        return TaskSpec.model_validate(data)
    except Exception as exc:
        msg = f"{path}: {exc}"
        raise TaskValidationError(msg) from exc


def load_suite(suite_dir: str | pathlib.Path) -> LoadedSuite:
    """Load a single suite directory containing ``*.yaml`` task files."""
    sdir = pathlib.Path(suite_dir)
    manifest_path = sdir / "suite.yaml"
    has_manifest = manifest_path.exists()
    if not sdir.is_dir():
        msg = f"suite directory not found: {sdir}"
        raise TaskValidationError(msg)
    manifest_data: dict[str, Any] = {}
    if has_manifest:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            manifest_data = loaded
    manifest = SuiteManifest(
        name=manifest_data.get("name", sdir.name), description=manifest_data.get("description", "")
    )
    tasks = [
        _parse_task_file(p) for p in sorted(sdir.glob("*.yaml")) if p.name != "suite.yaml"
    ]
    for p in sorted(sdir.glob("*.json")):
        tasks.append(_parse_task_file(p))
    seen: set[str] = set()
    for t in tasks:
        if t.id in seen:
            msg = f"{sdir}: duplicate task id {t.id}"
            raise TaskValidationError(msg)
        seen.add(t.id)
    return LoadedSuite(
        name=manifest.name, manifest=manifest, tasks=tasks, has_manifest=has_manifest
    )


def load_suites(root: str | pathlib.Path, names: list[str] | None = None) -> dict[str, LoadedSuite]:
    """Load all (or selected) suites under the suites root directory."""
    base = pathlib.Path(root)
    if not base.is_dir():
        msg = f"suites root not found: {base}"
        raise TaskValidationError(msg)
    wanted = set(names) if names else None
    suites: dict[str, LoadedSuite] = {}
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        suite = load_suite(child)
        if wanted is not None and suite.manifest.name not in wanted:
            continue
        # Keep suites that have tasks or explicitly declared themselves via
        # suite.yaml; skip bare empty directories.
        if suite.tasks or suite.has_manifest:
            suites[suite.manifest.name] = suite
    if wanted is not None:
        missing = wanted - set(suites)
        if missing:
            msg = f"suites not found: {sorted(missing)}"
            raise TaskValidationError(msg)
    return suites


def default_suites_root(repo_root: str | pathlib.Path | None = None) -> pathlib.Path:
    """Locate the shipped ``tasks/suites`` directory."""
    env_dir = os.environ.get(SUITES_DIR_ENV)
    if env_dir:
        return pathlib.Path(env_dir)
    base = pathlib.Path(repo_root) if repo_root else pathlib.Path.cwd()
    candidate = base / "tasks" / "suites"
    return candidate


def validate_all(root: str | pathlib.Path) -> tuple[int, int]:
    """Validate every suite; returns (suite_count, task_count).

    Raises :class:`TaskValidationError` on the first invalid file.
    """
    suites = load_suites(root)
    return len(suites), sum(len(s.tasks) for s in suites.values())
