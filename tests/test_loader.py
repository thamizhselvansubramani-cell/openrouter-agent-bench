"""Tests for task/suite loading and validation."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from openrouter_agent_bench.tasks.loader import (
    TaskValidationError,
    load_suite,
    load_suites,
    validate_all,
)
from openrouter_agent_bench.tasks.schema import ExactMatchGrader

VALID_TASK = """
id: sample-task-1
suite: coding
title: Sample
category: algorithms
difficulty: 3
prompt: Fix the thing.
files:
  mod.py: |
    X = 1
grader:
  type: exact_match
  expected: done
"""


def _make_suites(root: pathlib.Path) -> pathlib.Path:
    suites = root / "suites"
    coding = suites / "coding"
    coding.mkdir(parents=True)
    (coding / "task_a.yaml").write_text(textwrap.dedent(VALID_TASK), encoding="utf-8")
    agentic = suites / "agentic"
    agentic.mkdir()
    (agentic / "suite.yaml").write_text("name: agentic\ndescription: Tools\n", encoding="utf-8")
    return suites


def test_load_suite_tasks(tmp_path: pathlib.Path) -> None:
    suites = _make_suites(tmp_path)
    loaded = load_suite(suites / "coding")
    assert len(loaded.tasks) == 1
    task = loaded.tasks[0]
    assert task.id == "sample-task-1"
    assert isinstance(task.grader, ExactMatchGrader)


def test_suite_manifest_defaults(tmp_path: pathlib.Path) -> None:
    suites = _make_suites(tmp_path)
    loaded = load_suite(suites / "agentic")
    assert loaded.manifest.name == "agentic"
    assert loaded.manifest.description == "Tools"


def test_missing_suite_dir_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TaskValidationError, match="not found"):
        load_suite(tmp_path / "nope")


def test_invalid_task_reports_path(tmp_path: pathlib.Path) -> None:
    suites = _make_suites(tmp_path)
    bad = suites / "coding" / "bad.yaml"
    bad.write_text("id: no-suite-field\n", encoding="utf-8")
    with pytest.raises(TaskValidationError, match="bad.yaml"):
        load_suite(suites / "coding")


def test_duplicate_task_ids_rejected(tmp_path: pathlib.Path) -> None:
    suites = _make_suites(tmp_path)
    dup = suites / "coding" / "task_b.yaml"
    dup.write_text(textwrap.dedent(VALID_TASK), encoding="utf-8")
    with pytest.raises(TaskValidationError, match="duplicate"):
        load_suite(suites / "coding")


def test_load_suites_filters_and_errors(tmp_path: pathlib.Path) -> None:
    suites = _make_suites(tmp_path)
    picked = load_suites(suites, ["coding"])
    assert set(picked) == {"coding"}
    with pytest.raises(TaskValidationError, match="not found"):
        load_suites(suites, ["long_context"])


def test_validate_all_counts(tmp_path: pathlib.Path) -> None:
    suites = _make_suites(tmp_path)
    n_suites, n_tasks = validate_all(suites)
    assert (n_suites, n_tasks) == (2, 1)


def test_shipped_suites_are_valid(repo_root: pathlib.Path) -> None:
    n_suites, n_tasks = validate_all(repo_root / "tasks" / "suites")
    assert n_suites >= 1
    assert n_tasks >= 12
