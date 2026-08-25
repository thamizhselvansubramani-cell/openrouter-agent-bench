"""Tests for run provenance capture."""

from __future__ import annotations

from openrouter_agent_bench.provenance import run_provenance, suite_hash


def test_suite_hash_is_order_independent() -> None:
    a = suite_hash([("t1", "prompt one"), ("t2", "prompt two")])
    b = suite_hash([("t2", "prompt two"), ("t1", "prompt one")])
    assert a == b


def test_suite_hash_changes_when_a_prompt_is_reworded() -> None:
    before = suite_hash([("t1", "prompt one")])
    after = suite_hash([("t1", "prompt one!")])
    assert before != after


def test_suite_hash_changes_when_a_task_is_added() -> None:
    before = suite_hash([("t1", "p")])
    after = suite_hash([("t1", "p"), ("t2", "p")])
    assert before != after


def test_suite_hash_separates_fields() -> None:
    """A task id must not be confusable with the tail of a prompt."""
    assert suite_hash([("ab", "c")]) != suite_hash([("a", "bc")])


def test_run_provenance_reports_environment() -> None:
    prov = run_provenance(task_ids_and_prompts=[("t1", "p")])
    assert prov["harness_version"]
    assert prov["python_version"]
    assert prov["platform"]
    assert prov["suite_hash"]


def test_run_provenance_without_tasks_has_no_suite_hash() -> None:
    assert run_provenance()["suite_hash"] is None


def test_run_provenance_outside_a_git_worktree(tmp_path: object) -> None:
    """``git_sha`` degrades to None rather than raising off a work tree."""
    prov = run_provenance(repo_root=tmp_path)  # type: ignore[arg-type]
    assert prov["git_sha"] is None or isinstance(prov["git_sha"], str)
