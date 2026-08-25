"""Tests for offline re-grading under alternative extraction policies."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from openrouter_agent_bench.evaluation.regrade import (
    BASELINE,
    POLICIES,
    PolicyOutcome,
    extract_with_policy,
    regrade_attempt,
    splice_fragment,
    summarize_regrade,
)
from openrouter_agent_bench.tasks.loader import default_suites_root, load_suites

ORIGINAL = textwrap.dedent(
    '''
    """Module docstring."""

    import math


    def keep(x: int) -> int:
        return x + 1


    def broken(x: int) -> int:
        return math.floor(x) - 1
    '''
).lstrip()


# --------------------------------------------------------------------------
# splice_fragment
# --------------------------------------------------------------------------


def test_splice_grafts_a_fragment_and_keeps_siblings() -> None:
    fragment = "def broken(x: int) -> int:\n    return math.floor(x) + 1\n"
    merged = splice_fragment(ORIGINAL, fragment)
    assert merged is not None
    assert "import math" in merged
    assert "def keep" in merged
    assert "math.floor(x) + 1" in merged
    assert "math.floor(x) - 1" not in merged


def test_splice_declines_a_complete_module() -> None:
    """A submission that loses nothing should be written as-is."""
    complete = ORIGINAL.replace("- 1", "+ 1")
    assert splice_fragment(ORIGINAL, complete) is None


def test_splice_declines_unparseable_submission() -> None:
    assert splice_fragment(ORIGINAL, "def broken(: syntax error") is None


def test_splice_declines_when_nothing_overlaps() -> None:
    assert splice_fragment(ORIGINAL, "def unrelated():\n    return 0\n") is None


def test_splice_declines_a_submission_with_no_definitions() -> None:
    assert splice_fragment(ORIGINAL, "x = 1\n") is None


def test_splice_preserves_decorators_on_the_replacement() -> None:
    original = "import functools\n\n\ndef keep():\n    return 1\n\n\ndef f():\n    return 0\n"
    fragment = "@functools.cache\ndef f():\n    return 2\n"
    merged = splice_fragment(original, fragment)
    assert merged is not None
    assert "@functools.cache" in merged
    assert "return 2" in merged
    assert "def keep" in merged


def test_splice_appends_a_new_definition() -> None:
    fragment = "def broken(x: int) -> int:\n    return 0\n\n\ndef helper():\n    return 1\n"
    merged = splice_fragment(ORIGINAL, fragment)
    assert merged is not None
    assert "def helper" in merged
    assert "def keep" in merged


# --------------------------------------------------------------------------
# extract_with_policy
# --------------------------------------------------------------------------


@pytest.fixture
def coding_task():  # type: ignore[no-untyped-def]
    root = default_suites_root(pathlib.Path(__file__).resolve().parents[1])
    suite = load_suites(root, names=["coding"])["coding"]
    return next(t for t in suite.tasks if t.id == "binary-search-boundary")


def test_unknown_policy_is_rejected(coding_task) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="unknown policy"):
        extract_with_policy("```\nx\n```", coding_task, "nope")


def test_strict_never_reports_a_change(coding_task) -> None:  # type: ignore[no-untyped-def]
    files, changed = extract_with_policy("```python\nx = 1\n```", coding_task, "strict")
    assert changed is False
    assert files == {coding_task.target_file: "x = 1\n"}


def test_strict_takes_the_first_block(coding_task) -> None:  # type: ignore[no-untyped-def]
    answer = "before:\n```python\nSMALL\n```\nafter:\n```python\nMUCH LONGER BLOCK\n```"
    files, _ = extract_with_policy(answer, coding_task, "strict")
    assert files[coding_task.target_file].strip() == "SMALL"


def test_largest_block_prefers_the_bigger_fence(coding_task) -> None:  # type: ignore[no-untyped-def]
    answer = "before:\n```python\nSMALL\n```\nafter:\n```python\nMUCH LONGER BLOCK\n```"
    files, changed = extract_with_policy(answer, coding_task, "largest_block")
    assert changed is True
    assert files[coding_task.target_file].strip() == "MUCH LONGER BLOCK"


def test_largest_block_is_inert_with_a_single_fence(coding_task) -> None:  # type: ignore[no-untyped-def]
    files, changed = extract_with_policy("```python\nonly\n```", coding_task, "largest_block")
    assert changed is False
    assert files[coding_task.target_file].strip() == "only"


def test_concat_blocks_joins_every_fence(coding_task) -> None:  # type: ignore[no-untyped-def]
    answer = "```python\nA = 1\n```\ntext\n```python\nB = 2\n```"
    files, changed = extract_with_policy(answer, coding_task, "concat_blocks")
    assert changed is True
    body = files[coding_task.target_file]
    assert "A = 1" in body
    assert "B = 2" in body


def test_marked_replies_are_left_alone_by_block_policies(coding_task) -> None:  # type: ignore[no-untyped-def]
    """With FILE: markers the mapping is unambiguous; policies must not meddle."""
    answer = (
        "### FILE: search_bounds.py\n```python\nA = 1\n```\n"
        "### FILE: other.py\n```python\nB = 2\n```"
    )
    for policy in ("largest_block", "concat_blocks"):
        files, changed = extract_with_policy(answer, coding_task, policy)
        assert changed is False
        assert set(files) == {"search_bounds.py", "other.py"}


def test_test_files_are_never_written(coding_task) -> None:  # type: ignore[no-untyped-def]
    """A model must not be able to overwrite the hidden tests."""
    answer = "### FILE: test_search_bounds.py\n```python\ndef test_x(): pass\n```"
    files, _ = extract_with_policy(answer, coding_task, "strict")
    assert files == {}


# --------------------------------------------------------------------------
# end-to-end: the confound this module exists to measure
# --------------------------------------------------------------------------

FRAGMENT_FIX = '''Here is the corrected function:

```python
def find_rightmost(values: list[int], target: int) -> int:
    """Return the index of the last element equal to target, or -1."""
    lo, hi = 0, len(values) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid - 1
    if lo > 0 and values[lo - 1] == target:
        return lo - 1
    return -1
```
'''


def test_a_correct_fragment_fails_strict_but_passes_splice(coding_task) -> None:  # type: ignore[no-untyped-def]
    """The headline result: identical reply, opposite verdicts.

    A correct fix packaged as a fragment scores zero under the shipped policy
    because overwriting the file drops the sibling functions the tests import.
    That is a packaging failure being counted as a reasoning failure.
    """
    strict = regrade_attempt(coding_task, FRAGMENT_FIX, "strict", model="m")
    spliced = regrade_attempt(coding_task, FRAGMENT_FIX, "splice", model="m")

    assert strict.passed is False
    assert strict.changed_submission is False
    assert spliced.passed is True
    assert spliced.changed_submission is True


def test_regrade_reports_malformed_when_there_is_no_code(coding_task) -> None:  # type: ignore[no-untyped-def]
    out = regrade_attempt(coding_task, "I cannot help with that.", "strict", model="m")
    assert out.passed is False
    assert out.fault == "malformed_submission"


def test_regrade_rejects_non_unit_test_tasks() -> None:
    root = default_suites_root(pathlib.Path(__file__).resolve().parents[1])
    task = load_suites(root, names=["agentic"])["agentic"].tasks[0]
    with pytest.raises(TypeError, match="unit_tests"):
        regrade_attempt(task, "```\nx\n```", "strict")


# --------------------------------------------------------------------------
# summarize_regrade
# --------------------------------------------------------------------------


def _o(policy: str, task: str, passed: bool, **kw: object) -> PolicyOutcome:
    return PolicyOutcome(policy=policy, task_id=task, model="m", passed=passed, **kw)  # type: ignore[arg-type]


def test_summary_computes_delta_and_names_flipped_attempts() -> None:
    outcomes = [
        _o("strict", "t1", False),
        _o("strict", "t2", True),
        _o("splice", "t1", True, changed_submission=True),
        _o("splice", "t2", False),
    ]
    report = summarize_regrade(outcomes, ["strict", "splice"])

    assert report["baseline"] == BASELINE
    assert report["baseline_pass_rate"] == 0.5
    splice = report["policies"]["splice"]
    assert splice["pass_rate"] == 0.5
    assert splice["delta_vs_baseline"] == 0.0
    assert splice["recovered"] == ["m:t1"]
    assert splice["broken"] == ["m:t2"]
    assert splice["submissions_changed"] == 1


def test_summary_reports_a_positive_delta() -> None:
    outcomes = [
        _o("strict", "t1", False),
        _o("strict", "t2", False),
        _o("splice", "t1", True),
        _o("splice", "t2", True),
    ]
    report = summarize_regrade(outcomes, ["strict", "splice"])
    assert report["policies"]["splice"]["delta_vs_baseline"] == 1.0
    assert report["policies"]["strict"]["delta_vs_baseline"] == 0.0


def test_summary_handles_an_empty_corpus() -> None:
    report = summarize_regrade([], list(POLICIES))
    assert report["attempts"] == 0
    assert report["baseline_pass_rate"] is None
    assert report["policies"] == {}
