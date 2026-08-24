"""Tests for grader logic and submission extraction."""

from __future__ import annotations

import pytest

from openrouter_agent_bench.evaluation.graders import (
    build_prompt,
    extract_file_blocks,
    grade_exact_match,
    grade_keyed_facts,
)
from openrouter_agent_bench.tasks.schema import ExactMatchGrader, KeyedFactsGrader, TaskSpec


def test_extract_marked_blocks() -> None:
    answer = (
        "Here is the fix:\n\n"
        "### FILE: src/a.py\n"
        "```python\n"
        "VALUE = 1\n"
        "```\n\n"
        "### FILE: src/b.py\n"
        "```python\n"
        "VALUE = 2\n"
        "```\n"
    )
    files = extract_file_blocks(answer)
    assert files == {"src/a.py": "VALUE = 1\n", "src/b.py": "VALUE = 2\n"}


def test_extract_fence_info_path() -> None:
    answer = "```python path=pkg/mod.py\nX = 9\n```"
    assert extract_file_blocks(answer) == {"pkg/mod.py": "X = 9\n"}


def test_extract_single_bare_block() -> None:
    answer = "```python\nprint('hi')\n```"
    assert extract_file_blocks(answer) == {"": "print('hi')\n"}


def test_extract_no_blocks() -> None:
    assert extract_file_blocks("I could not solve it.") == {}


@pytest.mark.parametrize(
    ("expected", "answer", "passed"),
    [
        ("paris", "The answer is PARIS.", True),
        ("paris", "london", False),
        ("CaseSensitive", "casesensitive", False),
    ],
)
def test_exact_match(expected: str, answer: str, passed: bool) -> None:
    task = TaskSpec(
        id="t",
        suite="coding",
        title="T",
        category="c",
        difficulty=1,
        prompt="p",
        grader=ExactMatchGrader(expected=expected),
    )
    result = grade_exact_match(task, task.grader, answer)  # type: ignore[arg-type]
    assert result.passed is passed


def test_exact_match_case_sensitive() -> None:
    task = TaskSpec(
        id="t",
        suite="coding",
        title="T",
        category="c",
        difficulty=1,
        prompt="p",
        grader=ExactMatchGrader(expected="Paris", case_sensitive=True),
    )
    result = grade_exact_match(task, task.grader, "PARIS")  # type: ignore[arg-type]
    assert not result.passed


def test_keyed_facts_partial_credit() -> None:
    grader = KeyedFactsGrader(facts={"q1": "42", "q2": "berlin", "q3": "zeta"})
    result = grade_keyed_facts(grader, "the answers are 42 and BERLIN but not omega")
    assert not result.passed
    assert result.score == pytest.approx(2 / 3)
    full = grade_keyed_facts(grader, "42, berlin, zeta")
    assert full.passed and full.score == 1.0


def test_build_prompt_appends_format_for_unit_tests() -> None:
    from openrouter_agent_bench.tasks.schema import UnitTestGrader

    task = TaskSpec(
        id="t",
        suite="coding",
        title="T",
        category="c",
        difficulty=1,
        target_file="mod.py",
        prompt="fix it",
        grader=UnitTestGrader(tests={"test_mod.py": "def test_x():\n    assert True\n"}),
    )
    prompt = build_prompt(task)
    assert "### FILE:" in prompt
    assert "`mod.py`" in prompt
