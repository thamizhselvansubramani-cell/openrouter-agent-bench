"""Graders: turn a model's final answer into a pass/fail decision.

All graders return a :class:`GradeResult` with a normalized ``score`` in
``[0, 1]`` so downstream statistics never depend on grader internals.
"""

from __future__ import annotations

import pathlib
import re

from pydantic import BaseModel

from openrouter_agent_bench.sandbox.executor import SandboxExecutor, SandboxWorkspace
from openrouter_agent_bench.tasks.schema import (
    ExactMatchGrader,
    KeyedFactsGrader,
    TaskSpec,
    UnitTestGrader,
)

FILE_MARKER_RE = re.compile(r"^#{1,4}\s*FILE:\s*(?P<path>[\w./\\-]+)\s*$", re.MULTILINE)
FENCED_BLOCK_RE = re.compile(
    r"```[\w.-]*[ \t]*(?:path=(?P<info_path>[\w./\\-]+))?\n(?P<body>.*?)```",
    re.DOTALL,
)

CODING_RESPONSE_FORMAT = """

Respond with the complete corrected content of each changed file. Use one
fenced code block per file, each preceded by a marker line:

### FILE: <relative/path>
```python
<full file content>
```

Do not include explanations outside the blocks. Do not modify test files.
"""


class GradeResult(BaseModel):
    """Normalized grading outcome."""

    passed: bool
    score: float
    detail: str
    #: Machine-readable fault tag when the run failed operationally
    #: (timeout, malformed submission...). None for plain wrong answers.
    fault: str | None = None


def extract_file_blocks(answer: str) -> dict[str, str]:
    """Extract ``{relative_path: content}`` from a structured reply.

    Supported conventions:
    - a ``### FILE: <path>`` marker line directly before each fenced block;
    - `````python path=<name>` fence info strings.
    A bare reply with exactly one fenced block and no markers yields that
    block under the caller's target file name (handled by the caller).
    """
    files: dict[str, str] = {}
    matches = list(FENCED_BLOCK_RE.finditer(answer))
    for match in matches:
        info_path = match.group("info_path")
        if info_path:
            files[info_path.replace("\\", "/")] = match.group("body").strip() + "\n"
    if files:
        return files

    markers = list(FILE_MARKER_RE.finditer(answer))
    consumed: set[int] = set()
    for marker in markers:
        start = marker.end()
        # Search forward through the remaining blocks so each marker pairs
        # with the first fence that follows it.
        block_idx = next(
            (
                i
                for i, m in enumerate(matches)
                if i not in consumed and m.start() >= start
            ),
            None,
        )
        if block_idx is None:
            continue
        consumed.add(block_idx)
        block = FENCED_BLOCK_RE.search(answer[start : matches[block_idx].end()])
        if block:
            path = marker.group("path").replace("\\", "/")
            files[path] = block.group("body").strip() + "\n"
    if not files and matches:
        files[""] = matches[0].group("body").strip() + "\n"
    return files


def _normalize(text: str, case_sensitive: bool) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip())
    return collapsed if case_sensitive else collapsed.lower()


def grade_exact_match(task: TaskSpec, grader: ExactMatchGrader, answer: str) -> GradeResult:
    """Grade an answer against ``grader.expected``.

    Matching rules:
    - ``case_sensitive=True``: strict equality after whitespace collapsing.
    - otherwise, when the expected answer is all lowercase: lenient
      case-insensitive containment (models often embed the answer in a
      sentence such as ``The answer is PARIS.``).
    - otherwise: strict equality.
    """
    got = re.sub(r"\s+", " ", answer.strip())
    want = re.sub(r"\s+", " ", grader.expected.strip())
    if grader.case_sensitive:
        passed = got == want
    elif want.islower():
        passed = want in got.lower()
    else:
        passed = got == want
    if not passed and grader.must_include:
        haystack = got if grader.case_sensitive else got.lower()
        passed = all(
            _normalize(fact, grader.case_sensitive) in haystack for fact in grader.must_include
        )
        if passed:
            return GradeResult(passed=True, score=1.0, detail="matched all required facts")
    return GradeResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        detail=f"expected {want[:200]!r}, got {got[:200]!r}",
    )


def grade_keyed_facts(grader: KeyedFactsGrader, answer: str) -> GradeResult:
    haystack = answer.lower()
    missing = [key for key, value in grader.facts.items() if value.lower() not in haystack]
    found = len(grader.facts) - len(missing)
    ratio = found / max(len(grader.facts), 1)
    detail = f"{found}/{len(grader.facts)} facts matched"
    if missing:
        detail += f"; missing keys: {', '.join(sorted(missing))}"
    return GradeResult(passed=ratio >= 1.0, score=ratio, detail=detail)


def grade_unit_tests(
    task: TaskSpec,
    grader: UnitTestGrader,
    answer: str,
    sandbox: SandboxExecutor,
    workspace: SandboxWorkspace,
) -> GradeResult:
    """Write the submitted files, then run the hidden tests in the sandbox."""
    blocks = extract_file_blocks(answer)
    if not blocks:
        return GradeResult(
            passed=False,
            score=0.0,
            detail="no fenced code blocks found in reply",
            fault="malformed_submission",
        )
    default_target = task.target_file or "submission.py"
    for rel, content in blocks.items():
        path = rel or default_target
        if "test" not in pathlib.PurePosixPath(path).name:
            workspace.write(path, content)
    test_files = list(grader.tests.keys())
    results = sandbox.run_pytest(workspace, test_files, timeout_s=min(task.timeout_s, 300))
    passed = results.ok
    summary = (results.stdout or "").strip().splitlines()
    tail = "\n".join(summary[-5:]) if summary else results.stderr[-400:]
    return GradeResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        detail=tail,
        fault="test_crash" if results.timed_out else None,
    )


def grade_answer(
    task: TaskSpec,
    answer: str,
    *,
    sandbox: SandboxExecutor | None = None,
    workspace: SandboxWorkspace | None = None,
    judge_runner: object | None = None,
) -> GradeResult:
    """Dispatch to the right grader implementation for a task."""
    grader = task.grader
    if isinstance(grader, ExactMatchGrader):
        return grade_exact_match(task, grader, answer)
    if isinstance(grader, KeyedFactsGrader):
        return grade_keyed_facts(grader, answer)
    if isinstance(grader, UnitTestGrader):
        if sandbox is None or workspace is None:
            msg = "unit_tests grading requires sandbox and workspace"
            raise ValueError(msg)
        return grade_unit_tests(task, grader, answer, sandbox, workspace)
    # Remaining graders (llm_judge and any future rubric-based types) are
    # delegated to the injected judge runner.
    if judge_runner is None:
        msg = "llm_judge grading requires a judge runner"
        raise ValueError(msg)
    return judge_runner.grade(task, grader, answer)  # type: ignore[attr-defined,no-any-return]


def build_prompt(task: TaskSpec) -> str:
    """Full user prompt for a task, including response-format instructions."""
    prompt = task.prompt
    if isinstance(task.grader, UnitTestGrader):
        target_note = (
            f"\nSubmit your changes to `{task.target_file}`." if task.target_file else ""
        )
        prompt += target_note + CODING_RESPONSE_FORMAT
    return prompt


__all__ = [
    "CODING_RESPONSE_FORMAT",
    "GradeResult",
    "build_prompt",
    "extract_file_blocks",
    "grade_answer",
    "grade_exact_match",
    "grade_keyed_facts",
    "grade_unit_tests",
]
