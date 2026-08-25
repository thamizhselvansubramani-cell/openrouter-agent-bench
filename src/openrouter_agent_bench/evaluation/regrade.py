"""Re-score stored completions under alternative submission-extraction policies.

A ``unit_tests`` score is not a pure measure of coding ability: it also measures
whether the model packaged its answer the way the grader expects. The default
policy overwrites the target file with the *first* fenced block it finds, so a
reply containing only the corrected function -- a reasonable reading of "fix
this bug" -- silently loses the file's imports and fails at collection. That
scores identically to genuinely wrong logic.

Because :class:`~openrouter_agent_bench.storage.storage.AttemptRow` persists the
raw reply, that conflation is measurable offline: re-grade one fixed corpus of
completions under several policies and read the difference. No API calls are
made here, so the ablation is free and repeatable.

The policies:

``strict``
    The shipped behaviour, reproduced exactly. Baseline.
``largest_block``
    For unmarked replies, take the largest fenced block rather than the first.
    Isolates the cost of models that show a "before" snippet first.
``concat_blocks``
    Concatenate every non-test block in order. Isolates replies split across
    several fences.
``splice``
    If overwriting the target would drop definitions or imports the original
    had -- the tests usually import those siblings, so the module fails at
    collection -- graft the submission's top-level definitions into the
    original by name instead. Isolates the fragment-versus-whole-file
    convention specifically.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from openrouter_agent_bench.evaluation.graders import (
    FENCED_BLOCK_RE,
    GradeResult,
    extract_file_blocks,
)
from openrouter_agent_bench.sandbox.executor import SandboxExecutor, SandboxWorkspace
from openrouter_agent_bench.tasks.schema import TaskSpec, UnitTestGrader

__all__ = [
    "BASELINE",
    "POLICIES",
    "PolicyOutcome",
    "extract_with_policy",
    "grade_result_of",
    "regrade_attempt",
    "splice_fragment",
    "summarize_regrade",
]

POLICIES: tuple[str, ...] = ("strict", "largest_block", "concat_blocks", "splice")

#: Baseline policy every other policy is compared against.
BASELINE = "strict"


@dataclass(frozen=True)
class PolicyOutcome:
    """Result of grading one stored completion under one policy."""

    policy: str
    task_id: str
    model: str
    passed: bool
    fault: str | None = None
    detail: str = ""
    #: True when the policy altered what got written, relative to ``strict``.
    changed_submission: bool = False
    files: dict[str, str] = field(default_factory=dict)


def _blocks(answer: str) -> list[str]:
    """Every fenced block body, in order of appearance."""
    return [m.group("body").strip() + "\n" for m in FENCED_BLOCK_RE.finditer(answer)]


def _is_test_path(path: str) -> bool:
    return "test" in pathlib.PurePosixPath(path).name


def _top_level_defs(tree: ast.Module) -> dict[str, ast.stmt]:
    out: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out[node.name] = node
    return out


def _has_top_level_import(tree: ast.Module) -> bool:
    return any(isinstance(n, ast.Import | ast.ImportFrom) for n in tree.body)


def _span(node: ast.stmt) -> tuple[int, int]:
    """1-indexed inclusive line span of ``node``, including any decorators."""
    start = node.lineno
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        start = min(start, min(d.lineno for d in decorators))
    end = node.end_lineno or node.lineno
    return start, end


def splice_fragment(original: str, submission: str) -> str | None:
    """Graft ``submission``'s top-level definitions into ``original``.

    Returns the merged source, or ``None`` when splicing does not apply --
    either side failing to parse, the submission defining nothing at top
    level, or the submission already looking like a complete module.
    """
    try:
        orig_tree = ast.parse(original)
        sub_tree = ast.parse(submission)
    except SyntaxError:
        return None

    sub_defs = _top_level_defs(sub_tree)
    if not sub_defs:
        return None
    orig_defs = _top_level_defs(orig_tree)

    # The submission must be talking about this file at all.
    if not set(sub_defs) & set(orig_defs):
        return None

    # Splice only when overwriting would *lose* something the original had --
    # sibling definitions the tests import, or the module's imports. If the
    # submission already covers everything, it is a whole file and overwriting
    # is the correct reading.
    drops_defs = bool(set(orig_defs) - set(sub_defs))
    drops_imports = _has_top_level_import(orig_tree) and not _has_top_level_import(sub_tree)
    if not drops_defs and not drops_imports:
        return None

    orig_lines = original.splitlines(keepends=True)
    sub_lines = submission.splitlines(keepends=True)

    # Replace matching definitions from the bottom up so earlier line numbers
    # stay valid as we edit.
    replacements: list[tuple[int, int, list[str]]] = []
    for name, sub_node in sub_defs.items():
        if name not in orig_defs:
            continue
        o_start, o_end = _span(orig_defs[name])
        s_start, s_end = _span(sub_node)
        replacements.append((o_start, o_end, sub_lines[s_start - 1 : s_end]))
    for o_start, o_end, new_lines in sorted(replacements, reverse=True):
        orig_lines[o_start - 1 : o_end] = new_lines

    merged = "".join(orig_lines)
    # Definitions the original did not have are appended.
    extra = [n for name, n in sub_defs.items() if name not in orig_defs]
    if extra:
        tail = []
        for node in extra:
            s_start, s_end = _span(node)
            tail.extend(sub_lines[s_start - 1 : s_end])
        if not merged.endswith("\n"):
            merged += "\n"
        merged += "\n\n" + "".join(tail)
    return merged


def extract_with_policy(
    answer: str, task: TaskSpec, policy: str
) -> tuple[dict[str, str], bool]:
    """Files a policy would write, plus whether it diverged from ``strict``.

    The divergence flag is what makes the ablation interpretable: a policy that
    never changes the submission cannot explain a score difference.
    """
    if policy not in POLICIES:
        msg = f"unknown policy: {policy!r} (expected one of {POLICIES})"
        raise ValueError(msg)

    target = task.target_file or "submission.py"
    strict_raw = extract_file_blocks(answer)
    strict_files = {
        (rel or target): body
        for rel, body in strict_raw.items()
        if not _is_test_path(rel or target)
    }

    if policy == "strict":
        return strict_files, False

    if policy == "largest_block":
        # Only meaningful for unmarked replies: with markers, paths already
        # disambiguate the blocks and strict extraction is unambiguous.
        if list(strict_raw) != [""]:
            return strict_files, False
        blocks = _blocks(answer)
        if len(blocks) < 2:
            return strict_files, False
        best = max(blocks, key=len)
        return {target: best}, best != next(iter(strict_files.values()), None)

    if policy == "concat_blocks":
        if list(strict_raw) != [""]:
            return strict_files, False
        blocks = _blocks(answer)
        if len(blocks) < 2:
            return strict_files, False
        joined = "\n".join(b.rstrip() + "\n" for b in blocks)
        return {target: joined}, True

    # splice
    files = dict(strict_files)
    changed = False
    for rel, body in list(files.items()):
        original = task.files.get(rel)
        if original is None:
            continue
        merged = splice_fragment(original, body)
        if merged is not None and merged != body:
            files[rel] = merged
            changed = True
    return files, changed


def regrade_attempt(
    task: TaskSpec,
    answer: str,
    policy: str,
    *,
    model: str = "",
    timeout_s: float = 60.0,
    memory_mb: int | None = 512,
) -> PolicyOutcome:
    """Grade one stored completion under ``policy`` in a fresh sandbox."""
    grader = task.grader
    if not isinstance(grader, UnitTestGrader):
        msg = f"{task.id}: regrading only applies to unit_tests tasks"
        raise TypeError(msg)

    files, changed = extract_with_policy(answer, task, policy)
    if not files:
        return PolicyOutcome(
            policy=policy,
            task_id=task.id,
            model=model,
            passed=False,
            fault="malformed_submission",
            detail="no fenced code blocks found in reply",
            changed_submission=changed,
        )

    workspace = SandboxWorkspace.create({**task.files, **grader.tests})
    try:
        for rel, content in files.items():
            workspace.write(rel, content)
        sandbox = SandboxExecutor(timeout_s=timeout_s, memory_mb=memory_mb)
        result = sandbox.run_pytest(
            workspace, list(grader.tests), timeout_s=min(task.timeout_s, 300)
        )
    finally:
        workspace.cleanup()

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    return PolicyOutcome(
        policy=policy,
        task_id=task.id,
        model=model,
        passed=result.ok,
        fault="test_crash" if result.timed_out else None,
        detail=stdout or stderr,
        changed_submission=changed,
        files=files,
    )


def grade_result_of(outcome: PolicyOutcome) -> GradeResult:
    """Adapt a :class:`PolicyOutcome` to the common grading result type."""
    return GradeResult(
        passed=outcome.passed,
        score=1.0 if outcome.passed else 0.0,
        detail=outcome.detail,
        fault=outcome.fault,
    )


def summarize_regrade(
    outcomes: Iterable[PolicyOutcome], policies: Sequence[str] = POLICIES
) -> dict[str, Any]:
    """Aggregate outcomes into per-policy rates and deltas against the baseline.

    ``recovered`` and ``broken`` name the attempts whose verdict flipped, which
    is the actionable part: they are concrete cases where the measured score
    reflected packaging rather than correctness.
    """
    rows = list(outcomes)
    by_policy: dict[str, list[PolicyOutcome]] = {p: [] for p in policies}
    for o in rows:
        by_policy.setdefault(o.policy, []).append(o)

    baseline = {(o.task_id, o.model): o.passed for o in by_policy.get(BASELINE, [])}
    base_n = len(baseline)
    base_passed = sum(1 for v in baseline.values() if v)

    report: dict[str, Any] = {
        "baseline": BASELINE,
        "attempts": base_n,
        "baseline_passed": base_passed,
        "baseline_pass_rate": (base_passed / base_n) if base_n else None,
        "policies": {},
    }
    for policy in policies:
        outs = by_policy.get(policy, [])
        if not outs:
            continue
        n = len(outs)
        passed = sum(1 for o in outs if o.passed)
        recovered = sorted(
            f"{o.model}:{o.task_id}"
            for o in outs
            if o.passed and baseline.get((o.task_id, o.model)) is False
        )
        broken = sorted(
            f"{o.model}:{o.task_id}"
            for o in outs
            if not o.passed and baseline.get((o.task_id, o.model)) is True
        )
        report["policies"][policy] = {
            "attempts": n,
            "passed": passed,
            "pass_rate": passed / n if n else None,
            "delta_vs_baseline": (passed - base_passed) / base_n if base_n else None,
            "submissions_changed": sum(1 for o in outs if o.changed_submission),
            "recovered": recovered,
            "broken": broken,
            "malformed": sum(1 for o in outs if o.fault == "malformed_submission"),
        }
    return report
