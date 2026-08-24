"""Build tasks/suites/{agentic,long_context}/*.yaml and validate them.

The ``long_context`` tasks embed a programmatically synthesized document with
hidden "needle" facts directly in the prompt, graded by ``keyed_facts``. The
``agentic`` tasks are single-turn planning/tool-selection problems (the current
runner is single-turn) also graded by ``keyed_facts`` / ``exact_match``.

Run from repo root:  uv run python scripts/gen_suite/build_extra_suites.py
"""

from __future__ import annotations

import pathlib
import random
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from openrouter_agent_bench.tasks.schema import SuiteManifest, TaskSpec  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUITES = ROOT / "tasks" / "suites"

# --- Deterministic filler generation for long-context documents -------------

_SERVICES = [
    "auth-gateway", "billing-worker", "search-indexer", "notification-fanout",
    "image-resizer", "ledger-reconciler", "session-store", "webhook-dispatch",
    "rate-limiter", "audit-logger", "feature-flags", "cache-warmer",
]
_REGIONS = ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-1", "sa-east-1"]
_VERBS = [
    "reconciled pending records", "flushed the write buffer", "rotated the signing key",
    "rebalanced shard ownership", "evicted stale sessions", "compacted the WAL segment",
    "retried the upstream call", "emitted a heartbeat", "checkpointed offsets",
]


def _filler_lines(rng: random.Random, count: int) -> list[str]:
    lines: list[str] = []
    for i in range(count):
        svc = rng.choice(_SERVICES)
        region = rng.choice(_REGIONS)
        verb = rng.choice(_VERBS)
        ms = rng.randint(3, 900)
        lines.append(
            f"[2026-05-{rng.randint(1, 28):02d}T"
            f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z] "
            f"seq={i:04d} svc={svc} region={region} {verb} in {ms}ms"
        )
    return lines


def _long_document(rng: random.Random, needles: dict[int, str], total: int) -> str:
    """Build ``total`` filler lines and splice each needle at its 1-based line."""
    lines = _filler_lines(rng, total)
    for line_no, needle in needles.items():
        idx = max(0, min(line_no - 1, total - 1))
        lines[idx] = needle
    header = (
        "OPERATIONS LOG EXPORT — retention tier: cold. Lines are append-only and "
        "ordered by sequence. Most entries are routine; a few carry configuration "
        "facts that an on-call engineer may be asked to recover.\n"
    )
    return header + "\n".join(lines) + "\n"


# --- long_context tasks -----------------------------------------------------


def _long_context_tasks() -> list[TaskSpec]:
    rng = random.Random(20260524)

    doc1 = _long_document(
        rng,
        needles={
            37: "[config] CANARY_ROLLOUT_PERCENT for search-indexer is set to 17 percent.",
            118: "[config] The audit-logger retention window is EXACTLY 913 days.",
            206: "[config] On-call escalation owner is prisha.rao@example.net.",
        },
        total=240,
    )
    task1 = TaskSpec(
        id="needle-ops-log",
        suite="long_context",
        title="Recover config facts from a long operations log",
        category="retrieval",
        difficulty=3,
        generator="ops_log",
        context_tokens=len(doc1) // 4,
        timeout_s=180,
        prompt=(
            "Below is a long operations log. Read it carefully and answer three "
            "questions using EXACT values copied from the log.\n\n"
            "Questions:\n"
            "1. What percent is CANARY_ROLLOUT_PERCENT for search-indexer?\n"
            "2. How many days is the audit-logger retention window?\n"
            "3. What is the on-call escalation owner's email address?\n\n"
            "Answer with the three values, each on its own line.\n\n"
            "=== BEGIN LOG ===\n" + doc1 + "=== END LOG ===\n"
        ),
        grader={
            "type": "keyed_facts",
            "facts": {
                "canary_percent": "17",
                "retention_days": "913",
                "owner_email": "prisha.rao@example.net",
            },
        },  # type: ignore[arg-type]
    )

    doc2 = _long_document(
        rng,
        needles={
            52: "[policy] Data classified as RESTRICTED must be encrypted with key alias k-emerald.",
            141: "[policy] Backups older than 30 days move to archive bucket vault-nimbus-7.",
            222: "[policy] The break-glass admin role is named oncall-breakglass-omega.",
        },
        total=260,
    )
    task2 = TaskSpec(
        id="needle-policy-doc",
        suite="long_context",
        title="Extract policy identifiers from a long document",
        category="retrieval",
        difficulty=4,
        generator="policy_doc",
        context_tokens=len(doc2) // 4,
        timeout_s=180,
        prompt=(
            "The following is a long, mostly-routine policy log. Extract three exact "
            "identifiers and report them, each on its own line:\n"
            "1. The key alias required for RESTRICTED data.\n"
            "2. The archive bucket name for backups older than 30 days.\n"
            "3. The name of the break-glass admin role.\n\n"
            "=== BEGIN DOCUMENT ===\n" + doc2 + "=== END DOCUMENT ===\n"
        ),
        grader={
            "type": "keyed_facts",
            "facts": {
                "key_alias": "k-emerald",
                "archive_bucket": "vault-nimbus-7",
                "breakglass_role": "oncall-breakglass-omega",
            },
        },  # type: ignore[arg-type]
    )
    return [task1, task2]


# --- agentic tasks ----------------------------------------------------------


def _agentic_tasks() -> list[TaskSpec]:
    triage = TaskSpec(
        id="incident-triage-plan",
        suite="agentic",
        title="Order the remediation steps for a failing deploy",
        category="planning",
        difficulty=2,
        max_turns=1,
        timeout_s=120,
        prompt=(
            "You are the on-call engineer. A new deploy of `billing-worker` is "
            "crash-looping: readiness probes fail and error rates spiked right "
            "after the rollout. You can roll back, inspect logs, and page the "
            "owner.\n\n"
            "Produce a short ordered remediation plan. Your answer MUST mention, "
            "using these exact words: `rollback` to restore the last good "
            "revision, checking `readiness` probe status, inspecting recent "
            "`logs`, and whether to `page` the service owner. Number each step."
        ),
        grader={
            "type": "keyed_facts",
            "facts": {
                "rollback": "rollback",
                "readiness": "readiness",
                "logs": "logs",
                "page": "page",
            },
        },  # type: ignore[arg-type]
    )

    tool_select = TaskSpec(
        id="tool-selection-json",
        suite="agentic",
        title="Select the right tool and emit a JSON call",
        category="tool_use",
        difficulty=2,
        max_turns=1,
        timeout_s=120,
        prompt=(
            "You have exactly these tools:\n"
            "- `search_orders(customer_id: str, since: str)`\n"
            "- `refund_order(order_id: str, amount_cents: int)`\n"
            "- `send_email(to: str, subject: str)`\n\n"
            "Goal: issue a refund of $12.50 on order `ord_88213`.\n\n"
            "Respond with a single JSON object naming the tool and its arguments, "
            'e.g. {\"tool\": \"...\", \"arguments\": {...}}. Amounts are in cents.'
        ),
        grader={
            "type": "keyed_facts",
            "facts": {
                "tool": "refund_order",
                "order_id": "ord_88213",
                "amount": "1250",
            },
        },  # type: ignore[arg-type]
    )

    repo_nav = TaskSpec(
        id="repo-root-cause",
        suite="agentic",
        title="Name the root-cause file from a repo tree and traceback",
        category="debugging",
        difficulty=3,
        max_turns=1,
        timeout_s=120,
        prompt=(
            "Repository tree:\n"
            "```\n"
            "app/\n"
            "  __init__.py\n"
            "  handlers/\n"
            "    orders.py\n"
            "    payments.py\n"
            "  services/\n"
            "    pricing.py\n"
            "    tax.py\n"
            "```\n\n"
            "Traceback:\n"
            "```\n"
            "File \"app/services/pricing.py\", line 42, in apply_discount\n"
            "    return base - base * rate\n"
            "TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'\n"
            "```\n\n"
            "Which single file must be edited to fix the root cause? Answer with the "
            "exact relative path."
        ),
        grader={
            "type": "exact_match",
            "expected": "app/services/pricing.py",
            "case_sensitive": False,
        },  # type: ignore[arg-type]
    )
    return [triage, tool_select, repo_nav]


# --- emit -------------------------------------------------------------------


def _write_suite(name: str, description: str, tasks: list[TaskSpec]) -> None:
    out = SUITES / name
    out.mkdir(parents=True, exist_ok=True)
    manifest = SuiteManifest(name=name, description=description)
    (out / "suite.yaml").write_text(
        yaml.safe_dump(manifest.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    for task in tasks:
        # Round-trip through TaskSpec to guarantee the emitted YAML validates.
        payload = task.model_dump(exclude_none=True)
        reparsed = TaskSpec.model_validate(payload)
        assert reparsed.id == task.id
        (out / f"{task.id}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        print(f"  wrote {name}/{task.id}.yaml  (grader={task.grader.type})")


def main() -> int:
    print("Building agentic suite...")
    _write_suite(
        "agentic",
        "Single-turn planning, tool-selection, and debugging tasks.",
        _agentic_tasks(),
    )
    print("Building long_context suite...")
    _write_suite(
        "long_context",
        "Needle-in-haystack retrieval from long synthesized documents.",
        _long_context_tasks(),
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
