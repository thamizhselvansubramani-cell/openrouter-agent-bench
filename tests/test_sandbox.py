"""Tests for the sandbox executor."""

from __future__ import annotations

from openrouter_agent_bench.sandbox.executor import SandboxExecutor, SandboxWorkspace


def test_workspace_write_and_read() -> None:
    ws = SandboxWorkspace.create({"a/b.txt": "hello"})
    try:
        assert ws.read("a/b.txt") == "hello"
        ws.write("a/b.txt", "bye")
        assert ws.read("a/b.txt") == "bye"
    finally:
        ws.cleanup()


def test_run_python_success() -> None:
    ws = SandboxWorkspace.create()
    try:
        ex = SandboxExecutor(timeout_s=30)
        result = ex.run_python("print(21 * 2)", workspace=ws)
        assert result.ok
        assert result.stdout.strip() == "42"
    finally:
        ws.cleanup()


def test_run_python_failure_exit_code() -> None:
    ws = SandboxWorkspace.create()
    try:
        ex = SandboxExecutor(timeout_s=30)
        result = ex.run_python("raise SystemExit(3)", workspace=ws)
        assert not result.ok
        assert result.exit_code == 3
    finally:
        ws.cleanup()


def test_timeout_enforced() -> None:
    ws = SandboxWorkspace.create()
    try:
        ex = SandboxExecutor(timeout_s=2)
        result = ex.run_python(
            "import time; time.sleep(10)",
            workspace=ws,
            timeout_s=1,
        )
        assert not result.ok
        assert result.timed_out
    finally:
        ws.cleanup()


def test_network_blocked() -> None:
    ws = SandboxWorkspace.create()
    try:
        ex = SandboxExecutor(timeout_s=30)
        code = (
            "import socket\n"
            "s = socket.socket()\n"
            "try:\n"
            "    s.connect(('93.184.216.34', 80))\n"
            "except OSError as e:\n"
            "    print('blocked')\n"
        )
        result = ex.run_python(code, workspace=ws)
        assert result.ok
        assert "blocked" in result.stdout
    finally:
        ws.cleanup()


def test_loopback_allowed_for_asyncio() -> None:
    """Windows proactor loop needs a loopback self-pipe; it must survive."""
    ws = SandboxWorkspace.create()
    try:
        ex = SandboxExecutor(timeout_s=30)
        code = "import asyncio\nasync def m(): await asyncio.sleep(0)\nasyncio.run(m())\nprint('loop ok')"
        result = ex.run_python(code, workspace=ws)
        assert result.ok
        assert "loop ok" in result.stdout
    finally:
        ws.cleanup()
