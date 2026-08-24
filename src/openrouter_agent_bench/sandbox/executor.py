"""Isolated code execution for grading and agent tool use.

Guarantees per execution:
- a fresh temporary workspace directory;
- hard wall-clock timeout;
- network egress blocked by default via an injected ``sitecustomize.py``
  that disables socket creation (cross-platform, no root required);
- on POSIX, memory (``RLIMIT_AS``) and CPU-time (``RLIMIT_CPU``) limits via
  a preexec hook. On Windows these two limits degrade gracefully to
  timeout-only enforcement (documented limitation).
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

BLOCK_NETWORK_SITECUSTOMIZE = '''\
"""Injected by openrouter-agent-bench: blocks outbound sockets in the sandbox.

Loopback traffic is allowed so that event loops and test runners that use
127.0.0.1 self-pipes keep working; all other egress raises.
"""
import ipaddress as _ipaddress
import socket as _socket

_MSG = "network access is disabled in the benchmark sandbox"
_ORIG_CONNECT = _socket.socket.connect
_ORIG_CONNECT_EX = _socket.socket.connect_ex
_ORIG_SENDTO = getattr(_socket.socket, "sendto", None)


def _is_loopback(address):
    host = address[0] if isinstance(address, tuple) else None
    if not isinstance(host, str):
        return False
    try:
        return _ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guard(impl):
    def wrapped(self, address, *args):  # noqa: ANN001 - signature passthrough
        if _is_loopback(address):
            return impl(self, address, *args)
        raise OSError(_MSG)

    return wrapped


_socket.socket.connect = _guard(_ORIG_CONNECT)  # type: ignore[method-assign]
_socket.socket.connect_ex = _guard(_ORIG_CONNECT_EX)  # type: ignore[method-assign]
if _ORIG_SENDTO is not None:
    _socket.socket.sendto = _guard(_ORIG_SENDTO)  # type: ignore[method-assign]
_socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(OSError(_MSG))
'''

MAX_OUTPUT_CHARS = 20_000


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one sandboxed process run."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class SandboxWorkspace:
    """A temporary directory tree handed to the sandboxed process."""

    path: pathlib.Path

    @classmethod
    def create(cls, files: dict[str, str] | None = None) -> SandboxWorkspace:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="oab-ws-"))
        ws = cls(path=tmp)
        for rel, content in (files or {}).items():
            ws.write(rel, content)
        return ws

    def write(self, rel_path: str, content: str) -> None:
        target = self.path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read(self, rel_path: str) -> str:
        return (self.path / rel_path).read_text(encoding="utf-8")

    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _clip(text: str) -> str:
    clipped = text[:MAX_OUTPUT_CHARS]
    suffix = "\n...[truncated]" if len(text) > MAX_OUTPUT_CHARS else ""
    return clipped + suffix


class SandboxExecutor:
    """Runs subprocesses inside an isolated workspace."""

    def __init__(
        self,
        timeout_s: float = 60.0,
        memory_mb: int | None = 512,
        allow_network: bool = False,
    ) -> None:
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.allow_network = allow_network

    def _build_env(self, guard_dir: pathlib.Path, extra_pythonpath: list[pathlib.Path]) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("PYTHONSTARTUP", None)
        parts = [str(guard_dir), *[str(p) for p in extra_pythonpath]]
        existing = env.get("PYTHONPATH")
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OAB_SANDBOX"] = "1"
        return env

    def _preexec(self) -> object | None:
        """POSIX-only resource limits; ``None`` on other platforms."""
        if os.name != "posix":
            return None

        memory_bytes = (self.memory_mb or 2048) * 1024 * 1024

        def apply() -> None:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            cpu_cap = max(int(self.timeout_s), 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_cap, cpu_cap + 1))

        return apply

    def run(
        self,
        argv: list[str],
        *,
        cwd: pathlib.Path,
        timeout_s: float | None = None,
    ) -> ExecutionResult:
        """Execute ``argv`` with cwd pinned inside the workspace."""
        timeout = timeout_s or self.timeout_s
        with tempfile.TemporaryDirectory(prefix="oab-guard-") as guard:
            guard_dir = pathlib.Path(guard)
            (guard_dir / "sitecustomize.py").write_text(
                BLOCK_NETWORK_SITECUSTOMIZE, encoding="utf-8"
            )
            started = time.monotonic()
            try:
                proc = subprocess.run(
                    argv,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=self._build_env(guard_dir, [cwd]),
                    preexec_fn=self._preexec(),  # type: ignore[call-arg]
                )
                return ExecutionResult(
                    exit_code=proc.returncode,
                    stdout=_clip(proc.stdout),
                    stderr=_clip(proc.stderr),
                    timed_out=False,
                    duration_s=time.monotonic() - started,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                return ExecutionResult(
                    exit_code=-1,
                    stdout=_clip(stdout),
                    stderr=_clip(stderr) + f"\n[sandbox] timed out after {timeout:.0f}s",
                    timed_out=True,
                    duration_s=time.monotonic() - started,
                )

    def run_python(
        self,
        code: str,
        *,
        workspace: SandboxWorkspace,
        args: list[str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecutionResult:
        """Run a Python snippet with the workspace as CWD and import root."""
        script = workspace.path / "_sandbox_entry.py"
        script.write_text(code, encoding="utf-8")
        try:
            return self.run(
                [sys.executable, "-X", "faulthandler", str(script), *(args or [])],
                cwd=workspace.path,
                timeout_s=timeout_s,
            )
        finally:
            script.unlink(missing_ok=True)

    def run_pytest(
        self,
        workspace: SandboxWorkspace,
        test_paths: list[str],
        *,
        timeout_s: float | None = None,
    ) -> ExecutionResult:
        """Run hidden pytest files inside the workspace."""
        argv = [
            sys.executable,
            "-m",
            "pytest",
            *test_paths,
            "-x",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ]
        return self.run(argv, cwd=workspace.path, timeout_s=timeout_s)
