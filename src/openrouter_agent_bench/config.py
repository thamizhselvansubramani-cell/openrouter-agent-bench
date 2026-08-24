"""Environment loading and runtime settings for the harness.

A tiny stdlib-only ``.env`` loader plus a :class:`Settings` facade over the
environment variables the harness understands. Existing process environment
variables always win over ``.env`` values.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

ENV_FILE_NAME = ".env"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8420

SUITES_DIR_ENV = "OAB_SUITES_DIR"
MODELS_FILE_ENV = "OAB_MODELS_FILE"


def load_env_file(
    start_dir: str | pathlib.Path | None = None, *, override: bool = False
) -> pathlib.Path | None:
    """Load ``KEY=VALUE`` pairs from the nearest ``.env`` file.

    Walks up from ``start_dir`` (default: current working directory) looking
    for a ``.env`` file. Returns the path that was loaded, or ``None``.
    Values already present in ``os.environ`` are kept unless ``override`` is
    true.
    """
    directory = pathlib.Path(start_dir) if start_dir else pathlib.Path.cwd()
    env_path: pathlib.Path | None = None
    for candidate in (directory, *directory.parents):
        probe = candidate / ENV_FILE_NAME
        if probe.is_file():
            env_path = probe
            break
    if env_path is None:
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = _strip_quotes(value.strip())
        if key and (override or key not in os.environ):
            os.environ[key] = value
    return env_path


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _env_flag(name: str, *, default: bool) -> bool:
    """Parse a boolean env var; unset values fall back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Snapshot of harness configuration read from the environment."""

    openrouter_api_key: str | None
    openrouter_base_url: str
    suites_dir: pathlib.Path | None
    models_file: pathlib.Path | None
    server_host: str
    server_port: int
    testing: bool
    free_models_only: bool


def get_settings() -> Settings:
    """Build a :class:`Settings` from the current environment."""
    suites_dir_raw = os.environ.get(SUITES_DIR_ENV, "").strip()
    models_file_raw = os.environ.get(MODELS_FILE_ENV, "").strip()
    try:
        port = int(os.environ.get("OAB_SERVER_PORT", DEFAULT_SERVER_PORT))
    except ValueError:
        port = DEFAULT_SERVER_PORT
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip()
        or DEFAULT_BASE_URL,
        suites_dir=pathlib.Path(suites_dir_raw) if suites_dir_raw else None,
        models_file=pathlib.Path(models_file_raw) if models_file_raw else None,
        server_host=os.environ.get("OAB_SERVER_HOST", DEFAULT_SERVER_HOST).strip()
        or DEFAULT_SERVER_HOST,
        server_port=port,
        testing=os.environ.get("OAB_TESTING", "").strip() == "1",
        free_models_only=_env_flag("OAB_FREE_MODELS_ONLY", default=True),
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "MODELS_FILE_ENV",
    "SUITES_DIR_ENV",
    "Settings",
    "get_settings",
    "load_env_file",
]
