"""Shared pytest fixtures."""

from __future__ import annotations

import pathlib

import pytest

from openrouter_agent_bench.config import load_env_file

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def pytest_configure(config: pytest.Config) -> None:
    """Load the repo-level ``.env`` before tests collect."""
    load_env_file(REPO_ROOT)


@pytest.fixture
def repo_root() -> pathlib.Path:
    return REPO_ROOT
