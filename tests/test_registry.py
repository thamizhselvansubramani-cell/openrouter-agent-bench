"""Tests for the model registry."""

from __future__ import annotations

import textwrap

import pytest

from openrouter_agent_bench.models.registry import ModelRegistry


def _write(tmp_path: object, content: str) -> str:
    p = tmp_path / "models.yaml"  # type: ignore[operator]
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


VALID = """
models:
  - id: a/model-1
    display_name: Model One
    context_window: 128000
    max_output_tokens: 4096
    pricing:
      prompt_per_million: 1.0
      completion_per_million: 2.0
  - id: b/model-2
    display_name: Model Two
    context_window: 1000000
    max_output_tokens: 8192
    supports_tools: false
    pricing:
      prompt_per_million: 0.0
      completion_per_million: 0.0
"""


def test_load_valid_registry(tmp_path: object) -> None:
    reg = ModelRegistry.load(_write(tmp_path, VALID))
    assert len(reg) == 2
    spec = reg.get("a/model-1")
    assert spec.display_name == "Model One"
    assert spec.context_window == 128000
    assert spec.supports_vision is False
    assert reg.get("b/model-2").supports_tools is False


def test_unknown_model_raises(tmp_path: object) -> None:
    reg = ModelRegistry.load(_write(tmp_path, VALID))
    with pytest.raises(KeyError):
        reg.get("nope/missing")


def test_duplicate_ids_rejected(tmp_path: object) -> None:
    dup = """
    models:
      - id: x/y
        display_name: A
        context_window: 1000
        max_output_tokens: 100
        pricing: {prompt_per_million: 1, completion_per_million: 1}
      - id: x/y
        display_name: B
        context_window: 1000
        max_output_tokens: 100
        pricing: {prompt_per_million: 1, completion_per_million: 1}
    """
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry.load(_write(tmp_path, dup))


def test_negative_context_window_rejected(tmp_path: object) -> None:
    bad = """
    models:
      - id: x/y
        display_name: A
        context_window: -5
        max_output_tokens: 100
        pricing: {prompt_per_million: 1, completion_per_million: 1}
    """
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        ModelRegistry.load(_write(tmp_path, bad))


def test_missing_models_list_rejected(tmp_path: object) -> None:
    with pytest.raises(ValueError, match="'models' list"):
        ModelRegistry.load(_write(tmp_path, "foo: bar"))


def test_extra_field_rejected(tmp_path: object) -> None:
    bad = """
    models:
      - id: x/y
        display_name: A
        context_window: 1000
        max_output_tokens: 100
        bogus_field: true
        pricing: {prompt_per_million: 1, completion_per_million: 1}
    """
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        ModelRegistry.load(_write(tmp_path, bad))


def test_shipped_default_registry() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    reg = ModelRegistry.load(root / "models.yaml")
    assert reg.has("stealth/ox-alpha")
    ids = reg.ids()
    assert len(ids) >= 4


def test_is_free_detection(tmp_path: object) -> None:
    reg = ModelRegistry.load(_write(tmp_path, VALID))
    assert not reg.get("a/model-1").is_free
    free_model = reg.get("b/model-2")
    assert free_model.is_free
    free_ids = {spec.id for spec in reg.free()}
    assert "b/model-2" in free_ids
    assert "a/model-1" not in free_ids


def test_shipped_registry_has_free_models() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    reg = ModelRegistry.load(root / "models.yaml")
    free = reg.free()
    assert free
    assert all(spec.is_free for spec in free)
    assert any(spec.id.endswith(":free") for spec in free)
