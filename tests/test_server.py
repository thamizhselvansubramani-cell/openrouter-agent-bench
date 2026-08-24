"""Tests for the FastAPI server endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from openrouter_agent_bench.client.schemas import CompletionResponse, Usage
from openrouter_agent_bench.server.app import create_app


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    app = create_app(load_dotenv=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


async def test_health(client: httpx.AsyncClient) -> None:
    res = await client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["api_key_configured"] is False


async def test_models_endpoint(client: httpx.AsyncClient) -> None:
    res = await client.get("/api/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) > 0
    entry = models[0]
    for key in ("id", "display_name", "context_window", "prompt_per_million"):
        assert key in entry


async def test_suites_hide_grader_material(client: httpx.AsyncClient) -> None:
    res = await client.get("/api/suites")
    assert res.status_code == 200
    suites = res.json()
    assert any(s["name"] == "coding" for s in suites)
    coding = next(s for s in suites if s["name"] == "coding")
    assert coding["tasks"]
    task: dict[str, Any] = coding["tasks"][0]
    assert "prompt" in task
    assert "grader" not in task
    assert task["grader_type"] in {"exact_match", "unit_tests", "llm_judge", "keyed_facts"}


async def test_chat_requires_api_key(client: httpx.AsyncClient) -> None:
    res = await client.post(
        "/api/chat",
        json={"model": "stealth/ox-alpha", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert res.status_code == 503
    assert "OPENROUTER_API_KEY" in res.json()["detail"]


async def test_models_default_restricted_to_free(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OAB_FREE_MODELS_ONLY", "1")
    res = await client.get("/api/models")
    assert res.status_code == 200
    models = res.json()
    assert models
    assert all(m["is_free"] for m in models)
    assert any(m["id"].endswith(":free") or m["id"] == "stealth/ox-alpha" for m in models)


async def test_models_paid_allowed_when_disabled(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OAB_FREE_MODELS_ONLY", "0")
    all_res = await client.get("/api/models")
    assert all_res.status_code == 200
    assert any(not m["is_free"] for m in all_res.json())
    override_res = await client.get("/api/models", params={"free_only": True})
    assert override_res.status_code == 200
    assert all(m["is_free"] for m in override_res.json())


async def test_chat_rejects_paid_model_in_free_mode(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OAB_FREE_MODELS_ONLY", "1")
    res = await client.post(
        "/api/chat",
        json={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert res.status_code == 400
    assert "not a free model" in res.json()["detail"]


async def test_chat_with_mocked_client(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *exc_info: object) -> bool:
            return False

        async def chat_completion(self, **kwargs: Any) -> CompletionResponse:
            return CompletionResponse(
                id="fake-1",
                model=kwargs["model"],
                content="hello!",
                finish_reason="stop",
                usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
                latency_s=0.01,
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("openrouter_agent_bench.server.app.OpenRouterClient", FakeClient)
    res = await client.post(
        "/api/chat",
        json={"model": "stealth/ox-alpha", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["content"] == "hello!"
    assert body["usage"]["total_tokens"] == 7
