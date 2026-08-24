"""Tests for retry logic in the OpenRouter client."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from openrouter_agent_bench.client.api import (
    ClientAPIError,
    OpenRouterClient,
    RateLimitError,
)
from openrouter_agent_bench.client.schemas import ChatMessage, RetryEvent


def _transport(
    handler: httpx.MockTransport, on_retry: Callable[[RetryEvent], None] | None = None
) -> OpenRouterClient:
    return OpenRouterClient(
        api_key="test-key",
        base_url="http://mock",
        max_retries=3,
        timeout_s=5.0,
        on_retry=on_retry,
        http_client=httpx.AsyncClient(transport=handler),
    )


def _ok_body() -> dict:
    return {
        "id": "cmpl-1",
        "model": "test/model",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.001,
        },
    }


async def test_success_first_try() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_ok_body())

    async with _transport(httpx.MockTransport(handler)) as client:
        resp = await client.chat_completion(
            model="test/model", messages=[ChatMessage(role="user", content="hi")]
        )
    assert resp.content == "hello"
    assert resp.usage.prompt_tokens == 10
    assert len(calls) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503])
async def test_retries_then_succeeds(status: int) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(status, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=_ok_body())

    events: list[RetryEvent] = []
    async with _transport(httpx.MockTransport(handler), on_retry=events.append) as client:
        resp = await client.chat_completion(
            model="test/model", messages=[ChatMessage(role="user", content="hi")]
        )
    assert resp.content == "hello"
    assert len(calls) == 3
    assert len(events) == 2
    assert all(isinstance(e, RetryEvent) for e in events)


async def test_retry_honors_retry_after_header() -> None:
    waits = []

    def handler(request: httpx.Request) -> httpx.Response:
        if len(waits) == 0:
            waits.append(1)
            resp = httpx.Response(429, json={}, headers={"Retry-After": "0"})
            return resp
        return httpx.Response(200, json=_ok_body())

    async with _transport(httpx.MockTransport(handler)) as client:
        resp = await client.chat_completion(
            model="test/model", messages=[ChatMessage(role="user", content="hi")]
        )
    assert resp.content == "hello"


async def test_gives_up_after_max_retries() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    async with _transport(httpx.MockTransport(handler)) as client:
        with pytest.raises(Exception):  # noqa: B017 - ServerError from tenacity reraise
            await client.chat_completion(
                model="test/model", messages=[ChatMessage(role="user", content="hi")]
            )
    assert len(calls) == 3


async def test_non_retryable_4xx_raises_immediately() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    async with _transport(httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientAPIError):
            await client.chat_completion(
                model="test/model", messages=[ChatMessage(role="user", content="hi")]
            )
    assert len(calls) == 1


def test_missing_api_key_raises() -> None:
    import os

    old = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        with pytest.raises(ValueError):
            OpenRouterClient()
    finally:
        if old is not None:
            os.environ["OPENROUTER_API_KEY"] = old


async def test_cost_fallback_from_pricing_table() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _ok_body()
        del body["usage"]["cost"]
        return httpx.Response(200, json=body)

    async with _transport(httpx.MockTransport(handler)) as client:
        client.register_pricing("test/model", prompt_per_million=2.0, completion_per_million=4.0)
        resp = await client.chat_completion(
            model="test/model", messages=[ChatMessage(role="user", content="hi")]
        )
    # 10 tokens * $2/Mtok + 5 tokens * $4/Mtok = 0.00002 + 0.00002
    assert resp.usage.cost == pytest.approx(0.00004)


async def test_rate_limit_error_attributes() -> None:
    err = RateLimitError(429, "slow down")
    assert err.status_code == 429
    assert err.retry_after is None
