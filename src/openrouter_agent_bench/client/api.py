"""Async OpenRouter client wrapping the OpenAI-compatible Chat Completions API.

Design notes:
- Retries use ``tenacity``: exponential backoff capped at ``max_wait``, with
  server-provided ``Retry-After`` taking precedence when present.
- Only transient failures are retried: HTTP 429, the 5xx family returned by
  OpenRouter (including edge/proxy codes 524/529), and transport errors.
  Other 4xx responses surface immediately as :class:`ClientAPIError`.
- Usage is captured from ``response.usage`` (OpenRouter includes ``cost``
  when accounting is available); otherwise cost is estimated from a
  registered price table.
- Streaming follows SSE conventions and yields text deltas followed by one
  aggregated :class:`CompletionResponse` carrying full usage.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
)

from openrouter_agent_bench.client.pricing import estimate_cost
from openrouter_agent_bench.client.schemas import (
    ChatMessage,
    CompletionResponse,
    RetryEvent,
    ToolCall,
    ToolCallFunction,
    ToolSpec,
    Usage,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 524, 529})
_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"


class APIError(Exception):
    """Base error for OpenRouter API failures."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.retry_after: float | None = None


class RateLimitError(APIError):
    """HTTP 429 from the provider."""


class ServerError(APIError):
    """5xx-class error from the provider."""


class ClientAPIError(APIError):
    """Non-retryable 4xx error."""


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, RateLimitError | ServerError | httpx.TransportError)


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return None


class BackoffPolicy:
    """Exponential backoff that prefers a server-provided ``Retry-After``."""

    def __init__(
        self, multiplier: float = 1.0, min_wait: float = 1.0, max_wait: float = 60.0
    ) -> None:
        self.multiplier = multiplier
        self.min_wait = min_wait
        self.max_wait = max_wait

    def compute(self, state: RetryCallState) -> float:
        outcome = state.outcome
        exc = outcome.exception() if outcome is not None and outcome.failed else None
        retry_after = getattr(exc, "retry_after", None)
        if isinstance(retry_after, int | float) and retry_after > 0:
            return min(float(retry_after), self.max_wait)
        delay = self.multiplier * float(2 ** max(int(state.attempt_number) - 1, 0))
        return min(max(delay, self.min_wait), self.max_wait)
    def __call__(self, state: RetryCallState) -> float:
        return self.compute(state)


@dataclass(frozen=True)
class ChatRequestParams:
    """Parameters for a chat completion request."""

    model: str
    temperature: float | None = 0.0
    max_tokens: int | None = None
    seed: int | None = None
    tools: tuple[ToolSpec, ...] | None = None
    tool_choice: str | None = None

    def to_payload(self, *, stream_options: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model}
        optional: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        payload.update({k: v for k, v in optional.items() if v is not None})
        if self.tools:
            payload["tools"] = [t.model_dump() for t in self.tools]
            if self.tool_choice is not None:
                payload["tool_choice"] = self.tool_choice
        if stream_options is not None:
            payload["stream"] = True
            payload["stream_options"] = stream_options
        return payload


def parse_usage(raw: object) -> Usage:
    """Extract harness-known usage fields from an arbitrary API mapping."""
    fields: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key in Usage.model_fields:
            value = raw.get(key)
            if isinstance(value, int | float):
                fields[key] = value
    return Usage(**fields)


def parse_response(data: dict[str, Any], latency_s: float) -> CompletionResponse:
    """Normalize a chat.completion payload into :class:`CompletionResponse`."""
    choices = data.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message") or {}
    tool_calls_raw = message.get("tool_calls")
    tool_calls = [
        ToolCall(
            id=t.get("id", ""),
            type="function",
            function=ToolCallFunction(
                name=(t.get("function") or {}).get("name", ""),
                arguments=(t.get("function") or {}).get("arguments", ""),
            ),
        )
        for t in tool_calls_raw or []
    ]
    return CompletionResponse(
        id=data.get("id"),
        model=data.get("model"),
        content=message.get("content"),
        reasoning_content=message.get("reasoning_content") or message.get("reasoning"),
        tool_calls=tool_calls or None,
        finish_reason=choice.get("finish_reason"),
        usage=parse_usage(data.get("usage")),
        latency_s=latency_s,
    )


class OpenRouterClient:
    """Thin, typed async client for the OpenRouter Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 180.0,
        max_retries: int = 5,
        on_retry: Callable[[RetryEvent], None] | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenRouter API key provided. Set OPENROUTER_API_KEY or pass api_key."
            )
        self._api_key = resolved_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._on_retry = on_retry
        #: model id -> (prompt $/Mtok, completion $/Mtok); fallback when the
        #: API response carries no explicit cost.
        self._pricing = dict(pricing) if pricing else {}
        self.throttle_events: list[RetryEvent] = []
        self._client = http_client

    @property
    def http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def register_pricing(
        self, model_id: str, prompt_per_million: float, completion_per_million: float
    ) -> None:
        """Register fallback per-million-token prices for a model."""
        self._pricing[model_id] = (prompt_per_million, completion_per_million)

    def _record_retry(self, state: RetryCallState) -> None:
        outcome = state.outcome
        exc = outcome.exception() if outcome is not None and outcome.failed else None
        status = getattr(exc, "status_code", None)
        event = RetryEvent(
            attempt=max(state.attempt_number, 1),
            status_code=status if isinstance(status, int) else None,
            wait_s=state.next_action.sleep if state.next_action else 0.0,
            reason=type(exc).__name__ if exc is not None else "unknown",
        )
        self.throttle_events.append(event)
        if self._on_retry is not None:
            self._on_retry(event)

    def _apply_cost(self, requested_model: str, parsed: CompletionResponse) -> None:
        if parsed.usage.cost is not None:
            return
        prices = self._pricing.get(requested_model)
        if prices is not None:
            parsed.usage.cost = estimate_cost(prices[0], prices[1], parsed.usage)

    async def _post_once(
        self, params: ChatRequestParams, messages: Sequence[ChatMessage]
    ) -> httpx.Response:
        body = params.to_payload()
        body["messages"] = [m.model_dump(exclude_none=True) for m in messages]
        return await self.http.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(self._api_key),
            json=body,
        )

    async def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | None = None,
    ) -> CompletionResponse:
        """Send a non-streaming chat completion request with retries."""
        params = ChatRequestParams(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            tools=tuple(tools) if tools else None,
            tool_choice=tool_choice,
        )
        start = time.monotonic()
        backoff = BackoffPolicy()

        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            wait=backoff,
            stop=stop_after_attempt(max(self._max_retries, 1)),
            sleep=asyncio.sleep,
            before_sleep=self._record_retry,
            reraise=True,
        ):
            with attempt:
                response = await self._post_once(params, messages)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    exc_cls = RateLimitError if response.status_code == 429 else ServerError
                    exc = exc_cls(response.status_code, response.text[:500])
                    exc.retry_after = _retry_after(response)
                    raise exc
                if response.status_code >= 400:
                    raise ClientAPIError(response.status_code, response.text[:500])
                parsed = parse_response(response.json(), time.monotonic() - start)
                self._apply_cost(model, parsed)
                return parsed
        # AsyncRetrying(reraise=True) either returns from inside the loop or
        # re-raises; this line is unreachable by construction.
        raise AssertionError("retry loop exited without returning")

    async def chat_completion_stream(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | None = None,
    ) -> AsyncIterator[CompletionResponse]:
        """Stream a completion: yields text deltas, then one aggregated final
        response carrying full usage."""
        params = ChatRequestParams(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            tools=tuple(tools) if tools else None,
            tool_choice=tool_choice,
        )
        body = params.to_payload(stream_options={"include_usage": True})
        body["messages"] = [m.model_dump(exclude_none=True) for m in messages]
        request = self.http.build_request(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers(self._api_key),
            json=body,
        )
        start = time.monotonic()
        response = await self.http.send(request, stream=True)
        try:
            response.raise_for_status()
            content_parts: list[str] = []
            finish_reason: str | None = None
            completion_id: str | None = None
            returned_model: str | None = None
            usage = Usage()
            async for line in response.aiter_lines():
                if not line.startswith(_SSE_DATA_PREFIX):
                    continue
                chunk_body = line[len(_SSE_DATA_PREFIX) :].strip()
                if not chunk_body or chunk_body == _SSE_DONE:
                    continue
                chunk = json.loads(chunk_body)
                completion_id = chunk.get("id") or completion_id
                returned_model = chunk.get("model") or returned_model
                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    usage = parse_usage(chunk_usage)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        content_parts.append(piece)
                        yield CompletionResponse(content=piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
            final = CompletionResponse(
                id=completion_id,
                model=returned_model,
                content="".join(content_parts),
                finish_reason=finish_reason,
                usage=usage,
                latency_s=time.monotonic() - start,
            )
            self._apply_cost(model, final)
            yield final
        finally:
            await response.aclose()


__all__ = [
    "DEFAULT_BASE_URL",
    "RETRYABLE_STATUS_CODES",
    "APIError",
    "BackoffPolicy",
    "ChatRequestParams",
    "ClientAPIError",
    "OpenRouterClient",
    "RateLimitError",
    "ServerError",
    "parse_response",
    "parse_usage",
]
