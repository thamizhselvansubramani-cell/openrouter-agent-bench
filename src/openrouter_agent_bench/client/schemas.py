"""Pydantic schemas for OpenRouter chat-completion requests and responses.

Mirrors the OpenAI-compatible Chat Completions API exposed at
``https://openrouter.ai/api/v1/chat/completions``. Only the fields the
harness consumes are modeled; unknown fields in raw payloads are ignored.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool", "developer"]


class ToolCallFunction(BaseModel):
    """The function invocation inside a tool call."""

    model_config = ConfigDict(extra="ignore")

    name: str
    arguments: str = ""


class ToolCall(BaseModel):
    """A single function call emitted by a model."""

    model_config = ConfigDict(extra="ignore")

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ToolFunctionSpec(BaseModel):
    """JSON-schema description of a callable function."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})


class ToolSpec(BaseModel):
    """OpenAI-compatible tool definition."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: ToolFunctionSpec


class Usage(BaseModel):
    """Token usage and cost accounting from an API response."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None
    reasoning_tokens: int | None = None


class CompletionResponse(BaseModel):
    """Normalized completion response used throughout the harness."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    #: Model id the provider actually served. May differ from the requested id
    #: for routed endpoints (e.g. ``openrouter/free``), so it is recorded
    #: separately for provenance.
    model: str | None = None
    #: Upstream provider that served the request, when OpenRouter reports it.
    provider: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0


class RetryEvent(BaseModel):
    """Record of a throttled / retried request."""

    attempt: int
    status_code: int | None = None
    wait_s: float
    reason: str
