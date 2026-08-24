"""Cost calculation from a model price table.

Prices are expressed in US dollars per million tokens, matching the format
used by the OpenRouter models endpoint and this harness's ``models.yaml``.
"""

from __future__ import annotations

from openrouter_agent_bench.client.schemas import Usage

PRICE_PRECISION = 1_000_000


def estimate_cost(
    prompt_per_million: float,
    completion_per_million: float,
    usage: Usage,
) -> float:
    """Estimate request cost from per-million token prices and usage counts."""
    prompt_cost = usage.prompt_tokens / PRICE_PRECISION * prompt_per_million
    completion_cost = usage.completion_tokens / PRICE_PRECISION * completion_per_million
    return round(prompt_cost + completion_cost, 10)
