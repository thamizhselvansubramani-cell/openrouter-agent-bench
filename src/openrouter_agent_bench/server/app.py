"""FastAPI application exposing the harness over HTTP.

Endpoints:
- ``GET  /api/health``  — liveness + config summary.
- ``GET  /api/models``  — the model catalog from ``models.yaml``.
- ``GET  /api/suites``  — loaded task suites (hidden grader material stripped).
- ``POST /api/chat``    — proxy a completion through :class:`OpenRouterClient`.

The single-page frontend lives in ``server/static`` and is served at ``/``.
"""

from __future__ import annotations

import pathlib
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from openrouter_agent_bench import __version__
from openrouter_agent_bench.client.api import OpenRouterClient
from openrouter_agent_bench.client.pricing import estimate_cost
from openrouter_agent_bench.client.schemas import ChatMessage
from openrouter_agent_bench.config import get_settings, load_env_file
from openrouter_agent_bench.models.registry import ModelRegistry
from openrouter_agent_bench.tasks.loader import (
    TaskValidationError,
    default_suites_root,
    load_suites,
)
from openrouter_agent_bench.tasks.schema import TaskSpec

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _registry() -> ModelRegistry:
    settings = get_settings()
    path = settings.models_file or _repo_root() / "models.yaml"
    return ModelRegistry.load(path)


def _safe_task(task: TaskSpec) -> dict[str, Any]:
    """Project a task to its public fields, hiding grader internals."""
    return {
        "id": task.id,
        "suite": task.suite,
        "title": task.title,
        "category": task.category,
        "difficulty": task.difficulty,
        "prompt": task.prompt,
        "max_turns": task.max_turns,
        "timeout_s": task.timeout_s,
        "target_file": task.target_file,
        "grader_type": task.grader.type,
    }


class ChatRequestBody(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)


def _pricing_map(registry: ModelRegistry) -> dict[str, tuple[float, float]]:
    return {
        spec.id: (spec.pricing.prompt_per_million, spec.pricing.completion_per_million)
        for spec in registry.all()
    }


def create_app(*, load_dotenv: bool = True) -> FastAPI:
    """Build the FastAPI application."""
    if load_dotenv:
        load_env_file(_repo_root())
    app = FastAPI(title="openrouter-agent-bench", version=__version__)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        settings = get_settings()
        return {
            "status": "ok",
            "version": __version__,
            "testing": settings.testing,
            "api_key_configured": settings.openrouter_api_key is not None,
            "base_url": settings.openrouter_base_url,
            "free_models_only": settings.free_models_only,
        }

    @app.get("/api/models")
    async def models(free_only: bool | None = Query(default=None)) -> list[dict[str, Any]]:
        settings = get_settings()
        try:
            registry = _registry()
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500, detail=f"failed to load model catalog: {exc}"
            ) from exc
        restrict_free = settings.free_models_only if free_only is None else free_only
        specs = registry.free() if restrict_free else registry.all()

        def serialize(spec: Any) -> dict[str, Any]:
            return {
                "id": spec.id,
                "display_name": spec.display_name,
                "context_window": spec.context_window,
                "max_output_tokens": spec.max_output_tokens,
                "supports_vision": spec.supports_vision,
                "supports_tools": spec.supports_tools,
                "prompt_per_million": spec.pricing.prompt_per_million,
                "completion_per_million": spec.pricing.completion_per_million,
                "is_free": spec.is_free,
            }

        return [serialize(spec) for spec in specs]

    @app.get("/api/suites")
    async def suites() -> list[dict[str, Any]]:
        root = get_settings().suites_dir or default_suites_root(_repo_root())
        try:
            loaded = load_suites(root)
        except TaskValidationError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return [
            {
                "name": suite.manifest.name,
                "description": suite.manifest.description,
                "tasks": [_safe_task(t) for t in suite.tasks],
            }
            for suite in loaded.values()
        ]

    @app.post("/api/chat")
    async def chat(body: ChatRequestBody) -> dict[str, Any]:
        settings = get_settings()
        if not settings.openrouter_api_key:
            raise HTTPException(
                status_code=503,
                detail="OPENROUTER_API_KEY is not configured; set it in .env",
            )
        registry = _registry()
        if settings.free_models_only:
            try:
                spec = registry.get(body.model)
            except KeyError:
                raise HTTPException(
                    status_code=400, detail=f"unknown model id: {body.model!r}"
                ) from None
            if not spec.is_free:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"model {body.model!r} is not a free model and "
                        "OAB_FREE_MODELS_ONLY is enabled"
                    ),
                )
        async with OpenRouterClient(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            pricing=_pricing_map(registry),
        ) as client:
            reply = await client.chat_completion(
                model=body.model,
                messages=body.messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        usage = reply.usage
        cost = usage.cost
        if cost is None and registry.has(body.model):
            spec = registry.get(body.model)
            cost = estimate_cost(
                spec.pricing.prompt_per_million,
                spec.pricing.completion_per_million,
                usage,
            )
        return {
            "id": reply.id,
            "model": reply.model,
            "content": reply.content,
            "finish_reason": reply.finish_reason,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": cost,
            },
            "latency_s": reply.latency_s,
        }

    if STATIC_DIR.is_dir():
        index_path = STATIC_DIR / "index.html"

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(index_path)

        from fastapi.staticfiles import StaticFiles

        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def run() -> None:
    """Console-script entry point: ``bench-web``."""
    import uvicorn

    load_env_file(_repo_root())
    settings = get_settings()
    uvicorn.run(
        "openrouter_agent_bench.server.app:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.testing,
    )


app = create_app()
