"""Model registry: declarative model catalog loaded from ``models.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelPricing(BaseModel):
    """Per-million-token prices in USD."""

    model_config = ConfigDict(extra="forbid")

    prompt_per_million: float = Field(ge=0)
    completion_per_million: float = Field(ge=0)


class ModelSpec(BaseModel):
    """A registered model's capabilities and pricing."""

    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    supports_vision: bool = False
    supports_tools: bool = True
    supports_seed: bool = True
    pricing: ModelPricing

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model id must not be blank")
        return v

    @property
    def is_free(self) -> bool:
        """True when the model costs nothing (``:free`` suffix or zero pricing)."""
        return self.id.endswith(":free") or (
            self.pricing.prompt_per_million == 0.0 and self.pricing.completion_per_million == 0.0
        )


class ModelRegistry:
    """Validated collection of :class:`ModelSpec` entries."""

    def __init__(self, models: list[ModelSpec]) -> None:
        by_id: dict[str, ModelSpec] = {}
        for m in models:
            if m.id in by_id:
                msg = f"duplicate model id: {m.id}"
                raise ValueError(msg)
            by_id[m.id] = m
        self._by_id = by_id

    @classmethod
    def load(cls, path: str | Path) -> ModelRegistry:
        raw_path = Path(path)
        data = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            msg = f"{raw_path}: expected top-level 'models' list"
            raise ValueError(msg)
        specs = [ModelSpec.model_validate(entry) for entry in data["models"]]
        return cls(specs)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._by_id[model_id]
        except KeyError:
            raise KeyError(f"unknown model id: {model_id!r}") from None

    def has(self, model_id: str) -> bool:
        return model_id in self._by_id

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def all(self) -> list[ModelSpec]:
        return [self._by_id[i] for i in sorted(self._by_id)]

    def free(self) -> list[ModelSpec]:
        """All zero-cost (free-tier) models, sorted by id."""
        return [spec for spec in self.all() if spec.is_free]

    def __len__(self) -> int:
        return len(self._by_id)
