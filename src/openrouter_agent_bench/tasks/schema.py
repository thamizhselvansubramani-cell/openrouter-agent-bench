"""Pydantic schemas for benchmark task definitions."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Difficulty = Annotated[int, Field(ge=1, le=5)]


class ExactMatchGrader(BaseModel):
    """Case-normalized exact match against an expected answer."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["exact_match"] = "exact_match"
    expected: str
    case_sensitive: bool = False
    #: Substrings that must all appear in the answer (used for keyed facts).
    must_include: list[str] = Field(default_factory=list)


class UnitTestGrader(BaseModel):
    """Run hidden pytest files inside the sandbox against the submission."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["unit_tests"] = "unit_tests"
    #: Hidden test file path -> contents. Never shown to the model.
    tests: dict[str, str]
    #: Extra pip-style deps the sandbox should assume are available.
    requirements: list[str] = Field(default_factory=list)


class LLMJudgeGrader(BaseModel):
    """Rubric-based LLM-as-judge scoring on a 1-5 scale."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["llm_judge"] = "llm_judge"
    rubric_id: str
    min_pass_score: float = Field(default=4.0, ge=1.0, le=5.0)
    judge_repeats: int = Field(default=2, ge=1)


class KeyedFactsGrader(BaseModel):
    """Check a set of keyed facts (needle answers) in a long-context reply."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["keyed_facts"] = "keyed_facts"
    facts: dict[str, str]


Grader = Annotated[
    ExactMatchGrader | UnitTestGrader | LLMJudgeGrader | KeyedFactsGrader,
    Field(discriminator="type"),
]


class TaskSpec(BaseModel):
    """A declarative benchmark task."""

    model_config = ConfigDict(extra="forbid")

    id: str
    suite: Literal["coding", "agentic", "long_context"]
    title: str
    category: str
    difficulty: Difficulty
    prompt: str
    #: Workspace fixtures written into the sandbox before the run.
    files: dict[str, str] = Field(default_factory=dict)
    grader: Grader
    max_turns: int = Field(default=1, ge=1)
    timeout_s: float = Field(default=600.0, gt=0)
    #: For single-file coding tasks: the file the model must return.
    target_file: str | None = None
    #: Name of the programmatic context generator (long_context suite only).
    generator: str | None = None
    #: Approximate generated context size in tokens (long_context suite only).
    context_tokens: int | None = None


class SuiteManifest(BaseModel):
    """Metadata describing a task suite directory."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
