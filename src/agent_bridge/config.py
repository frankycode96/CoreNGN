"""Configuration management for Hybrid Agent Bridge.

Centralizes filesystem paths, Git branch scoping overrides, SQLite WAL tuning
parameters, System-1 (Jev/Local) decision thresholds, and System-2
(Ollama / llama.cpp / vLLM) endpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SQLiteConfig(BaseModel):
    """Low-level SQLite engine tuning parameters."""

    journal_mode: str = Field(default="WAL", description="SQLite journal mode (WAL mandatory for concurrency)")
    synchronous: str = Field(default="NORMAL", description="NORMAL balances durability with NVMe speed in WAL")
    busy_timeout_ms: int = Field(default=5000, ge=100, description="Lock wait timeout in milliseconds")
    cache_size_kb: int = Field(default=-64000, description="Negative value means kibibytes (64MB page cache)")
    foreign_keys: bool = Field(default=True, description="Enforce relational foreign keys")
    vector_dim: int = Field(default=64, ge=8, le=4096, description="Dimension of CPU-embedded dense vectors")


class ContextBudgetConfig(BaseModel):
    """Hard limits for Context Injector to protect local LLM KV-cache."""

    max_context_tokens: int = Field(
        default=800,
        ge=64,
        le=800,
        description="Strict upper bound on injected memory context tokens (< 800 tokens)",
    )
    chars_per_token: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Conservative character-to-token approximation ratio",
    )
    top_k_declarative: int = Field(default=12, ge=1, le=50)
    top_k_episodic: int = Field(default=4, ge=1, le=20)
    bm25_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    @property
    def max_context_chars(self) -> int:
        """Maximum characters allowed in the injected memory payload."""
        return self.max_context_tokens * self.chars_per_token


class DecisionConfig(BaseModel):
    """System-1 Decision Controller configuration (Jev-Mem pattern)."""

    backend: Literal["local_embedded", "typesafe_jev"] = Field(
        default="local_embedded",
        description="Active System-1 driver",
    )
    jev_api_url: str = Field(
        default="https://api.jev-typesafe.local/v1",
        description="Endpoint for remote/sidecar Jev TypeSafe engine",
    )
    jev_api_key: str | None = Field(default=None, description="Optional Bearer token for Jev API")
    jev_timeout_seconds: float = Field(default=5.0, ge=0.5, le=30.0)
    onnx_model_path: Path | None = Field(
        default=None,
        description="Optional path to quantized ONNX classification model for CPU inference",
    )
    gatekeeper_min_score: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum normalized Score(0..1) required by System1_Gatekeeper to accept an artifact",
    )
    stopping_min_noul: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum Noul(0..1) truth probability that sub-worker task has converged",
    )


class LLMConfig(BaseModel):
    """System-2 Local Generative Inference configuration (Ollama / llama.cpp / vLLM)."""

    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible local inference server URL",
    )
    model: str = Field(
        default="qwen2.5-coder:7b",
        description="Model identifier served by Ollama, llama-server, or vLLM",
    )
    api_key: str = Field(default="local-inference", description="API key placeholder for local endpoints")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1536, ge=64, le=8192)
    request_timeout_seconds: float = Field(default=45.0, ge=1.0, le=120.0)


class SubWorkerConfig(BaseModel):
    """Deterministic Sub-Worker Isolation parameters."""

    max_rounds: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Strict upper limit on internal exploratory sub-worker rounds (max_round <= 3)",
    )
    timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        le=120.0,
        description="Mandatory wall-clock timeout for the isolated sub-worker cluster",
    )
    max_gatekeeper_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum macro-FSM retries when System1_Gatekeeper rejects an artifact",
    )


class BridgeConfig(BaseModel):
    """Root configuration object for the Hybrid Agent Bridge."""

    db_path: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("AGENT_BRIDGE_DB_PATH", "~/.agent_bridge/memory.db")
        ).expanduser(),
        description="Unified SQLite persistence file path",
    )
    repo_root: Path = Field(
        default_factory=lambda: Path(os.environ.get("AGENT_BRIDGE_REPO_ROOT", os.getcwd())).resolve(),
        description="Repository root path for Repo Truth Hierarchy and Git scoping",
    )
    git_branch_override: str | None = Field(
        default_factory=lambda: os.environ.get("AGENT_BRIDGE_GIT_BRANCH"),
        description="Optional explicit Git branch override for memory scoping",
    )
    repo_id_override: str | None = Field(
        default_factory=lambda: os.environ.get("AGENT_BRIDGE_REPO_ID"),
        description="Optional explicit repository identifier override",
    )
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    budget: ContextBudgetConfig = Field(default_factory=ContextBudgetConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    subworker: SubWorkerConfig = Field(default_factory=SubWorkerConfig)

    @field_validator("db_path", mode="before")
    @classmethod
    def _expand_db_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("repo_root", mode="before")
    @classmethod
    def _expand_repo_root(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @classmethod
    def from_env(cls) -> BridgeConfig:
        """Construct configuration overridden by environment variables."""
        db_path = os.environ.get("AGENT_BRIDGE_DB_PATH", "~/.agent_bridge/memory.db")
        repo_root = os.environ.get("AGENT_BRIDGE_REPO_ROOT", os.getcwd())
        llm_base_url = os.environ.get("AGENT_BRIDGE_LLM_BASE_URL", "http://localhost:11434/v1")
        llm_model = os.environ.get("AGENT_BRIDGE_LLM_MODEL", "qwen2.5-coder:7b")
        decision_backend = os.environ.get("AGENT_BRIDGE_DECISION_BACKEND", "local_embedded")

        return cls(
            db_path=Path(db_path),
            repo_root=Path(repo_root),
            git_branch_override=os.environ.get("AGENT_BRIDGE_GIT_BRANCH"),
            repo_id_override=os.environ.get("AGENT_BRIDGE_REPO_ID"),
            llm=LLMConfig(base_url=llm_base_url, model=llm_model),
            decision=DecisionConfig(
                backend="typesafe_jev" if decision_backend == "typesafe_jev" else "local_embedded",
                jev_api_url=os.environ.get("AGENT_BRIDGE_JEV_URL", "https://api.jev-typesafe.local/v1"),
                jev_api_key=os.environ.get("AGENT_BRIDGE_JEV_API_KEY"),
            ),
        )
