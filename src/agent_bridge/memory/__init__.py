"""Unified 4-Tier Memory Subsystem for Hybrid Agent Bridge."""

from agent_bridge.memory.declarative import (
    DeclarativeFact,
    DeclarativeMemoryStore,
    PinnedMemoryError,
)
from agent_bridge.memory.engine import (
    LocalMemoryEngine,
    deserialize_f32_vector,
    detect_sqlite_vec_platform_guidance,
    serialize_f32_vector,
)
from agent_bridge.memory.episodic import (
    CPUEmbedder,
    EpisodicArtifact,
    EpisodicMemoryStore,
    compute_recency_decay,
)
from agent_bridge.memory.git_context import GitContextInfo, resolve_git_context
from agent_bridge.memory.injector import (
    ContextInjector,
    InjectedContextBundle,
    estimate_tokens,
    inject_bounded_context,
)

__all__ = [
    "CPUEmbedder",
    "ContextInjector",
    "DeclarativeFact",
    "DeclarativeMemoryStore",
    "EpisodicArtifact",
    "EpisodicMemoryStore",
    "GitContextInfo",
    "InjectedContextBundle",
    "LocalMemoryEngine",
    "PinnedMemoryError",
    "compute_recency_decay",
    "deserialize_f32_vector",
    "detect_sqlite_vec_platform_guidance",
    "estimate_tokens",
    "inject_bounded_context",
    "resolve_git_context",
    "serialize_f32_vector",
]
