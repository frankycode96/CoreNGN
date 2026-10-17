"""Hybrid Agent Bridge (`agent_bridge`).

Zero-Container Local-First 4-Tier SQLite Memory Engine paired with a System-1/System-2
Hierarchical LangGraph Orchestrator for constrained local inference (Ollama / llama.cpp / vLLM).
"""

from agent_bridge.config import (
    BridgeConfig,
    ContextBudgetConfig,
    DecisionConfig,
    LLMConfig,
    SQLiteConfig,
    SubWorkerConfig,
)
from agent_bridge.decision import (
    ChoiceResult,
    DecisionEngine,
    GatekeeperVerdict,
    LocalHeuristicDecisionEngine,
    NoulResult,
    ScoreResult,
    TypeSafeJevClient,
    create_decision_engine,
)
from agent_bridge.memory import (
    CPUEmbedder,
    ContextInjector,
    DeclarativeFact,
    DeclarativeMemoryStore,
    EpisodicArtifact,
    EpisodicMemoryStore,
    InjectedContextBundle,
    LocalMemoryEngine,
    estimate_tokens,
    inject_bounded_context,
)
from agent_bridge.orchestrator import (
    BridgeGraphState,
    HybridAgentBridgeRunner,
    IsolatedSubWorker,
    SubWorkerOutput,
)

__version__ = "0.1.0"

__all__ = [
    "BridgeConfig",
    "BridgeGraphState",
    "CPUEmbedder",
    "ChoiceResult",
    "ContextBudgetConfig",
    "ContextInjector",
    "DecisionConfig",
    "DecisionEngine",
    "DeclarativeFact",
    "DeclarativeMemoryStore",
    "EpisodicArtifact",
    "EpisodicMemoryStore",
    "GatekeeperVerdict",
    "HybridAgentBridgeRunner",
    "InjectedContextBundle",
    "IsolatedSubWorker",
    "LLMConfig",
    "LocalHeuristicDecisionEngine",
    "LocalMemoryEngine",
    "NoulResult",
    "SQLiteConfig",
    "ScoreResult",
    "SubWorkerConfig",
    "SubWorkerOutput",
    "TypeSafeJevClient",
    "__version__",
    "create_decision_engine",
    "estimate_tokens",
    "inject_bounded_context",
]
