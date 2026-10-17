"""Hybrid Agent Bridge (`agent_bridge`).

Zero-Container Local-First 4-Tier SQLite Memory Engine paired with a System-1/System-2
Hierarchical LangGraph Orchestrator, Git-Scoped Curation Toolkit, Zero-Bloat Textual TUI,
and On-Device Benchmark Harness for constrained local inference (Ollama / llama.cpp / vLLM).
"""

from agent_bridge.benchmark import (
    BenchmarkRunner,
    BenchmarkSuiteReport,
    FootprintTelemetry,
    LatencyProfile,
    ThroughputMetrics,
)
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
    ExplainRetrievalReport,
    ExplainedMemoryCandidate,
    ForgetResult,
    GitContextInfo,
    InjectedContextBundle,
    LocalMemoryEngine,
    MemoryCurator,
    PinnedMemoryError,
    PruneResult,
    compute_recency_decay,
    detect_sqlite_vec_platform_guidance,
    estimate_tokens,
    inject_bounded_context,
    resolve_git_context,
)
from agent_bridge.orchestrator import (
    BridgeGraphState,
    HybridAgentBridgeRunner,
    IsolatedSubWorker,
    SubWorkerOutput,
)
from agent_bridge.runtime import (
    ProviderProbeResult,
    RuntimeDetector,
    RuntimeDiagnosticReport,
)
from agent_bridge.ui import (
    AgentBridgeTUI,
    compute_repo_truth_diffs,
    render_rich_dashboard_snapshot,
)

__version__ = "0.2.0"

__all__ = [
    "AgentBridgeTUI",
    "BenchmarkRunner",
    "BenchmarkSuiteReport",
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
    "ExplainRetrievalReport",
    "ExplainedMemoryCandidate",
    "FootprintTelemetry",
    "ForgetResult",
    "GatekeeperVerdict",
    "GitContextInfo",
    "HybridAgentBridgeRunner",
    "InjectedContextBundle",
    "IsolatedSubWorker",
    "LLMConfig",
    "LatencyProfile",
    "LocalHeuristicDecisionEngine",
    "LocalMemoryEngine",
    "MemoryCurator",
    "NoulResult",
    "PinnedMemoryError",
    "ProviderProbeResult",
    "PruneResult",
    "RuntimeDetector",
    "RuntimeDiagnosticReport",
    "SQLiteConfig",
    "ScoreResult",
    "SubWorkerConfig",
    "SubWorkerOutput",
    "ThroughputMetrics",
    "TypeSafeJevClient",
    "__version__",
    "compute_recency_decay",
    "compute_repo_truth_diffs",
    "create_decision_engine",
    "detect_sqlite_vec_platform_guidance",
    "estimate_tokens",
    "inject_bounded_context",
    "render_rich_dashboard_snapshot",
    "resolve_git_context",
]
