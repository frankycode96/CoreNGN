"""Rigorous TypedDict State Schema for the Hierarchical LangGraph Macro-FSM."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class AuditEntry(TypedDict):
    """Structured telemetry record emitted by each macro-FSM node transition."""

    node: str
    event: str
    details: dict[str, Any]


class BridgeGraphState(TypedDict, total=False):
    """State schema governing the 4-node Hybrid Agent Bridge StateGraph.

    Note: The isolated sub-worker scratchpad is intentionally excluded from
    this state schema to enforce strict memory isolation and prevent context bloat.
    """

    # Input specification
    task: str
    thread_id: str
    declarative_updates: dict[str, dict[str, Any]]
    tags: list[str]

    # Node 1: Context_Loader outputs
    injected_context: str
    context_tokens: int
    repo_truth_entries: list[str]

    # Node 2: SubWorker_Node outputs (converging artifact only; scratchpad destroyed)
    candidate_artifact: str
    subworker_rounds_used: int
    subworker_converged_noul: float
    scratchpad_destroyed: bool

    # Node 3: System1_Gatekeeper outputs
    gatekeeper_passed: bool
    gatekeeper_score: float
    gatekeeper_noul: float
    gatekeeper_action: Literal["accept", "retry", "abort"]
    gatekeeper_feedback: str
    retry_count: int

    # Node 4: Episodic_Consolidator outputs
    consolidated_episode_id: int | None
    extracted_facts_count: int
    final_artifact: str
    status: Literal["initialized", "context_loaded", "subworker_done", "validated", "consolidated", "aborted"]

    # Deterministic execution trace
    audit_trail: list[AuditEntry]
