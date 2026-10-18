"""The Bridge: Hierarchical LangGraph Macro-FSM Runner (`graph.py`).

Wires the 4-node deterministic orchestration topology:
1. `Context_Loader`: Pure Python extraction of Repo Truth (via RLM `peek`/`search`), Declarative Facts,
   and Episodic Memories (< 800 tokens).
2. `SubWorker_Node`: Sandboxed async sub-cluster (`<= 3` rounds, `<= 120s` timeout, immediate scratchpad destruction).
3. `System1_Gatekeeper`: Jev/Local System-1 verification (`Score` + `Noul` + 5-Layer AST Sandbox) gating retry vs consolidation.
4. `Episodic_Consolidator`: Compaction, persistence of validated `(Task, Final_Artifact)` pairs into SQLite,
   automatic semantic deduplication (`cosine >= 0.92`), and `reasoning_journal` recording.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agent_bridge.config import BridgeConfig
from agent_bridge.decision.base import DecisionEngine
from agent_bridge.decision.jev_client import create_decision_engine
from agent_bridge.memory.curation import MemoryCurator
from agent_bridge.memory.declarative import DeclarativeMemoryStore
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore
from agent_bridge.memory.injector import ContextInjector
from agent_bridge.memory.journal import ReasoningJournalStore
from agent_bridge.orchestrator.state import AuditEntry, BridgeGraphState
from agent_bridge.orchestrator.subworker import AsyncLLMCallable, IsolatedSubWorker


class HybridAgentBridgeRunner:
    """Hierarchical LangGraph Orchestrator backed by the Unified SQLite Memory Engine."""

    def __init__(
        self,
        config: BridgeConfig | None = None,
        engine: LocalMemoryEngine | None = None,
        decision_engine: DecisionEngine | None = None,
        llm_callable: AsyncLLMCallable | None = None,
        allow_offline_synthesis: bool = True,
    ) -> None:
        self.config = config or BridgeConfig.from_env()
        self.engine = engine or LocalMemoryEngine(self.config)
        self.engine.initialize()

        self.declarative = DeclarativeMemoryStore(self.engine)
        self.episodic = EpisodicMemoryStore(self.engine)
        self.journal = ReasoningJournalStore(self.engine)
        self.curator = MemoryCurator(self.engine, self.declarative, self.episodic)
        self.injector = ContextInjector(
            engine=self.engine,
            declarative_store=self.declarative,
            episodic_store=self.episodic,
        )
        self.decision = decision_engine or create_decision_engine(self.config)
        self.subworker = IsolatedSubWorker(
            config=self.config,
            decision_engine=self.decision,
            llm_callable=llm_callable,
            allow_offline_synthesis=allow_offline_synthesis,
        )

    # ------------------------------------------------------------------
    # Node 1: Context_Loader
    # ------------------------------------------------------------------
    def context_loader_node(self, state: BridgeGraphState) -> dict[str, Any]:
        """Pure Python extraction of Repo Truth, declarative facts, and salient episodes (< 800 tokens)."""
        task = state["task"]
        thread_id = state.get("thread_id") or "default-thread"

        decl_updates = state.get("declarative_updates") or {}
        upserted_count = 0
        for category, kv_map in decl_updates.items():
            if isinstance(kv_map, dict):
                for key, val in kv_map.items():
                    route = self.decision.route_memory_ingestion(
                        {"category": category, "key": key, "value": val}
                    )
                    if route.chosen != "discard":
                        self.declarative.upsert_fact(category, key, val)
                        upserted_count += 1

        bundle = self.injector.build_context(
            task=task,
            max_tokens=self.config.budget.max_context_tokens,
        )

        trail = list(state.get("audit_trail") or [])
        step_idx = len(trail) + 1
        self.journal.record_thought(
            thread_id=thread_id,
            task=task,
            step_index=step_idx,
            node_name="Context_Loader",
            thought_summary=(
                f"Assembled bounded context ({bundle.estimated_tokens}/{bundle.max_token_budget} tok): "
                f"{len(bundle.repo_truth_entries)} repo_truth, {len(bundle.included_facts)} facts, "
                f"{len(bundle.included_episodes)} episodes."
            ),
            noul_confidence=1.0,
        )

        trail.append(
            AuditEntry(
                node="Context_Loader",
                event="context_assembled",
                details={
                    "estimated_tokens": bundle.estimated_tokens,
                    "max_token_budget": bundle.max_token_budget,
                    "facts_loaded": len(bundle.included_facts),
                    "episodes_loaded": len(bundle.included_episodes),
                    "repo_truth_count": len(bundle.repo_truth_entries),
                    "upserted_facts": upserted_count,
                },
            )
        )

        return {
            "injected_context": bundle.formatted_block,
            "context_tokens": bundle.estimated_tokens,
            "repo_truth_entries": bundle.repo_truth_entries,
            "retry_count": int(state.get("retry_count") or 0),
            "extracted_facts_count": upserted_count,
            "status": "context_loaded",
            "audit_trail": trail,
        }

    # ------------------------------------------------------------------
    # Node 2: SubWorker_Node
    # ------------------------------------------------------------------
    def subworker_node(self, state: BridgeGraphState) -> dict[str, Any]:
        """Sandboxed exploratory sub-cluster with mandatory 120s timeout and scratchpad cleanup."""
        task = state["task"]
        thread_id = state.get("thread_id") or "default-thread"
        injected_context = state.get("injected_context") or ""
        feedback = state.get("gatekeeper_feedback") or ""

        output = self.subworker.execute_sync(
            task=task,
            injected_context=injected_context,
            gatekeeper_feedback=feedback,
        )

        trail = list(state.get("audit_trail") or [])
        step_idx = len(trail) + 1
        self.journal.record_thought(
            thread_id=thread_id,
            task=task,
            step_index=step_idx,
            node_name="SubWorker_Node",
            thought_summary=(
                f"Completed synthesis in {output.rounds_used} round(s) via {output.inference_mode} "
                f"(converged_noul={output.converged_noul:.2f}, scratchpad_destroyed={output.scratchpad_destroyed})."
            ),
            noul_confidence=output.converged_noul,
        )

        trail.append(
            AuditEntry(
                node="SubWorker_Node",
                event="subworker_completed",
                details={
                    "rounds_used": output.rounds_used,
                    "converged_noul": output.converged_noul,
                    "scratchpad_destroyed": output.scratchpad_destroyed,
                    "timed_out": output.timed_out,
                    "inference_mode": output.inference_mode,
                },
            )
        )

        return {
            "candidate_artifact": output.artifact,
            "subworker_rounds_used": output.rounds_used,
            "subworker_converged_noul": output.converged_noul,
            "scratchpad_destroyed": output.scratchpad_destroyed,
            "status": "subworker_done",
            "audit_trail": trail,
        }

    # ------------------------------------------------------------------
    # Node 3: System1_Gatekeeper
    # ------------------------------------------------------------------
    def system1_gatekeeper_node(self, state: BridgeGraphState) -> dict[str, Any]:
        """System-1 Gatekeeper validating syntax, 5-layer sandbox safety, and compliance via Score + Noul."""
        task = state["task"]
        thread_id = state.get("thread_id") or "default-thread"
        candidate = state.get("candidate_artifact") or ""
        retry_count = int(state.get("retry_count") or 0)
        max_retries = int(self.config.subworker.max_gatekeeper_retries)
        min_score = float(self.config.decision.gatekeeper_min_score)

        verdict = self.decision.evaluate_gatekeeper(
            task=task,
            artifact=candidate,
            retry_count=retry_count,
            max_retries=max_retries,
            min_score=min_score,
        )

        next_retry_count = retry_count + 1 if verdict.action == "retry" else retry_count
        next_status: Literal["validated", "subworker_done", "aborted"] = (
            "validated" if verdict.passed else ("aborted" if verdict.action == "abort" else "subworker_done")
        )

        trail = list(state.get("audit_trail") or [])
        step_idx = len(trail) + 1
        self.journal.record_thought(
            thread_id=thread_id,
            task=task,
            step_index=step_idx,
            node_name="System1_Gatekeeper",
            thought_summary=f"Verdict={verdict.action}: {verdict.feedback}",
            noul_confidence=verdict.compliance_probability,
        )

        trail.append(
            AuditEntry(
                node="System1_Gatekeeper",
                event=f"gatekeeper_{verdict.action}",
                details={
                    "passed": verdict.passed,
                    "score": verdict.score,
                    "compliance_noul": verdict.compliance_probability,
                    "action": verdict.action,
                    "retry_count": next_retry_count,
                    "feedback": verdict.feedback,
                },
            )
        )

        return {
            "gatekeeper_passed": verdict.passed,
            "gatekeeper_score": verdict.score,
            "gatekeeper_noul": verdict.compliance_probability,
            "gatekeeper_action": verdict.action,
            "gatekeeper_feedback": verdict.feedback,
            "retry_count": next_retry_count,
            "status": next_status,
            "audit_trail": trail,
        }

    # ------------------------------------------------------------------
    # Node 4: Episodic_Consolidator
    # ------------------------------------------------------------------
    def episodic_consolidator_node(self, state: BridgeGraphState) -> dict[str, Any]:
        """Compaction node: discards intermediate messages, saves validated artifact, and deduplicates."""
        task = state["task"]
        thread_id = state.get("thread_id") or "default-thread"
        candidate = state.get("candidate_artifact") or ""
        quality = float(state.get("gatekeeper_score") or 1.0)
        tags = list(state.get("tags") or ["validated", "langgraph-bridge"])

        route = self.decision.route_memory_ingestion({"task": task, "artifact": candidate})
        episode_id: int | None = None
        merged_duplicates = 0
        if route.chosen != "discard":
            episode = self.episodic.record_episode(
                task=task,
                final_artifact=candidate,
                tags=tags,
                quality_score=quality,
            )
            episode_id = episode.id
            consolidation = self.curator.consolidate_duplicates(similarity_threshold=0.92, dry_run=False)
            merged_duplicates = consolidation.total_merged

        trail = list(state.get("audit_trail") or [])
        step_idx = len(trail) + 1
        self.journal.record_thought(
            thread_id=thread_id,
            task=task,
            step_index=step_idx,
            node_name="Episodic_Consolidator",
            thought_summary=(
                f"Consolidated episode_id={episode_id} (quality={quality:.2f}, "
                f"auto_merged_duplicates={merged_duplicates})."
            ),
            noul_confidence=quality,
        )

        trail.append(
            AuditEntry(
                node="Episodic_Consolidator",
                event="episode_consolidated",
                details={
                    "episode_id": episode_id,
                    "routed_tier": route.chosen,
                    "quality_score": quality,
                    "auto_merged_duplicates": merged_duplicates,
                },
            )
        )

        return {
            "consolidated_episode_id": episode_id,
            "final_artifact": candidate,
            "status": "consolidated",
            "audit_trail": trail,
        }

    # ------------------------------------------------------------------
    # Graph Builder & Execution
    # ------------------------------------------------------------------
    @staticmethod
    def _route_after_gatekeeper(
        state: BridgeGraphState,
    ) -> Literal["Episodic_Consolidator", "SubWorker_Node", "__end__"]:
        action = state.get("gatekeeper_action", "abort")
        if action == "accept":
            return "Episodic_Consolidator"
        if action == "retry":
            return "SubWorker_Node"
        return "__end__"

    def build_state_graph(self) -> StateGraph:
        """Construct the uncompiled 4-node LangGraph StateGraph definition."""
        workflow = StateGraph(BridgeGraphState)

        workflow.add_node("Context_Loader", self.context_loader_node)
        workflow.add_node("SubWorker_Node", self.subworker_node)
        workflow.add_node("System1_Gatekeeper", self.system1_gatekeeper_node)
        workflow.add_node("Episodic_Consolidator", self.episodic_consolidator_node)

        workflow.add_edge(START, "Context_Loader")
        workflow.add_edge("Context_Loader", "SubWorker_Node")
        workflow.add_edge("SubWorker_Node", "System1_Gatekeeper")
        workflow.add_conditional_edges(
            "System1_Gatekeeper",
            self._route_after_gatekeeper,
            {
                "Episodic_Consolidator": "Episodic_Consolidator",
                "SubWorker_Node": "SubWorker_Node",
                "__end__": END,
            },
        )
        workflow.add_edge("Episodic_Consolidator", END)
        return workflow

    def run(
        self,
        task: str,
        thread_id: str | None = None,
        declarative_updates: dict[str, dict[str, Any]] | None = None,
        tags: list[str] | None = None,
    ) -> BridgeGraphState:
        """Execute the full 4-node macro-FSM with atomic SQLite Working Memory checkpointing."""
        active_thread_id = thread_id or f"thread-{uuid.uuid4().hex[:12]}"
        initial_state: BridgeGraphState = {
            "task": task,
            "thread_id": active_thread_id,
            "declarative_updates": declarative_updates or {},
            "tags": tags or ["validated"],
            "retry_count": 0,
            "status": "initialized",
            "audit_trail": [],
        }

        workflow = self.build_state_graph()
        with self.engine.langgraph_checkpointer() as checkpointer:
            compiled = workflow.compile(checkpointer=checkpointer)
            result = compiled.invoke(
                initial_state,
                config={"configurable": {"thread_id": active_thread_id}},
            )
            return BridgeGraphState(**result)
