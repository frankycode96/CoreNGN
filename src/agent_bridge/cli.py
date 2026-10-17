"""Management CLI for Hybrid Agent Bridge (`agentic init`, `agentic run`, `agentic memory inspect`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from agent_bridge.config import BridgeConfig
from agent_bridge.memory.declarative import DeclarativeMemoryStore
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore
from agent_bridge.memory.injector import ContextInjector
from agent_bridge.orchestrator.graph import HybridAgentBridgeRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic",
        description="Hybrid Agent Bridge: Local-First 4-Tier SQLite Memory & System-1/System-2 LangGraph Orchestrator",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override unified SQLite database path (default: ~/.agent_bridge/memory.db)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. `agentic init`
    init_p = subparsers.add_parser("init", help="Initialize the unified SQLite WAL memory database")
    init_p.add_argument("--seed-defaults", action="store_true", help="Seed baseline declarative architecture facts")

    # 2. `agentic run`
    run_p = subparsers.add_parser("run", help="Execute a task through the 4-node LangGraph Bridge FSM")
    run_p.add_argument("task", type=str, help="Task specification for the orchestrator")
    run_p.add_argument("--thread-id", type=str, default=None, help="Optional LangGraph Working Memory thread ID")
    run_p.add_argument("--base-url", type=str, default=None, help="Override OpenAI-compatible local LLM base URL")
    run_p.add_argument("--model", type=str, default=None, help="Override local LLM model identifier")
    run_p.add_argument(
        "--fact",
        action="append",
        default=[],
        help="Upsert declarative fact before run in format 'category:key=value'",
    )

    # 3. `agentic memory ...`
    mem_p = subparsers.add_parser("memory", help="Inspect or manage the 4-tier SQLite memory engine")
    mem_sub = mem_p.add_subparsers(dest="memory_command", required=True)

    inspect_p = mem_sub.add_parser("inspect", help="Inspect database stats, declarative facts, and episodic artifacts")
    inspect_p.add_argument("--query", type=str, default="", help="Optional query to test bounded context injection")
    inspect_p.add_argument("--limit", type=int, default=10, help="Max records to display per memory tier")

    upsert_p = mem_sub.add_parser("upsert", help="Deterministically upsert a declarative fact")
    upsert_p.add_argument("category", type=str)
    upsert_p.add_argument("key", type=str)
    upsert_p.add_argument("value", type=str)

    return parser


def _parse_fact_arg(raw: str) -> tuple[str, str, Any]:
    if ":" not in raw or "=" not in raw:
        raise ValueError(f"Invalid --fact format {raw!r}. Expected 'category:key=value'.")
    cat_part, rest = raw.split(":", 1)
    key_part, val_part = rest.split("=", 1)
    try:
        parsed_val = json.loads(val_part)
    except json.JSONDecodeError:
        parsed_val = val_part
    return cat_part.strip(), key_part.strip(), parsed_val


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for the `agentic` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = BridgeConfig.from_env()
    if args.db_path is not None:
        config.db_path = Path(args.db_path).expanduser().resolve()

    if args.command == "init":
        engine = LocalMemoryEngine(config=config)
        info = engine.initialize()
        if args.seed_defaults:
            decl = DeclarativeMemoryStore(engine)
            decl.upsert_fact("architecture", "persistence_engine", "SQLite WAL + FTS5 + CPU float32 vectors")
            decl.upsert_fact("architecture", "context_budget_tokens", 800)
            decl.upsert_fact("architecture", "truth_hierarchy", "Repo Truth > Declarative > Episodic")
        print(json.dumps({"status": "initialized", **info}, indent=2))
        return 0

    if args.command == "run":
        if args.base_url:
            config.llm.base_url = args.base_url
        if args.model:
            config.llm.model = args.model

        decl_updates: dict[str, dict[str, Any]] = {}
        for item in args.fact:
            cat, k, v = _parse_fact_arg(item)
            decl_updates.setdefault(cat, {})[k] = v

        runner = HybridAgentBridgeRunner(config=config)
        final_state = runner.run(
            task=args.task,
            thread_id=args.thread_id,
            declarative_updates=decl_updates,
        )
        summary = {
            "thread_id": final_state.get("thread_id"),
            "status": final_state.get("status"),
            "context_tokens": final_state.get("context_tokens"),
            "subworker_rounds_used": final_state.get("subworker_rounds_used"),
            "scratchpad_destroyed": final_state.get("scratchpad_destroyed"),
            "gatekeeper_passed": final_state.get("gatekeeper_passed"),
            "gatekeeper_score": final_state.get("gatekeeper_score"),
            "gatekeeper_noul": final_state.get("gatekeeper_noul"),
            "consolidated_episode_id": final_state.get("consolidated_episode_id"),
            "final_artifact": final_state.get("final_artifact") or final_state.get("candidate_artifact"),
            "audit_trail": final_state.get("audit_trail"),
        }
        print(json.dumps(summary, indent=2))
        return 0 if final_state.get("status") == "consolidated" else 1

    if args.command == "memory":
        engine = LocalMemoryEngine(config=config)
        engine.initialize()
        decl = DeclarativeMemoryStore(engine)
        episodic = EpisodicMemoryStore(engine)

        if args.memory_command == "upsert":
            try:
                parsed_value = json.loads(args.value)
            except json.JSONDecodeError:
                parsed_value = args.value
            fact = decl.upsert_fact(args.category, args.key, parsed_value)
            print(json.dumps(fact.model_dump(), indent=2))
            return 0

        if args.memory_command == "inspect":
            stats = engine.inspect_stats()
            facts = [f.model_dump() for f in decl.list_facts(limit=args.limit)]
            episodes = [e.model_dump() for e in episodic.list_recent(limit=args.limit)]
            report: dict[str, Any] = {
                "engine_stats": stats,
                "declarative_facts": facts,
                "episodic_artifacts": episodes,
            }
            if args.query:
                injector = ContextInjector(engine, decl, episodic)
                bundle = injector.build_context(args.query)
                report["injected_context_preview"] = {
                    "estimated_tokens": bundle.estimated_tokens,
                    "max_token_budget": bundle.max_token_budget,
                    "formatted_block": bundle.formatted_block,
                }
            print(json.dumps(report, indent=2))
            return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
