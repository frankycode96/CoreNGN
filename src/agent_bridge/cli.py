"""Management CLI for Hybrid Agent Bridge (`src/agent_bridge/cli.py`).

Supports the full operational lifecycle:
- `agentic init`
- `agentic run "<task>"`
- `agentic doctor`
- `agentic tui [--snapshot]` / `agentic inspect`
- `agentic memory list | inspect | upsert | pin | unpin | forget | prune | explain`
- Top-level curation shortcuts: `agentic pin`, `agentic unpin`, `agentic forget`, `agentic prune`, `agentic explain`
- `agentic benchmark` (Phase 7 harness)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from agent_bridge.config import BridgeConfig
from agent_bridge.memory.curation import MemoryCurator
from agent_bridge.memory.declarative import DeclarativeMemoryStore
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore
from agent_bridge.memory.injector import ContextInjector
from agent_bridge.orchestrator.graph import HybridAgentBridgeRunner
from agent_bridge.runtime.detector import RuntimeDetector
from agent_bridge.ui.tui import AgentBridgeTUI, render_rich_dashboard_snapshot


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
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Override active Git branch scope for memory operations",
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

    # 3. `agentic doctor`
    doc_p = subparsers.add_parser(
        "doctor",
        help="Auto-discover local runtimes (Ollama, llama.cpp, vLLM) and verify SQLite/Git environment",
    )
    doc_p.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    doc_p.add_argument("--llamacpp-url", type=str, default="http://localhost:8080")
    doc_p.add_argument("--openai-url", type=str, default="http://localhost:1234/v1")

    # 4. `agentic tui` & `agentic inspect`
    for cmd_name in ("tui", "inspect"):
        tui_p = subparsers.add_parser(cmd_name, help="Launch Zero-Bloat Terminal UI or render Rich snapshot")
        tui_p.add_argument(
            "--snapshot",
            action="store_true",
            default=(cmd_name == "inspect"),
            help="Render non-interactive Rich dashboard snapshot to stdout",
        )
        tui_p.add_argument("--query", type=str, default="", help="Optional query to preview retrieval explanation")

    # 5. Top-level curation shortcuts (`pin`, `unpin`, `forget`, `prune`, `explain`)
    pin_top = subparsers.add_parser("pin", help="Mark a declarative fact as immutable (protected from overwrites)")
    pin_top.add_argument("category", type=str)
    pin_top.add_argument("key", type=str)

    unpin_top = subparsers.add_parser("unpin", help="Restore mutable behavior to a pinned declarative fact")
    unpin_top.add_argument("category", type=str)
    unpin_top.add_argument("key", type=str)

    forget_top = subparsers.add_parser("forget", help="Atomically purge a declarative fact or episodic artifact")
    forget_top.add_argument("target", type=str, help="Target ID, 'ep:<id>', 'fact:<id>', 'category:key', or 'key'")
    forget_top.add_argument("--force", action="store_true", help="Force deletion even if entry is pinned")

    prune_top = subparsers.add_parser("prune", help="Batch garbage-collect outdated or low-salience memories")
    prune_top.add_argument("--older-than", type=int, default=30, help="Age cutoff in days (default: 30)")
    prune_top.add_argument("--min-score", type=float, default=0.5, help="Minimum salience quality score (default: 0.5)")
    prune_top.add_argument("--dry-run", action="store_true", help="Preview prunable IDs without deleting")

    explain_top = subparsers.add_parser("explain", help="Explain BM25 + CPU Vector + Recency scoring and token budget")
    explain_top.add_argument("query", type=str, help="Task or search query to explain")
    explain_top.add_argument("--top-k", type=int, default=8)

    # 6. `agentic memory ...` subcommands
    mem_p = subparsers.add_parser("memory", help="Inspect or curate the 4-tier SQLite memory engine")
    mem_sub = mem_p.add_subparsers(dest="memory_command", required=True)

    for sub_name in ("inspect", "list"):
        m_insp = mem_sub.add_parser(sub_name, help="Inspect database stats, declarative facts, and episodic artifacts")
        m_insp.add_argument("--query", type=str, default="", help="Optional query to test bounded context injection")
        m_insp.add_argument("--limit", type=int, default=10, help="Max records to display per memory tier")
        m_insp.add_argument("--all-branches", action="store_true", help="List memories across all Git branches")

    m_upsert = mem_sub.add_parser("upsert", help="Deterministically upsert a declarative fact")
    m_upsert.add_argument("category", type=str)
    m_upsert.add_argument("key", type=str)
    m_upsert.add_argument("value", type=str)
    m_upsert.add_argument("--pin", action="store_true", help="Immediately pin fact as immutable")

    m_pin = mem_sub.add_parser("pin", help="Pin a declarative fact as immutable")
    m_pin.add_argument("category", type=str)
    m_pin.add_argument("key", type=str)

    m_unpin = mem_sub.add_parser("unpin", help="Unpin a declarative fact")
    m_unpin.add_argument("category", type=str)
    m_unpin.add_argument("key", type=str)

    m_forget = mem_sub.add_parser("forget", help="Atomically delete a fact or episode")
    m_forget.add_argument("target", type=str)
    m_forget.add_argument("--force", action="store_true")

    m_prune = mem_sub.add_parser("prune", help="Prune old or low-salience memories")
    m_prune.add_argument("--older-than", type=int, default=30)
    m_prune.add_argument("--min-score", type=float, default=0.5)
    m_prune.add_argument("--dry-run", action="store_true")

    m_explain = mem_sub.add_parser("explain", help="Explain hybrid retrieval scoring breakdown")
    m_explain.add_argument("query", type=str)
    m_explain.add_argument("--top-k", type=int, default=8)

    # 7. `agentic benchmark` (Phase 7)
    bench_p = subparsers.add_parser("benchmark", help="Run on-device latency, throughput, and RSS footprint benchmark")
    bench_p.add_argument("--iterations", type=int, default=100, help="Synthetic persistence iterations (default: 100)")
    bench_p.add_argument("--format", choices=["markdown", "json", "both"], default="both")

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
    if args.branch is not None:
        config.git_branch_override = args.branch

    engine = LocalMemoryEngine(config=config)
    engine.initialize()
    decl = DeclarativeMemoryStore(engine)
    episodic = EpisodicMemoryStore(engine)
    curator = MemoryCurator(engine, decl, episodic)

    if args.command == "init":
        info = engine.initialize()
        if args.seed_defaults:
            decl.upsert_fact("architecture", "persistence_engine", "SQLite WAL + FTS5 + CPU float32 vectors", respect_pin=False)
            decl.upsert_fact("architecture", "context_budget_tokens", 800, respect_pin=False)
            decl.upsert_fact("architecture", "truth_hierarchy", "Repo Truth > Declarative > Episodic", respect_pin=False)
        print(json.dumps({"status": "initialized", **info}, indent=2))
        return 0

    if args.command == "doctor":
        detector = RuntimeDetector(
            ollama_url=args.ollama_url,
            llamacpp_url=args.llamacpp_url,
            openai_compat_url=args.openai_url,
        )
        runtime_report = detector.discover_all_sync()
        payload = {
            "sqlite_engine": engine.inspect_stats(),
            "git_context": engine.git_context.model_dump(mode="json"),
            "runtime_discovery": runtime_report.model_dump(mode="json"),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command in {"tui", "inspect"}:
        if args.snapshot or not sys.stdout.isatty():
            print(render_rich_dashboard_snapshot(engine, query=args.query))
            return 0
        app = AgentBridgeTUI(engine=engine)
        app.run()
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

        runner = HybridAgentBridgeRunner(config=config, engine=engine)
        final_state = runner.run(
            task=args.task,
            thread_id=args.thread_id,
            declarative_updates=decl_updates,
        )
        summary = {
            "thread_id": final_state.get("thread_id"),
            "git_branch": engine.git_context.git_branch,
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

    # Handle top-level or nested `memory` curation commands
    sub_cmd = args.memory_command if args.command == "memory" else args.command

    if sub_cmd == "pin":
        fact = curator.pin(args.category, args.key)
        print(json.dumps(fact.model_dump(), indent=2))
        return 0

    if sub_cmd == "unpin":
        fact = curator.unpin(args.category, args.key)
        print(json.dumps(fact.model_dump(), indent=2))
        return 0

    if sub_cmd == "forget":
        res = curator.forget(args.target, force=args.force)
        print(json.dumps(res.model_dump(), indent=2))
        return 0 if res.deleted else 1

    if sub_cmd == "prune":
        res_p = curator.prune(
            older_than_days=args.older_than,
            min_salience_score=args.min_score,
            dry_run=args.dry_run,
        )
        print(json.dumps(res_p.model_dump(), indent=2))
        return 0

    if sub_cmd == "explain":
        rep = curator.explain(args.query, top_k=args.top_k)
        print(json.dumps(rep.model_dump(), indent=2))
        return 0

    if args.command == "memory" and sub_cmd == "upsert":
        try:
            parsed_value = json.loads(args.value)
        except json.JSONDecodeError:
            parsed_value = args.value
        fact = decl.upsert_fact(args.category, args.key, parsed_value, pinned=True if args.pin else None)
        print(json.dumps(fact.model_dump(), indent=2))
        return 0

    if args.command == "memory" and sub_cmd in {"inspect", "list"}:
        stats = engine.inspect_stats()
        facts = [f.model_dump() for f in decl.list_facts(limit=args.limit, all_branches=args.all_branches)]
        episodes = [e.model_dump() for e in episodic.list_recent(limit=args.limit, all_branches=args.all_branches)]
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

    if args.command == "benchmark":
        from agent_bridge.benchmark.runner import BenchmarkRunner

        bench_runner = BenchmarkRunner(config=config)
        bench_report = bench_runner.run_suite(synthetic_iterations=args.iterations)
        if args.format in {"markdown", "both"}:
            print(bench_Runner_to_markdown := bench_runner.format_markdown(bench_report))
        if args.format in {"json", "both"}:
            print(json.dumps(bench_report.model_dump(), indent=2))
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
