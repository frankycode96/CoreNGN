"""Deterministic integration tests for Phase 3 & 4: LangGraph Bridge, Sub-Worker Isolation, and CLI."""

from __future__ import annotations

from pathlib import Path

from agent_bridge.cli import main as cli_main
from agent_bridge.config import BridgeConfig, SubWorkerConfig
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore
from agent_bridge.orchestrator.graph import HybridAgentBridgeRunner


def test_subworker_internal_multi_round_convergence_and_scratchpad_cleanup(tmp_path: Path) -> None:
    """Verify SubWorker refines a broken Round 1 draft in Round 2 using System-1 Noul and destroys scratchpad."""
    cfg = BridgeConfig(
        db_path=tmp_path / "bridge.db",
        repo_root=tmp_path,
        subworker=SubWorkerConfig(max_rounds=3, timeout_seconds=30.0, max_gatekeeper_retries=2),
    )

    async def _two_round_llm(messages: list[dict[str, str]], round_idx: int) -> str:
        if round_idx == 1:
            # Round 1 returns a stub with a TODO placeholder
            return "```python\ndef normalize_vector(vec):\n    # TODO implement normalization\n    pass\n```"
        # Round 2 sees the System-1 Critic message in the private scratchpad and returns valid code
        assert any("SYSTEM-1 CRITIC ROUND 1" in m["content"] for m in messages)
        return (
            "```python\n"
            "import math\n\n"
            "def normalize_vector(vec: list[float]) -> list[float]:\n"
            "    norm = math.sqrt(sum(x * x for x in vec)) or 1.0\n"
            "    return [x / norm for x in vec]\n"
            "```"
        )

    runner = HybridAgentBridgeRunner(config=cfg, llm_callable=_two_round_llm)
    state = runner.run(
        task="Implement Python function normalize_vector to L2-normalize a float vector",
        thread_id="test-thread-subworker",
        declarative_updates={"math": {"norm_type": "L2"}},
    )

    assert state["status"] == "consolidated"
    assert state["subworker_rounds_used"] == 2
    assert state["scratchpad_destroyed"] is True
    assert state["gatekeeper_passed"] is True
    assert state["consolidated_episode_id"] is not None
    assert state["context_tokens"] < 800

    # Verify episode is persisted in SQLite FTS5 & vector table
    episodic = EpisodicMemoryStore(runner.engine)
    retrieved = episodic.retrieve_hybrid("normalize_vector L2 float", top_k=1)
    assert len(retrieved) == 1
    assert "def normalize_vector" in retrieved[0].final_artifact


def test_macro_fsm_gatekeeper_retry_loop(tmp_path: Path) -> None:
    """Verify System1_Gatekeeper triggers a macro-FSM retry back to SubWorker_Node when needed."""
    cfg = BridgeConfig(
        db_path=tmp_path / "macro_retry.db",
        repo_root=tmp_path,
        subworker=SubWorkerConfig(max_rounds=1, timeout_seconds=30.0, max_gatekeeper_retries=2),
    )

    invocation_counter = {"count": 0}

    async def _macro_retry_llm(messages: list[dict[str, str]], round_idx: int) -> str:
        invocation_counter["count"] += 1
        if invocation_counter["count"] == 1:
            # First SubWorker_Node execution returns a syntax error (max_rounds=1 so it exits to Gatekeeper)
            return "```python\ndef broken_syntax(x:\n    return x\n```"
        # Second SubWorker_Node execution receives PREVIOUS GATEKEEPER FEEDBACK and fixes it
        assert any("PREVIOUS GATEKEEPER FEEDBACK" in m["content"] for m in messages)
        return (
            "```python\n"
            "def fixed_syntax_parser(payload: str) -> dict[str, str]:\n"
            '    return {"parsed": payload.strip()}\n'
            "```"
        )

    runner = HybridAgentBridgeRunner(config=cfg, llm_callable=_macro_retry_llm)
    state = runner.run(task="Write Python function fixed_syntax_parser to parse payload string")

    assert state["status"] == "consolidated"
    assert state["retry_count"] == 1
    assert state["gatekeeper_passed"] is True
    assert state["consolidated_episode_id"] == 1

    node_sequence = [entry["node"] for entry in state["audit_trail"]]
    assert node_sequence == [
        "Context_Loader",
        "SubWorker_Node",
        "System1_Gatekeeper",
        "SubWorker_Node",
        "System1_Gatekeeper",
        "Episodic_Consolidator",
    ]


def test_cli_init_upsert_run_and_inspect(tmp_path: Path) -> None:
    db_file = str(tmp_path / "cli_test.db")

    assert cli_main(["--db-path", db_file, "init", "--seed-defaults"]) == 0
    assert cli_main(["--db-path", db_file, "memory", "upsert", "runtime", "inference_engine", "ollama-qwen2.5"]) == 0
    assert (
        cli_main(
            [
                "--db-path",
                db_file,
                "run",
                "Create a deterministic hash_identifier utility function in Python",
                "--fact",
                "security:hash_algo=blake2b",
            ]
        )
        == 0
    )
    assert cli_main(["--db-path", db_file, "memory", "inspect", "--query", "hash_identifier"]) == 0

    engine = LocalMemoryEngine(db_path=db_file)
    stats = engine.inspect_stats()
    assert stats["declarative_facts_count"] >= 5
    assert stats["episodic_artifacts_count"] == 1
