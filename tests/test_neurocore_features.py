"""Comprehensive test suite for NeuroCore-inspired enhancements (`tests/test_neurocore_features.py`).

Verifies:
1. Reciprocal Rank Fusion (RRF, k=60) in hybrid episodic retrieval (`compute_normalized_rrf`, `retrieve_hybrid`).
2. 6-Type Semantic Memory Taxonomy (`FACT`, `BELIEF`, `PREFERENCE`, `IDENTITY`, `RULE`, `EXPERIENCE`),
   automatic `RULE` pinning, `BELIEF` 30-day TTL expiration, and `memory_audit_log` tracking.
3. Deterministic Semantic Auto-Consolidation (`cosine >= 0.92`) in `MemoryCurator.consolidate_duplicates()`.
4. RLM (Recursive Language Model) Repo Truth Inspector (`RLMWorkspaceToolkit`: `peek`, `search`, `chunk`, session vars).
5. 5-Layer AST & Resource Sandbox Validator (`CodeSandboxValidator`, `SafeEnv`) integrated into System-1 Gatekeeper.
6. Structured Thought Journal (`ReasoningJournalStore`) decoupled from `episodic_artifacts`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_bridge.config import BridgeConfig
from agent_bridge.decision.local_heuristics import LocalHeuristicDecisionEngine
from agent_bridge.memory.curation import MemoryCurator
from agent_bridge.memory.declarative import DeclarativeMemoryStore, PinnedMemoryError
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore, compute_normalized_rrf
from agent_bridge.memory.journal import ReasoningJournalStore
from agent_bridge.orchestrator.graph import HybridAgentBridgeRunner
from agent_bridge.rlm.tools import RLMWorkspaceToolkit
from agent_bridge.sandbox.validator import CodeSandboxValidator, SafeEnv


def test_reciprocal_rank_fusion_scoring(tmp_path: Path) -> None:
    """Verify RRF (k=60) math and integration into hybrid episodic retrieval."""
    assert compute_normalized_rrf(1, 1, rrf_k=60) == pytest.approx(1.0, rel=1e-4)
    assert compute_normalized_rrf(1, None, w_bm25=0.5, w_vec=0.5, rrf_k=60) == pytest.approx(0.5, rel=1e-4)
    assert compute_normalized_rrf(1, 1, rrf_k=60) > compute_normalized_rrf(3, 5, rrf_k=60)

    engine = LocalMemoryEngine(db_path=tmp_path / "rrf.db")
    engine.initialize()
    episodic = EpisodicMemoryStore(engine)

    episodic.record_episode(
        task="Implement SQLite WAL connection pool with FTS5",
        final_artifact="def create_wal_pool(db_path: str) -> None:\n    return None\n",
        tags=["sqlite", "wal"],
        quality_score=0.95,
    )
    episodic.record_episode(
        task="Configure Textual CSS layout for terminal dashboard",
        final_artifact="def build_tui_css() -> str:\n    return 'Screen { layout: vertical; }'\n",
        tags=["tui", "textual"],
        quality_score=0.88,
    )

    results = episodic.retrieve_hybrid("SQLite WAL connection pool FTS5", top_k=2)
    assert len(results) >= 1
    top = results[0]
    assert top.rrf_score > 0.0
    assert top.hybrid_score > 0.0
    assert "SQLite WAL" in top.task


def test_six_type_taxonomy_ttl_and_audit_log(tmp_path: Path) -> None:
    """Verify 6-type taxonomy, RULE auto-pinning, BELIEF TTL expiration, and audit trail."""
    engine = LocalMemoryEngine(db_path=tmp_path / "taxonomy.db")
    engine.initialize()
    decl = DeclarativeMemoryStore(engine)
    episodic = EpisodicMemoryStore(engine)
    curator = MemoryCurator(engine, decl, episodic)

    # 1. Explicit RULE auto-pins
    rule_fact = decl.upsert_fact(
        "security",
        "zero_vram_persistence",
        "All embeddings must execute on CPU float32",
        memory_type="RULE",
    )
    assert rule_fact.memory_type == "RULE"
    assert rule_fact.pinned is True

    with pytest.raises(PinnedMemoryError):
        decl.upsert_fact("security", "zero_vram_persistence", "Attempted overwrite")

    # 2. BELIEF gets automatic 30-day expires_at TTL
    belief_fact = decl.upsert_fact(
        "hypothesis",
        "active_provider",
        "llama.cpp on port 8080",
        memory_type="BELIEF",
    )
    assert belief_fact.memory_type == "BELIEF"
    assert belief_fact.expires_at is not None

    # 3. Expired BELIEF is excluded from get_fact and pruned by curator
    expired_belief = decl.upsert_fact(
        "hypothesis",
        "stale_assumption",
        "temporary port 9999",
        memory_type="BELIEF",
        ttl_days=-1.0,  # Already expired in the past
    )
    assert expired_belief.expires_at is not None
    assert decl.get_fact("hypothesis", "stale_assumption") is None
    active_keys = {f.key for f in decl.list_facts(include_expired=False)}
    all_keys = {f.key for f in decl.list_facts(include_expired=True)}
    assert "stale_assumption" not in active_keys
    assert "stale_assumption" in all_keys

    prune_res = curator.prune(older_than_days=365, min_salience_score=0.1, dry_run=False)
    assert expired_belief.id in prune_res.pruned_declarative_ids

    # 4. Audit log records deterministic mutations
    audits = engine.list_audit_events(limit=20)
    actions = {a["action"] for a in audits}
    assert "upsert" in actions
    assert "prune" in actions


def test_semantic_auto_consolidation_deduplicates_episodes(tmp_path: Path) -> None:
    """Verify cosine >= 0.92 semantic auto-consolidation of near-duplicate episodic artifacts."""
    engine = LocalMemoryEngine(db_path=tmp_path / "consolidate.db")
    engine.initialize()
    decl = DeclarativeMemoryStore(engine)
    episodic = EpisodicMemoryStore(engine)
    curator = MemoryCurator(engine, decl, episodic)

    ep1 = episodic.record_episode(
        task="Create deterministic token estimator for 800 token budget",
        final_artifact="def estimate_tokens(text: str) -> int:\n    return len(text) // 4\n",
        quality_score=0.82,
    )
    # Near-identical duplicate with higher quality score
    ep2 = episodic.record_episode(
        task="Create deterministic token estimator for 800 token budget",
        final_artifact="def estimate_tokens(text: str) -> int:\n    return max(1, len(text) // 4)\n",
        quality_score=0.96,
    )
    # Distinct episode
    ep3 = episodic.record_episode(
        task="Probe Ollama and llama.cpp HTTP health endpoints",
        final_artifact="def probe_ollama() -> bool:\n    return True\n",
        quality_score=0.90,
    )

    res = curator.consolidate_duplicates(similarity_threshold=0.90, dry_run=False)
    assert res.total_merged == 1
    assert res.merged_pairs[0].merged_duplicate_id == ep1.id
    assert res.merged_pairs[0].kept_episode_id == ep2.id
    remaining_ids = {e.id for e in episodic.list_recent(limit=10)}
    assert ep1.id not in remaining_ids
    assert ep2.id in remaining_ids
    assert ep3.id in remaining_ids


def test_rlm_workspace_toolkit_peek_search_chunk(tmp_path: Path) -> None:
    """Verify surgical RLM file inspection (peek, search, chunk, session variables) and path guard."""
    sample_file = tmp_path / "src" / "sample_module.py"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"def func_{i}(x: int) -> int:\n    return x + {i}\n" for i in range(1, 45)]
    sample_file.write_text("".join(lines), encoding="utf-8")

    rlm = RLMWorkspaceToolkit(repo_root=tmp_path)

    # 1. Peek bounded lines
    snippet = rlm.peek("src/sample_module.py", start_line=1, end_line=10)
    assert snippet.exists is True
    assert snippet.start_line == 1
    assert snippet.end_line == 10
    assert "def func_1" in snippet.snippet

    # 2. Regex search across workspace
    hits = rlm.search(r"def func_25\(", max_matches=5)
    assert len(hits) == 1
    assert hits[0].rel_path == "src/sample_module.py"
    assert "def func_25" in hits[0].line_text

    # 3. Deterministic chunking
    chunks = rlm.chunk("src/sample_module.py", chunk_lines=20, overlap_lines=0)
    assert len(chunks) >= 2
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 20
    assert chunks[1].start_line == 21

    # 4. Session variables
    rlm.set_variable("target_symbol", "func_25")
    assert rlm.get_variable("target_symbol") == "func_25"

    # 5. Path traversal outside workspace root returns exists=False safely
    escaped = rlm.peek("../../etc/passwd", start_line=1, end_line=5)
    assert escaped.exists is False


def test_five_layer_sandbox_validator_and_system1_gatekeeper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 5-layer AST & resource sandbox validator blocks dangerous code in System-1."""
    validator = CodeSandboxValidator(max_output_bytes=1024)

    # Clean Python code passes all 5 layers
    clean_code = "def compute_sum(items: list[int]) -> int:\n    return sum(items)\n"
    clean_report = validator.validate_artifact(clean_code)
    assert clean_report.safe is True
    assert clean_report.safety_score == 1.0
    assert len(clean_report.violations) == 0

    # Malicious code with forbidden import (`pickle`), dangerous call (`eval`), and SSRF metadata IP
    unsafe_code = (
        "import pickle\n"
        "def exploit(payload: bytes) -> None:\n"
        "    url = 'http://169.254.169.254/latest/meta-data/'\n"
        "    eval('os.system(\"rm -rf /\")')\n"
    )
    unsafe_report = validator.validate_artifact(unsafe_code)
    assert unsafe_report.safe is False
    assert unsafe_report.safety_score < 0.5
    joined_violations = " ".join(unsafe_report.violations)
    assert "Layer 1 (Static AST)" in joined_violations
    assert "Layer 2 (Import Blocklist)" in joined_violations
    assert "Layer 4 (SSRF Guard)" in joined_violations

    # SafeEnv strips sensitive credentials
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-12345")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    sanitized = SafeEnv.build_sanitized_env()
    assert "OPENAI_API_KEY" not in sanitized
    assert "GITHUB_TOKEN" not in sanitized
    assert sanitized.get("PYTHONDONTWRITEBYTECODE") == "1"

    # System-1 Gatekeeper penalizes unsafe code in score() and choice()
    decision_engine = LocalHeuristicDecisionEngine()
    score_unsafe = decision_engine.score(
        unsafe_code,
        "Write a safe data parser",
        scale_rubric="production-grade python",
    )
    assert score_unsafe.score < 0.75
    assert score_unsafe.rubric_breakdown["sandbox_safety"] < 0.5


def test_reasoning_journal_recorded_during_fsm_run(tmp_path: Path) -> None:
    """Verify Thought Journal records structured checkpoints across all 4 LangGraph FSM nodes."""
    cfg = BridgeConfig(db_path=tmp_path / "journal_fsm.db", repo_root=tmp_path)
    engine = LocalMemoryEngine(config=cfg)
    engine.initialize()

    runner = HybridAgentBridgeRunner(config=cfg, engine=engine)
    state = runner.run(
        task="Implement a pure Python function that validates positive integers.",
        thread_id="thread-journal-test",
    )
    assert state["status"] == "consolidated"

    journal = ReasoningJournalStore(engine)
    entries = journal.get_thread_journal(thread_id="thread-journal-test")
    assert len(entries) >= 4
    nodes = [e.node_name for e in entries]
    assert "Context_Loader" in nodes
    assert "SubWorker_Node" in nodes
    assert "System1_Gatekeeper" in nodes
    assert "Episodic_Consolidator" in nodes
