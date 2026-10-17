"""Deterministic unit tests for Phase 1: Unified Persistence Layer (`LocalMemoryEngine`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_bridge.config import BridgeConfig
from agent_bridge.memory.declarative import DeclarativeMemoryStore
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore
from agent_bridge.memory.injector import ContextInjector, estimate_tokens


def test_sqlite_wal_and_schema_initialization(tmp_path: Path) -> None:
    db_file = tmp_path / "test_wal.db"
    cfg = BridgeConfig(db_path=db_file, repo_root=tmp_path)
    engine = LocalMemoryEngine(config=cfg)
    meta = engine.initialize()

    assert db_file.exists()
    assert meta["journal_mode"] == "WAL"
    assert meta["fts5_enabled"] is True
    assert meta["schema_version"] == 2

    stats = engine.inspect_stats()
    assert stats["declarative_facts_count"] == 0
    assert stats["episodic_artifacts_count"] == 0


def test_declarative_memory_deterministic_upsert(tmp_path: Path) -> None:
    engine = LocalMemoryEngine(db_path=tmp_path / "decl.db")
    engine.initialize()
    store = DeclarativeMemoryStore(engine)

    first = store.upsert_fact("Backend", "vector_engine", "pgvector")
    assert first.category == "backend"
    assert first.value == "pgvector"

    # Deterministic overwrite on (category, key) without LLM conflict resolution
    second = store.upsert_fact("backend", "vector_engine", "sqlite-vec-fts5")
    assert second.id == first.id
    assert second.value == "sqlite-vec-fts5"

    fetched = store.get_fact("backend", "vector_engine")
    assert fetched is not None
    assert fetched.value == "sqlite-vec-fts5"
    assert len(store.list_facts()) == 1


def test_episodic_memory_fts5_and_vector_hybrid_retrieval(tmp_path: Path) -> None:
    engine = LocalMemoryEngine(db_path=tmp_path / "episodic.db")
    engine.initialize()
    store = EpisodicMemoryStore(engine)

    ep1 = store.record_episode(
        task="Implement SQLite WAL mode connection pool with busy timeout",
        final_artifact="def configure_wal(conn):\n    conn.execute('PRAGMA journal_mode=WAL;')\n    return True",
        tags=["sqlite", "wal", "persistence"],
        quality_score=0.95,
    )
    store.record_episode(
        task="Configure FastAPI CORS middleware headers",
        final_artifact="def setup_cors(app):\n    return app",
        tags=["fastapi", "http"],
        quality_score=0.90,
    )

    # Reject error/traceback payloads from polluting episodic memory
    with pytest.raises(ValueError, match="Refusing to persist unvalidated error"):
        store.record_episode(
            task="Broken task",
            final_artifact="Traceback (most recent call last):\n  File 'x.py', line 1\nSyntaxError: invalid syntax",
        )

    results = store.retrieve_hybrid("SQLite WAL pragma busy_timeout persistence", top_k=2)
    assert len(results) >= 1
    assert results[0].id == ep1.id
    assert results[0].bm25_score > 0.0
    assert results[0].vector_score > 0.0
    assert results[0].hybrid_score > 0.5


def test_context_injector_hard_budget_and_repo_truth_hierarchy(tmp_path: Path) -> None:
    # Create a real file in repo_root and leave another file absent
    real_file = tmp_path / "existing_module.py"
    real_file.write_text("def real_repo_function() -> int:\n    return 42\n", encoding="utf-8")

    cfg = BridgeConfig(db_path=tmp_path / "injector.db", repo_root=tmp_path)
    engine = LocalMemoryEngine(config=cfg)
    engine.initialize()

    decl = DeclarativeMemoryStore(engine)
    episodic = EpisodicMemoryStore(engine)

    # Seed a stale fact claiming ghost_file.py exists, plus 30 large facts to test budget cap
    decl.upsert_fact("repo_file", "ghost_file.py", "Should be invalidated by Repo Truth")
    decl.upsert_fact("repo_file", "existing_module.py", "Stale description")
    for i in range(30):
        decl.upsert_fact("config", f"parameter_{i:02d}", f"value_payload_{'x' * 80}_{i}")

    for i in range(10):
        episodic.record_episode(
            task=f"Historical task number {i} for existing_module.py",
            final_artifact=f"def helper_{i}() -> int:\n    return {i} * 100\n",
            tags=["history"],
        )

    injector = ContextInjector(engine, decl, episodic)
    bundle = injector.build_context(
        task="Refactor existing_module.py and check ghost_file.py",
        max_tokens=800,
    )

    assert bundle.estimated_tokens < 800
    assert estimate_tokens(bundle.formatted_block) < 800
    assert "[REPO_TRUTH]" in bundle.formatted_block
    assert "existing_module.py: EXISTS" in bundle.formatted_block
    assert "ghost_file.py: MISSING_ON_DISK" in bundle.formatted_block
    # Stale declarative fact for ghost_file.py must be excluded from included_facts
    assert all(f.key != "ghost_file.py" for f in bundle.included_facts)
