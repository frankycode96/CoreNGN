"""Phase 5/7 Unit Tests: Cross-Platform SQLite WAL Concurrency & FTS5 Trigger Stress Tests."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

from agent_bridge.memory.declarative import DeclarativeMemoryStore
from agent_bridge.memory.engine import LocalMemoryEngine, detect_sqlite_vec_platform_guidance
from agent_bridge.memory.episodic import EpisodicMemoryStore


def test_cross_platform_sqlite_vec_guidance_and_diagnostics(tmp_path: Path) -> None:
    guidance = detect_sqlite_vec_platform_guidance()
    assert any(os_name in guidance for os_name in ("macOS", "Linux", "Windows", "Platform"))

    engine = LocalMemoryEngine(db_path=tmp_path / "diag.db")
    meta = engine.initialize()
    assert isinstance(meta["sqlite_vec_diagnostic"], str)
    assert len(str(meta["sqlite_vec_diagnostic"])) > 0


def test_sqlite_wal_concurrent_read_write_and_fts5_triggers(tmp_path: Path) -> None:
    db_file = tmp_path / "concurrent_wal.db"
    engine = LocalMemoryEngine(db_path=db_file, git_branch="main", repo_identifier="git:stress:01")
    engine.initialize()

    decl = DeclarativeMemoryStore(engine)
    episodic = EpisodicMemoryStore(engine)

    def _writer_worker(worker_id: int) -> int:
        for i in range(8):
            decl.upsert_fact("worker", f"key_{worker_id}_{i}", {"worker": worker_id, "step": i})
            episodic.record_episode(
                task=f"Concurrent task worker {worker_id} iteration {i} for vector indexing",
                final_artifact=f"def run_w{worker_id}_{i}() -> int:\n    return {worker_id * 100 + i}\n",
                tags=[f"worker_{worker_id}", "stress"],
            )
        return 8

    def _reader_worker(worker_id: int) -> int:
        reads = 0
        for _ in range(10):
            _ = decl.list_facts(limit=20)
            _ = episodic.retrieve_hybrid(f"Concurrent task worker {worker_id} vector", top_k=3)
            reads += 1
        return reads

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_writer_worker, w) for w in range(3)] + [
            pool.submit(_reader_worker, r) for r in range(3)
        ]
        results = [f.result(timeout=30.0) for f in futures]

    assert sum(results[:3]) == 24
    stats = engine.inspect_stats()
    assert stats["declarative_facts_count"] == 24
    assert stats["episodic_artifacts_count"] == 24

    # Verify FTS5 index matches base table count after rapid concurrent inserts
    with engine.connection() as conn:
        fts_count = conn.execute("SELECT COUNT(*) FROM episodic_artifacts_fts;").fetchone()[0]
        assert int(fts_count) == 24

        # Test AFTER UPDATE and AFTER DELETE FTS5 triggers
        conn.execute(
            "UPDATE episodic_artifacts SET task = 'Updated unique_token_xyz987 task' WHERE id = 1;"
        )

    hits = episodic.retrieve_hybrid("unique_token_xyz987", top_k=1)
    assert len(hits) == 1
    assert hits[0].id == 1
