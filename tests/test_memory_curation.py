"""Phase 6 Unit Tests: Memory Curation (`pin`, `unpin`, `forget`, `prune`, `explain`) & Zero-Bloat TUI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_bridge.cli import main as cli_main
from agent_bridge.config import BridgeConfig
from agent_bridge.memory.curation import MemoryCurator
from agent_bridge.memory.declarative import DeclarativeMemoryStore, PinnedMemoryError
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore
from agent_bridge.ui.tui import (
    AgentBridgeTUI,
    compute_repo_truth_diffs,
    render_rich_dashboard_snapshot,
)


def test_curation_pin_unpin_and_immutability_guard(tmp_path: Path) -> None:
    engine = LocalMemoryEngine(db_path=tmp_path / "pin_test.db", git_branch="main")
    engine.initialize()
    decl = DeclarativeMemoryStore(engine)
    curator = MemoryCurator(engine, decl)

    decl.upsert_fact("security", "sandbox_timeout_sec", 120)
    pinned = curator.pin("security", "sandbox_timeout_sec")
    assert pinned.pinned is True

    # Attempting to overwrite a pinned fact must raise PinnedMemoryError
    with pytest.raises(PinnedMemoryError):
        decl.upsert_fact("security", "sandbox_timeout_sec", 9999)

    # Attempting to forget a pinned fact without force=True must raise PinnedMemoryError
    with pytest.raises(PinnedMemoryError):
        curator.forget("security:sandbox_timeout_sec")

    # Unpinning restores mutable behavior
    unpinned = curator.unpin("security", "sandbox_timeout_sec")
    assert unpinned.pinned is False
    updated = decl.upsert_fact("security", "sandbox_timeout_sec", 90)
    assert updated.value == 90


def test_curation_forget_across_declarative_and_episodic_fts5(tmp_path: Path) -> None:
    engine = LocalMemoryEngine(db_path=tmp_path / "forget_test.db", git_branch="main")
    engine.initialize()
    decl = DeclarativeMemoryStore(engine)
    episodic = EpisodicMemoryStore(engine)
    curator = MemoryCurator(engine, decl, episodic)

    decl.upsert_fact("config", "temp_setting", "obsolete")
    ep = episodic.record_episode(
        task="Implement ephemeral_cache_cleaner utility",
        final_artifact="def ephemeral_cache_cleaner():\n    return True\n",
        tags=["cache"],
    )

    # Verify FTS5 finds the episode before forget
    assert len(episodic.retrieve_hybrid("ephemeral_cache_cleaner", top_k=1)) == 1

    res_ep = curator.forget(f"ep:{ep.id}")
    assert res_ep.deleted is True
    assert res_ep.tier == "episodic"
    assert len(episodic.retrieve_hybrid("ephemeral_cache_cleaner", top_k=1)) == 0

    res_decl = curator.forget("config:temp_setting")
    assert res_decl.deleted is True
    assert res_decl.tier == "declarative"
    assert decl.get_fact("config", "temp_setting") is None


def test_curation_prune_age_and_salience_respecting_pins(tmp_path: Path) -> None:
    engine = LocalMemoryEngine(db_path=tmp_path / "prune_test.db", git_branch="main")
    engine.initialize()
    decl = DeclarativeMemoryStore(engine)
    episodic = EpisodicMemoryStore(engine)
    curator = MemoryCurator(engine, decl, episodic)

    old_iso = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

    # Insert an old unpinned fact and an old PINNED fact
    f_unpinned = decl.upsert_fact("legacy", "old_unpinned", "remove_me")
    f_pinned = decl.upsert_fact("legacy", "old_pinned_rule", "keep_me", pinned=True)
    with engine.connection() as conn:
        conn.execute(
            "UPDATE declarative_facts SET updated_at = ? WHERE id IN (?, ?);",
            (old_iso, f_unpinned.id, f_pinned.id),
        )

    # Insert a low-score episode, an old episode, and a high-score fresh episode
    ep_low = episodic.record_episode(
        task="Low quality draft task",
        final_artifact="def draft():\n    return 0\n",
        quality_score=0.25,
    )
    ep_old = episodic.record_episode(
        task="Ancient task from 60 days ago",
        final_artifact="def ancient():\n    return 1\n",
        quality_score=0.90,
        created_at=old_iso,
    )
    ep_fresh = episodic.record_episode(
        task="Fresh high quality task",
        final_artifact="def fresh():\n    return 2\n",
        quality_score=0.95,
    )

    # Dry run first
    dry = curator.prune(older_than_days=30, min_salience_score=0.50, dry_run=True)
    assert dry.dry_run is True
    assert f_unpinned.id in dry.pruned_declarative_ids
    assert f_pinned.id not in dry.pruned_declarative_ids
    assert ep_low.id in dry.pruned_episodic_ids
    assert ep_old.id in dry.pruned_episodic_ids
    assert ep_fresh.id not in dry.pruned_episodic_ids
    assert dry.skipped_pinned_count == 1

    # Execute actual prune
    real = curator.prune(older_than_days=30, min_salience_score=0.50, dry_run=False)
    assert real.total_pruned == 3
    assert decl.get_fact("legacy", "old_pinned_rule") is not None
    assert decl.get_fact("legacy", "old_unpinned") is None
    remaining_eps = episodic.list_recent(limit=10)
    assert [e.id for e in remaining_eps] == [ep_fresh.id]


def test_curation_explain_retrieval_score_breakdown_and_tui_snapshot(tmp_path: Path) -> None:
    real_file = tmp_path / "module_a.py"
    real_file.write_text("def module_a_fn() -> int:\n    return 10\n", encoding="utf-8")

    cfg = BridgeConfig(db_path=tmp_path / "explain_tui.db", repo_root=tmp_path)
    engine = LocalMemoryEngine(config=cfg, git_branch="main")
    engine.initialize()
    decl = DeclarativeMemoryStore(engine)
    episodic = EpisodicMemoryStore(engine)
    curator = MemoryCurator(engine, decl, episodic)

    decl.upsert_fact("repo_file", "module_a.py", "def stale_version(): pass", pinned=True)
    episodic.record_episode(
        task="Optimize module_a.py integer calculation",
        final_artifact="def module_a_fn() -> int:\n    return 10\n",
        tags=["module_a", "optimization"],
        quality_score=0.96,
    )

    report = curator.explain("Optimize module_a.py integer calculation", top_k=5)
    assert report.max_token_budget < 800
    assert report.total_tokens_used < 800
    assert len(report.candidates) >= 3

    tiers = [c.tier for c in report.candidates]
    assert "repo_truth" in tiers
    assert "declarative" in tiers
    assert "episodic" in tiers

    ep_cand = next(c for c in report.candidates if c.tier == "episodic")
    assert ep_cand.bm25_raw_normalized > 0.0
    assert ep_cand.vector_cosine > 0.0
    assert ep_cand.final_hybrid_score > 0.5
    assert "BM25=" in ep_cand.justification and "VecCos=" in ep_cand.justification

    # Verify Repo Truth Diff detector and Rich TUI snapshot renderer
    diffs = compute_repo_truth_diffs(engine)
    assert len(diffs) >= 1
    assert diffs[0]["file"] == "module_a.py"
    assert diffs[0]["status"] == "DRIFT_MODIFIED_ON_DISK"

    snapshot_str = render_rich_dashboard_snapshot(engine, query="module_a.py")
    assert "Hybrid Agent Bridge" in snapshot_str
    assert "Tab 1A: Declarative Facts" in snapshot_str
    assert "Tab 3: Workspace & Repo Truth Diff" in snapshot_str

    # Verify Textual App instantiation and CLI Phase 6 commands
    tui_app = AgentBridgeTUI(engine=engine)
    assert tui_app.engine is engine

    db_str = str(cfg.db_path)
    assert cli_main(["--db-path", db_str, "pin", "repo_file", "module_a.py"]) == 0
    assert cli_main(["--db-path", db_str, "explain", "module_a.py"]) == 0
    assert cli_main(["--db-path", db_str, "prune", "--older-than", "90", "--dry-run"]) == 0
    assert cli_main(["--db-path", db_str, "tui", "--snapshot", "--query", "module_a.py"]) == 0
