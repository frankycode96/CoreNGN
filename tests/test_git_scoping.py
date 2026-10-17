"""Phase 5 Unit Tests: Git-Awareness, Branch-Scoped Memory Isolation, and V1->V2 Migration."""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

from agent_bridge.config import BridgeConfig
from agent_bridge.memory.declarative import DeclarativeMemoryStore
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore
from agent_bridge.memory.git_context import resolve_git_context


def test_git_context_outside_git_repository(tmp_path: Path) -> None:
    non_git_dir = tmp_path / "standalone_workspace"
    non_git_dir.mkdir(parents=True)

    ctx = resolve_git_context(non_git_dir)
    assert ctx.is_git_repo is False
    assert ctx.repo_identifier == "workspace:default"
    assert ctx.git_branch == "default"
    assert ctx.is_primary_branch is True


def test_real_git_repo_branch_detection_and_isolation(tmp_path: Path) -> None:
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir(parents=True)

    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@local.test"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.name", "Local Dev"], cwd=str(repo_dir), check=True)
    (repo_dir / "README.md").write_text("# Sample\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "init main"], cwd=str(repo_dir), check=True, capture_output=True)

    db_path = tmp_path / "git_scoped.db"
    cfg = BridgeConfig(db_path=db_path, repo_root=repo_dir)
    engine = LocalMemoryEngine(config=cfg)
    engine.initialize()

    assert engine.git_context.is_git_repo is True
    assert engine.git_context.git_branch == "main"
    assert engine.git_context.repo_identifier.startswith("git:sample_repo:")

    decl = DeclarativeMemoryStore(engine)
    episodic = EpisodicMemoryStore(engine)

    # 1. Seed baseline memories on `main`
    decl.upsert_fact("architecture", "auth_mode", "jwt-stateless")
    decl.upsert_fact("database", "pool_size", 10)
    episodic.record_episode(
        task="Implement JWT stateless token verifier for main branch",
        final_artifact="def verify_jwt(token: str) -> bool:\n    return len(token) > 10\n",
        tags=["auth", "main"],
    )

    # 2. Switch git branch to unmerged feature branch `feat/experimental-oauth`
    subprocess.run(
        ["git", "checkout", "-b", "feat/experimental-oauth"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
    )
    engine.refresh_git_context()
    assert engine.git_context.git_branch == "feat/experimental-oauth"

    # Override `auth_mode` on the feature branch and add a feature-only fact + episode
    decl.upsert_fact("architecture", "auth_mode", "oauth2-pkce-experimental")
    decl.upsert_fact("feature_flag", "enable_pkce_flow", True)
    episodic.record_episode(
        task="Implement experimental OAuth2 PKCE verifier for feature branch",
        final_artifact="def verify_pkce(code: str) -> bool:\n    return code.startswith('pkce_')\n",
        tags=["oauth", "experimental"],
    )

    # On `feat/experimental-oauth`: feature override wins, and `main` `pool_size` is inherited
    feat_auth = decl.get_fact("architecture", "auth_mode")
    assert feat_auth is not None and feat_auth.value == "oauth2-pkce-experimental"
    inherited_pool = decl.get_fact("database", "pool_size")
    assert inherited_pool is not None and inherited_pool.value == 10

    # 3. Switch back to `main` and verify zero contamination from unmerged feature branch
    subprocess.run(["git", "checkout", "main"], cwd=str(repo_dir), check=True, capture_output=True)
    engine.refresh_git_context()
    assert engine.git_context.git_branch == "main"

    main_auth = decl.get_fact("architecture", "auth_mode")
    assert main_auth is not None
    assert main_auth.value == "jwt-stateless"  # Untouched by feature branch!

    leaked_flag = decl.get_fact("feature_flag", "enable_pkce_flow")
    assert leaked_flag is None  # Feature-only fact is invisible on `main`!

    main_episodes = episodic.retrieve_hybrid("OAuth2 PKCE verifier token", top_k=5)
    assert all(ep.git_branch == "main" for ep in main_episodes)
    assert all("verify_pkce" not in ep.final_artifact for ep in main_episodes)


def test_backward_compatible_v1_to_v2_schema_migration(tmp_path: Path) -> None:
    db_file = tmp_path / "legacy_v1.db"
    conn = sqlite3.connect(str(db_file))
    conn.executescript(
        """
        CREATE TABLE declarative_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CONSTRAINT uq_declarative_category_key UNIQUE (category, key)
        );
        INSERT INTO declarative_facts (category, key, value_json, updated_at)
        VALUES ('legacy', 'version', '"1.0"', '2026-01-01T00:00:00Z');

        CREATE TABLE episodic_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            final_artifact TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            quality_score REAL NOT NULL DEFAULT 1.0,
            embedding BLOB,
            created_at TEXT NOT NULL
        );
        INSERT INTO episodic_artifacts (task, final_artifact, tags_json, quality_score, created_at)
        VALUES ('Legacy task', 'def legacy():\n    return 1\n', '["v1"]', 0.9, '2026-01-01T00:00:00Z');
        """
    )
    conn.commit()
    conn.close()

    engine = LocalMemoryEngine(db_path=db_file, git_branch="main", repo_identifier="git:migrated:001")
    meta = engine.initialize()
    assert meta["schema_version"] == 2

    decl = DeclarativeMemoryStore(engine)
    migrated_fact = decl.get_fact("legacy", "version")
    assert migrated_fact is not None
    assert migrated_fact.value == "1.0"
    assert migrated_fact.git_branch == "main"
    assert migrated_fact.pinned is False
