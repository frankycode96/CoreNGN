"""Unified Persistence Layer (`LocalMemoryEngine`).

Implements the Zero-Container Local-First persistence engine on a single SQLite
file in WAL mode with:
1. Working Memory: LangGraph SqliteSaver atomic checkpoints + `reasoning_journal` (Thought Journal).
2. Declarative Memory: `declarative_facts` scoped by `(repo_identifier, git_branch, category, key)`
   with typed taxonomy (`FACT`, `BELIEF` 30-day TTL, `PREFERENCE`, `IDENTITY`, `RULE`, `EXPERIENCE`),
   deterministic UPSERT, immutability pinning (`pinned`), and `memory_audit_log`.
3. Semantic & Episodic Memory: `episodic_artifacts` scoped by `(repo_identifier, git_branch)`
   and indexed via SQLite FTS5 (BM25) and CPU-embedded vector columns (`sqlite-vec` + SQLite float32 cosine UDF).
"""

from __future__ import annotations

import json
import logging
import platform
import sqlite3
import struct
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from agent_bridge.config import BridgeConfig
from agent_bridge.memory.git_context import GitContextInfo, resolve_git_context

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

CORE_DDL_V3 = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Layer 2: Declarative Memory (Git-Branch Scoped, Typed Taxonomy, TTL, Deterministic Upsert, Pinning)
CREATE TABLE IF NOT EXISTS declarative_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_identifier TEXT NOT NULL DEFAULT 'workspace:default',
    git_branch TEXT NOT NULL DEFAULT 'default',
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'FACT',
    pinned INTEGER NOT NULL DEFAULT 0,
    access_count INTEGER NOT NULL DEFAULT 0,
    expires_at TEXT,
    updated_at TEXT NOT NULL,
    CONSTRAINT uq_declarative_repo_branch_cat_key UNIQUE (repo_identifier, git_branch, category, key)
);

CREATE INDEX IF NOT EXISTS idx_declarative_repo_branch_cat_key
    ON declarative_facts (repo_identifier, git_branch, category, key);

CREATE INDEX IF NOT EXISTS idx_declarative_category_updated
    ON declarative_facts (category, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_declarative_expires_at
    ON declarative_facts (expires_at);

-- Layer 3 & 4: Semantic & Episodic Memory (Git-Branch Scoped, Validated Task/Artifact pairs only)
CREATE TABLE IF NOT EXISTS episodic_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_identifier TEXT NOT NULL DEFAULT 'workspace:default',
    git_branch TEXT NOT NULL DEFAULT 'default',
    task TEXT NOT NULL,
    final_artifact TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    quality_score REAL NOT NULL DEFAULT 1.0,
    pinned INTEGER NOT NULL DEFAULT 0,
    embedding BLOB,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodic_repo_branch_created
    ON episodic_artifacts (repo_identifier, git_branch, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_episodic_created_at
    ON episodic_artifacts (created_at DESC);

-- SQLite FTS5 Virtual Table for BM25 Lexical Matching
CREATE VIRTUAL TABLE IF NOT EXISTS episodic_artifacts_fts USING fts5(
    task,
    final_artifact,
    tags_json,
    content='episodic_artifacts',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Synchronization Triggers for FTS5 External Content Table
CREATE TRIGGER IF NOT EXISTS trg_episodic_ai AFTER INSERT ON episodic_artifacts BEGIN
    INSERT INTO episodic_artifacts_fts(rowid, task, final_artifact, tags_json)
    VALUES (new.id, new.task, new.final_artifact, new.tags_json);
END;

CREATE TRIGGER IF NOT EXISTS trg_episodic_ad AFTER DELETE ON episodic_artifacts BEGIN
    INSERT INTO episodic_artifacts_fts(episodic_artifacts_fts, rowid, task, final_artifact, tags_json)
    VALUES ('delete', old.id, old.task, old.final_artifact, old.tags_json);
END;

CREATE TRIGGER IF NOT EXISTS trg_episodic_au AFTER UPDATE ON episodic_artifacts BEGIN
    INSERT INTO episodic_artifacts_fts(episodic_artifacts_fts, rowid, task, final_artifact, tags_json)
    VALUES ('delete', old.id, old.task, old.final_artifact, old.tags_json);
    INSERT INTO episodic_artifacts_fts(rowid, task, final_artifact, tags_json)
    VALUES (new.id, new.task, new.final_artifact, new.tags_json);
END;

-- Memory Audit Log (Deterministic record of merges, pins, forgets, and conflict overrides)
CREATE TABLE IF NOT EXISTS memory_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_identifier TEXT NOT NULL DEFAULT 'workspace:default',
    git_branch TEXT NOT NULL DEFAULT 'default',
    action TEXT NOT NULL,
    target_tier TEXT NOT NULL,
    target_ref TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_audit_created
    ON memory_audit_log (created_at DESC);

-- Thought Journal (`reasoning_book` - structured reasoning steps decoupled from episodic artifacts)
CREATE TABLE IF NOT EXISTS reasoning_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_identifier TEXT NOT NULL DEFAULT 'workspace:default',
    git_branch TEXT NOT NULL DEFAULT 'default',
    thread_id TEXT NOT NULL,
    task TEXT NOT NULL,
    step_index INTEGER NOT NULL DEFAULT 1,
    node_name TEXT NOT NULL,
    thought_summary TEXT NOT NULL,
    noul_confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reasoning_journal_thread
    ON reasoning_journal (repo_identifier, git_branch, thread_id, step_index ASC);
"""


def serialize_f32_vector(vector: list[float] | np.ndarray) -> bytes:
    """Pack a 1-D float vector into a compact little-endian float32 BLOB (sqlite-vec compatible)."""
    arr = np.asarray(vector, dtype=np.float32).ravel()
    return struct.pack(f"<{len(arr)}f", *arr.tolist())


def deserialize_f32_vector(blob: bytes | None) -> np.ndarray | None:
    """Unpack a little-endian float32 BLOB into a 1-D NumPy array."""
    if not blob:
        return None
    count = len(blob) // 4
    if count == 0:
        return None
    return np.frombuffer(blob, dtype="<f4", count=count)


def _sqlite_cosine_similarity(blob_a: bytes | None, blob_b: bytes | None) -> float:
    """Deterministic SQLite scalar UDF computing cosine similarity between two float32 BLOBs."""
    if not blob_a or not blob_b or len(blob_a) != len(blob_b):
        return 0.0
    vec_a = np.frombuffer(blob_a, dtype="<f4")
    vec_b = np.frombuffer(blob_b, dtype="<f4")
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, sim))


def detect_sqlite_vec_platform_guidance() -> str:
    """Return cross-platform remediation guidance if `sqlite-vec` C extension fails to load."""
    os_name = sys.platform
    arch = platform.machine().lower()
    if os_name == "darwin":
        return (
            f"macOS ({arch}) detected: Ensure Python is built with `--enable-loadable-sqlite-extensions` "
            "(e.g., Homebrew `brew install python@3.11`) and `pip install sqlite-vec` provides `vec0.dylib`."
        )
    if os_name.startswith("linux"):
        return (
            f"Linux ({arch}) detected: Ensure glibc >= 2.28 and `libsqlite3-0` with loadable extension support "
            "are installed alongside `pip install sqlite-vec` (`vec0.so`)."
        )
    if os_name in {"win32", "cygwin"}:
        return (
            f"Windows ({arch}) detected: Ensure official python.org CPython >= 3.10 with `sqlite3.dll` "
            "extension loading enabled and `pip install sqlite-vec` (`vec0.dll`)."
        )
    return f"Platform {os_name}/{arch}: Verify `sqlite-vec` C shared library compatibility."


class LocalMemoryEngine:
    """Unified embedded SQLite engine managing Working, Declarative, Episodic, and Journal layers."""

    def __init__(
        self,
        config: BridgeConfig | None = None,
        db_path: Path | str | None = None,
        git_branch: str | None = None,
        repo_identifier: str | None = None,
    ) -> None:
        self.config = config or BridgeConfig.from_env()
        if db_path is not None:
            self.config.db_path = Path(db_path).expanduser().resolve()
        if git_branch is not None:
            self.config.git_branch_override = git_branch
        if repo_identifier is not None:
            self.config.repo_id_override = repo_identifier

        self.db_path: Path = self.config.db_path
        self.sqlite_vec_enabled: bool = False
        self.sqlite_vec_diagnostic: str = "uninitialized"
        self.git_context: GitContextInfo = resolve_git_context(
            workspace_path=self.config.repo_root,
            branch_override=self.config.git_branch_override,
            repo_id_override=self.config.repo_id_override,
        )
        self._ensure_directory()

    def refresh_git_context(
        self,
        branch_override: str | None = None,
        repo_id_override: str | None = None,
    ) -> GitContextInfo:
        """Re-evaluate the active Git context (e.g., after `git checkout` or branch switch)."""
        if branch_override is not None:
            self.config.git_branch_override = branch_override
        if repo_id_override is not None:
            self.config.repo_id_override = repo_id_override
        self.git_context = resolve_git_context(
            workspace_path=self.config.repo_root,
            branch_override=self.config.git_branch_override,
            repo_id_override=self.config.repo_id_override,
        )
        return self.git_context

    def _ensure_directory(self) -> None:
        """Ensure parent directory for the SQLite database exists."""
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply WAL pragmas, register vector UDFs, and load cross-platform sqlite-vec C extension."""
        conn.row_factory = sqlite3.Row
        sqlite_cfg = self.config.sqlite

        conn.execute(f"PRAGMA busy_timeout = {int(sqlite_cfg.busy_timeout_ms)};")
        conn.execute(f"PRAGMA journal_mode = {sqlite_cfg.journal_mode};")
        conn.execute(f"PRAGMA synchronous = {sqlite_cfg.synchronous};")
        conn.execute(f"PRAGMA cache_size = {int(sqlite_cfg.cache_size_kb)};")
        conn.execute(f"PRAGMA foreign_keys = {'ON' if sqlite_cfg.foreign_keys else 'OFF'};")

        conn.create_function("vec_cosine_similarity", 2, _sqlite_cosine_similarity, deterministic=True)

        if not hasattr(conn, "enable_load_extension"):
            self.sqlite_vec_enabled = False
            self.sqlite_vec_diagnostic = (
                "Python sqlite3 built without enable_load_extension; using CPU float32 UDF fallback. "
                + detect_sqlite_vec_platform_guidance()
            )
            return

        try:
            import sqlite_vec  # type: ignore[import-untyped]

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self.sqlite_vec_enabled = True
            self.sqlite_vec_diagnostic = f"loaded ({sys.platform}/{platform.machine().lower()})"
        except Exception as exc:  # pragma: no cover
            self.sqlite_vec_enabled = False
            self.sqlite_vec_diagnostic = (
                f"fallback_udf ({exc}). {detect_sqlite_vec_platform_guidance()}"
            )
            logger.debug("sqlite-vec extension not loaded: %s", self.sqlite_vec_diagnostic)

    def connect(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=self.config.sqlite.busy_timeout_ms / 1000.0)
        self._configure_connection(conn)
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager yielding a transactional WAL-configured SQLite connection."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _migrate_to_v3_if_needed(self, conn: sqlite3.Connection) -> None:
        """Idempotently migrate v1/v2 schemas to v3 (`memory_type`, `expires_at`, `access_count`)."""
        decl_info = conn.execute("PRAGMA table_info(declarative_facts);").fetchall()
        if decl_info:
            col_names = {str(r["name"]) for r in decl_info}
            if "git_branch" not in col_names or "repo_identifier" not in col_names or "pinned" not in col_names:
                default_repo = self.git_context.repo_identifier
                default_branch = self.git_context.git_branch
                conn.executescript(
                    f"""
                    ALTER TABLE declarative_facts RENAME TO _declarative_facts_v1;
                    CREATE TABLE declarative_facts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        repo_identifier TEXT NOT NULL DEFAULT 'workspace:default',
                        git_branch TEXT NOT NULL DEFAULT 'default',
                        category TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value_json TEXT NOT NULL,
                        memory_type TEXT NOT NULL DEFAULT 'FACT',
                        pinned INTEGER NOT NULL DEFAULT 0,
                        access_count INTEGER NOT NULL DEFAULT 0,
                        expires_at TEXT,
                        updated_at TEXT NOT NULL,
                        CONSTRAINT uq_declarative_repo_branch_cat_key UNIQUE (repo_identifier, git_branch, category, key)
                    );
                    INSERT INTO declarative_facts (
                        id, repo_identifier, git_branch, category, key, value_json,
                        memory_type, pinned, access_count, expires_at, updated_at
                    )
                    SELECT id, {default_repo!r}, {default_branch!r}, category, key, value_json,
                           'FACT', 0, 0, NULL, updated_at
                    FROM _declarative_facts_v1;
                    DROP TABLE _declarative_facts_v1;
                    """
                )
            else:
                if "memory_type" not in col_names:
                    conn.execute(
                        "ALTER TABLE declarative_facts ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'FACT';"
                    )
                if "access_count" not in col_names:
                    conn.execute(
                        "ALTER TABLE declarative_facts ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;"
                    )
                if "expires_at" not in col_names:
                    conn.execute("ALTER TABLE declarative_facts ADD COLUMN expires_at TEXT;")

        ep_info = conn.execute("PRAGMA table_info(episodic_artifacts);").fetchall()
        if ep_info:
            ep_cols = {str(r["name"]) for r in ep_info}
            if "repo_identifier" not in ep_cols:
                conn.execute(
                    f"ALTER TABLE episodic_artifacts ADD COLUMN repo_identifier TEXT NOT NULL DEFAULT {self.git_context.repo_identifier!r};"
                )
            if "git_branch" not in ep_cols:
                conn.execute(
                    f"ALTER TABLE episodic_artifacts ADD COLUMN git_branch TEXT NOT NULL DEFAULT {self.git_context.git_branch!r};"
                )
            if "pinned" not in ep_cols:
                conn.execute("ALTER TABLE episodic_artifacts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;")

    def initialize(self) -> dict[str, object]:
        """Initialize or migrate database schemas, FTS5 virtual tables, and sqlite-vec tables."""
        with self.connection() as conn:
            self._migrate_to_v3_if_needed(conn)
            conn.executescript(CORE_DDL_V3)

            if self.sqlite_vec_enabled:
                dim = int(self.config.sqlite.vector_dim)
                try:
                    conn.execute(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodic USING vec0("
                        f"artifact_id INTEGER PRIMARY KEY, embedding float[{dim}]);"
                    )
                except sqlite3.Error as exc:
                    logger.warning("Could not create vec0 virtual table: %s", exc)
                    self.sqlite_vec_enabled = False

            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?);",
                (SCHEMA_VERSION, now_iso),
            )

            journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]

        return {
            "db_path": str(self.db_path),
            "schema_version": SCHEMA_VERSION,
            "journal_mode": str(journal_mode).upper(),
            "sqlite_vec_enabled": self.sqlite_vec_enabled,
            "sqlite_vec_diagnostic": self.sqlite_vec_diagnostic,
            "fts5_enabled": True,
            "repo_identifier": self.git_context.repo_identifier,
            "git_branch": self.git_context.git_branch,
        }

    def record_audit_event(
        self,
        action: str,
        target_tier: str,
        target_ref: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Persist a deterministic entry in `memory_audit_log`."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO memory_audit_log (
                    repo_identifier, git_branch, action, target_tier, target_ref, details_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    self.git_context.repo_identifier,
                    self.git_context.git_branch,
                    action,
                    target_tier,
                    target_ref,
                    json.dumps(details or {}, ensure_ascii=False),
                    now_iso,
                ),
            )
            return int(cur.lastrowid or 0)

    def list_audit_events(self, limit: int = 25) -> list[dict[str, Any]]:
        """Retrieve recent entries from `memory_audit_log`."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, repo_identifier, git_branch, action, target_tier, target_ref, details_json, created_at
                FROM memory_audit_log
                ORDER BY id DESC
                LIMIT ?;
                """,
                (int(limit),),
            ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "repo_identifier": str(r["repo_identifier"]),
                "git_branch": str(r["git_branch"]),
                "action": str(r["action"]),
                "target_tier": str(r["target_tier"]),
                "target_ref": str(r["target_ref"]),
                "details": json.loads(r["details_json"] or "{}"),
                "created_at": str(r["created_at"]),
            }
            for r in rows
        ]

    @contextmanager
    def langgraph_checkpointer(self) -> Iterator[object]:
        """Yield a Working Memory checkpointer backed by the unified SQLite database."""
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-untyped]

            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._configure_connection(conn)
            try:
                saver = SqliteSaver(conn)
                saver.setup()
                yield saver
            finally:
                conn.close()
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import-untyped]

            logger.warning("langgraph-checkpoint-sqlite not installed; falling back to in-memory saver")
            yield MemorySaver()

    def list_working_checkpoints(self, limit: int = 25) -> list[dict[str, object]]:
        """List recent LangGraph Working Memory checkpoints stored in SQLite."""
        with self.connection() as conn:
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='checkpoints';"
            ).fetchone()
            if not table_exists:
                return []
            rows = conn.execute(
                """
                SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id
                FROM checkpoints
                ORDER BY rowid DESC
                LIMIT ?;
                """,
                (int(limit),),
            ).fetchall()
        return [
            {
                "thread_id": str(r["thread_id"]),
                "checkpoint_ns": str(r["checkpoint_ns"] or ""),
                "checkpoint_id": str(r["checkpoint_id"]),
                "parent_checkpoint_id": str(r["parent_checkpoint_id"] or ""),
            }
            for r in rows
        ]

    def inspect_stats(self) -> dict[str, object]:
        """Return operational telemetry for the unified SQLite memory engine."""
        with self.connection() as conn:
            decl_count = conn.execute("SELECT COUNT(*) FROM declarative_facts;").fetchone()[0]
            pinned_decl = conn.execute("SELECT COUNT(*) FROM declarative_facts WHERE pinned = 1;").fetchone()[0]
            ep_count = conn.execute("SELECT COUNT(*) FROM episodic_artifacts;").fetchone()[0]
            journal_count = conn.execute("SELECT COUNT(*) FROM reasoning_journal;").fetchone()[0]
            audit_count = conn.execute("SELECT COUNT(*) FROM memory_audit_log;").fetchone()[0]
            branch_rows = conn.execute(
                "SELECT DISTINCT git_branch FROM declarative_facts UNION SELECT DISTINCT git_branch FROM episodic_artifacts;"
            ).fetchall()
            journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]

        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "db_path": str(self.db_path),
            "size_bytes": size_bytes,
            "journal_mode": str(journal_mode).upper(),
            "declarative_facts_count": int(decl_count),
            "pinned_facts_count": int(pinned_decl),
            "episodic_artifacts_count": int(ep_count),
            "reasoning_journal_count": int(journal_count),
            "audit_events_count": int(audit_count),
            "sqlite_vec_enabled": self.sqlite_vec_enabled,
            "sqlite_vec_diagnostic": self.sqlite_vec_diagnostic,
            "repo_identifier": self.git_context.repo_identifier,
            "active_git_branch": self.git_context.git_branch,
            "stored_branches": sorted(str(r[0]) for r in branch_rows if r[0]),
        }
