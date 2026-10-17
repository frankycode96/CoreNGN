"""Unified Persistence Layer (LocalMemoryEngine).

Implements the Zero-Container Local-First persistence engine on a single SQLite
file in WAL mode with:
1. Working Memory: LangGraph SqliteSaver atomic checkpoints
2. Declarative Memory: `declarative_facts` with deterministic UPSERT on (category, key)
3. Semantic & Episodic Memory: `episodic_artifacts` indexed via SQLite FTS5 (BM25)
   and CPU-embedded vector columns (`sqlite-vec` + SQLite float32 cosine UDF).
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from agent_bridge.config import BridgeConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

CORE_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Layer 2: Declarative Memory (Deterministic Upsert, Latest-is-Truth)
CREATE TABLE IF NOT EXISTS declarative_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CONSTRAINT uq_declarative_category_key UNIQUE (category, key)
);

CREATE INDEX IF NOT EXISTS idx_declarative_category_updated
    ON declarative_facts (category, updated_at DESC);

-- Layer 3 & 4: Semantic & Episodic Memory (Validated Task/Artifact pairs only)
CREATE TABLE IF NOT EXISTS episodic_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    final_artifact TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    quality_score REAL NOT NULL DEFAULT 1.0,
    embedding BLOB,
    created_at TEXT NOT NULL
);

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


class LocalMemoryEngine:
    """Unified embedded SQLite engine managing Working, Declarative, and Episodic memory layers."""

    def __init__(self, config: BridgeConfig | None = None, db_path: Path | str | None = None) -> None:
        self.config = config or BridgeConfig.from_env()
        if db_path is not None:
            self.config.db_path = Path(db_path).expanduser().resolve()
        self.db_path: Path = self.config.db_path
        self.sqlite_vec_enabled: bool = False
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure parent directory for the SQLite database exists."""
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply WAL pragmas, register vector UDFs, and optionally load sqlite-vec."""
        conn.row_factory = sqlite3.Row
        sqlite_cfg = self.config.sqlite

        conn.execute(f"PRAGMA busy_timeout = {int(sqlite_cfg.busy_timeout_ms)};")
        conn.execute(f"PRAGMA journal_mode = {sqlite_cfg.journal_mode};")
        conn.execute(f"PRAGMA synchronous = {sqlite_cfg.synchronous};")
        conn.execute(f"PRAGMA cache_size = {int(sqlite_cfg.cache_size_kb)};")
        conn.execute(f"PRAGMA foreign_keys = {'ON' if sqlite_cfg.foreign_keys else 'OFF'};")

        # Register in-process CPU vector similarity function for guaranteed portable SQL vector queries
        conn.create_function("vec_cosine_similarity", 2, _sqlite_cosine_similarity, deterministic=True)

        # Attempt to load native C sqlite-vec extension if supported by the Python build
        if hasattr(conn, "enable_load_extension"):
            try:
                import sqlite_vec  # type: ignore[import-untyped]

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self.sqlite_vec_enabled = True
            except Exception as exc:  # pragma: no cover - graceful fallback when load_extension disabled
                logger.debug("sqlite-vec extension not loaded (using native float32 UDF fallback): %s", exc)

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

    def initialize(self) -> dict[str, object]:
        """Initialize database schemas, FTS5 virtual tables, and sqlite-vec tables if available."""
        with self.connection() as conn:
            conn.executescript(CORE_DDL)

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
            "fts5_enabled": True,
        }

    @contextmanager
    def langgraph_checkpointer(self) -> Iterator[object]:
        """Yield a Working Memory checkpointer backed by the unified SQLite database.

        Uses LangGraph's official `SqliteSaver` when installed, or falls back to
        `MemorySaver` only if `langgraph-checkpoint-sqlite` is unavailable.
        """
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

    def inspect_stats(self) -> dict[str, object]:
        """Return operational telemetry for the unified SQLite memory engine."""
        with self.connection() as conn:
            decl_count = conn.execute("SELECT COUNT(*) FROM declarative_facts;").fetchone()[0]
            ep_count = conn.execute("SELECT COUNT(*) FROM episodic_artifacts;").fetchone()[0]
            journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]

        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "db_path": str(self.db_path),
            "size_bytes": size_bytes,
            "journal_mode": str(journal_mode).upper(),
            "declarative_facts_count": int(decl_count),
            "episodic_artifacts_count": int(ep_count),
            "sqlite_vec_enabled": self.sqlite_vec_enabled,
        }
