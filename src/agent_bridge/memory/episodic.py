"""Semantic & Episodic Memory Layer (`episodic_artifacts` + SQLite FTS5 + CPU Vectors).

Stores only validated (Task, Final_Artifact) pairs, discarding intermediate tool-calls
and error traces. Provides sub-millisecond hybrid retrieval combining SQLite FTS5 BM25
and CPU-embedded float32 vector cosine similarity.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
from pydantic import BaseModel, Field

from agent_bridge.memory.engine import (
    LocalMemoryEngine,
    deserialize_f32_vector,
    serialize_f32_vector,
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


class EpisodicArtifact(BaseModel):
    """Validated episodic memory record combining task specification and final artifact."""

    id: int
    task: str
    final_artifact: str
    tags: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: str
    hybrid_score: float = Field(default=0.0, description="Combined BM25 + Vector relevance score")
    bm25_score: float = Field(default=0.0)
    vector_score: float = Field(default=0.0)


class CPUEmbedder:
    """Zero-dependency deterministic CPU vector embedder using signed feature projections.

    Produces L2-normalized float32 vectors on CPU in <0.2ms without GPU/network overhead,
    capturing word unigrams, bigrams, and identifier character 3-grams.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """Compute L2-normalized float32 vector of shape (dim,)."""
        vec = np.zeros(self.dim, dtype=np.float32)
        cleaned = text.lower().strip()
        if not cleaned:
            return vec

        tokens = _TOKEN_RE.findall(cleaned)
        features: list[tuple[str, float]] = []

        # Word unigrams (weight 1.0) and bigrams (weight 1.25)
        for i, tok in enumerate(tokens):
            features.append((f"w:{tok}", 1.0))
            if i + 1 < len(tokens):
                features.append((f"b:{tok}_{tokens[i + 1]}", 1.25))
            # Character 3-grams for sub-word / identifier similarity (weight 0.4)
            if len(tok) >= 4:
                for j in range(len(tok) - 2):
                    features.append((f"c:{tok[j : j + 3]}", 0.4))

        for feat, weight in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            hash_val = int.from_bytes(digest, "little")
            idx = hash_val % self.dim
            sign = 1.0 if ((hash_val >> 32) & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        return vec


class EpisodicMemoryStore:
    """Manages validated episodic artifacts with FTS5 BM25 + CPU vector hybrid search."""

    _FORBIDDEN_ERROR_MARKERS = (
        "traceback (most recent call last):",
        "unhandled exception:",
        "syntaxerror:",
        "__subworker_timeout_error__",
    )

    def __init__(self, engine: LocalMemoryEngine, embedder: CPUEmbedder | None = None) -> None:
        self.engine = engine
        self.embedder = embedder or CPUEmbedder(dim=engine.config.sqlite.vector_dim)

    @classmethod
    def _validate_artifact_payload(cls, task: str, final_artifact: str) -> None:
        """Ensure only clean, validated (Task, Final_Artifact) pairs enter episodic storage."""
        if not task or not task.strip():
            raise ValueError("Task description cannot be empty when recording an episodic artifact.")
        if not final_artifact or not final_artifact.strip():
            raise ValueError("Final artifact cannot be empty when recording an episodic artifact.")

        lowered = final_artifact.strip().lower()
        for marker in cls._FORBIDDEN_ERROR_MARKERS:
            if marker in lowered:
                raise ValueError(
                    f"Refusing to persist unvalidated error/traceback payload into episodic memory (matched '{marker}')."
                )

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Build a safe FTS5 OR-query from natural language or code task strings."""
        tokens = _TOKEN_RE.findall(query.lower())
        if not tokens:
            return ""
        # Deduplicate while preserving order, cap at 16 terms
        seen: set[str] = set()
        unique_tokens: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(f'"{t}"')
            if len(unique_tokens) >= 16:
                break
        return " OR ".join(unique_tokens)

    def record_episode(
        self,
        task: str,
        final_artifact: str,
        tags: list[str] | None = None,
        quality_score: float = 1.0,
    ) -> EpisodicArtifact:
        """Persist a validated (Task, Final_Artifact) pair indexed by FTS5 and CPU vector."""
        self._validate_artifact_payload(task, final_artifact)

        clean_task = task.strip()
        clean_artifact = final_artifact.strip()
        clean_tags = [t.strip().lower() for t in (tags or []) if t.strip()]
        tags_json = json.dumps(clean_tags, ensure_ascii=False)
        clamped_score = max(0.0, min(1.0, float(quality_score)))

        combined_text = f"{clean_task}\n{clean_artifact}\n{' '.join(clean_tags)}"
        vec = self.embedder.embed(combined_text)
        vec_blob = serialize_f32_vector(vec)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.engine.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO episodic_artifacts (task, final_artifact, tags_json, quality_score, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (clean_task, clean_artifact, tags_json, clamped_score, vec_blob, now_iso),
            )
            row_id = int(cursor.lastrowid or 0)

            if self.engine.sqlite_vec_enabled:
                conn.execute(
                    "INSERT OR REPLACE INTO vec_episodic (artifact_id, embedding) VALUES (?, ?);",
                    (row_id, vec_blob),
                )

        return EpisodicArtifact(
            id=row_id,
            task=clean_task,
            final_artifact=clean_artifact,
            tags=clean_tags,
            quality_score=clamped_score,
            created_at=now_iso,
            hybrid_score=1.0,
            bm25_score=1.0,
            vector_score=1.0,
        )

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 4,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> list[EpisodicArtifact]:
        """Hybrid retrieval fusing SQLite FTS5 BM25 ranking with CPU vector similarity."""
        w_bm25 = self.engine.config.budget.bm25_weight if bm25_weight is None else bm25_weight
        w_vec = self.engine.config.budget.vector_weight if vector_weight is None else vector_weight

        query_vec = self.embedder.embed(query)
        query_blob = serialize_f32_vector(query_vec)
        fts_query = self._sanitize_fts5_query(query)

        bm25_by_id: dict[int, float] = {}
        with self.engine.connection() as conn:
            if fts_query:
                # SQLite FTS5 bm25() returns negative values where more negative = better match
                fts_rows = conn.execute(
                    """
                    SELECT rowid AS id, (-1.0 * bm25(episodic_artifacts_fts, 2.0, 1.0, 0.5)) AS raw_bm25
                    FROM episodic_artifacts_fts
                    WHERE episodic_artifacts_fts MATCH ?
                    ORDER BY bm25(episodic_artifacts_fts)
                    LIMIT ?;
                    """,
                    (fts_query, max(top_k * 5, 25)),
                ).fetchall()
                for r in fts_rows:
                    bm25_by_id[int(r["id"])] = max(0.0, float(r["raw_bm25"]))

            # Fetch candidate episodes with SQL-level vector cosine similarity
            candidate_rows = conn.execute(
                """
                SELECT
                    id,
                    task,
                    final_artifact,
                    tags_json,
                    quality_score,
                    created_at,
                    vec_cosine_similarity(embedding, ?) AS vec_sim
                FROM episodic_artifacts
                ORDER BY vec_sim DESC, id DESC
                LIMIT ?;
                """,
                (query_blob, max(top_k * 8, 40)),
            ).fetchall()

        if not candidate_rows:
            return []

        max_bm25 = max(bm25_by_id.values(), default=0.0)
        results: list[EpisodicArtifact] = []

        for r in candidate_rows:
            row_id = int(r["id"])
            raw_bm25 = bm25_by_id.get(row_id, 0.0)
            norm_bm25 = (raw_bm25 / max_bm25) if max_bm25 > 1e-9 else 0.0
            vec_sim = max(0.0, float(r["vec_sim"] or 0.0))
            quality = float(r["quality_score"] or 1.0)

            hybrid = ((w_bm25 * norm_bm25) + (w_vec * vec_sim)) * (0.85 + 0.15 * quality)
            results.append(
                EpisodicArtifact(
                    id=row_id,
                    task=str(r["task"]),
                    final_artifact=str(r["final_artifact"]),
                    tags=json.loads(r["tags_json"] or "[]"),
                    quality_score=quality,
                    created_at=str(r["created_at"]),
                    hybrid_score=round(hybrid, 4),
                    bm25_score=round(norm_bm25, 4),
                    vector_score=round(vec_sim, 4),
                )
            )

        results.sort(key=lambda item: (item.hybrid_score, item.id), reverse=True)
        return results[:top_k]

    def list_recent(self, limit: int = 20) -> list[EpisodicArtifact]:
        """Return the most recently consolidated episodic artifacts."""
        with self.engine.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, task, final_artifact, tags_json, quality_score, created_at
                FROM episodic_artifacts
                ORDER BY id DESC
                LIMIT ?;
                """,
                (int(limit),),
            ).fetchall()

        return [
            EpisodicArtifact(
                id=int(r["id"]),
                task=str(r["task"]),
                final_artifact=str(r["final_artifact"]),
                tags=json.loads(r["tags_json"] or "[]"),
                quality_score=float(r["quality_score"]),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]

    def get_embedding_vector(self, episode_id: int) -> np.ndarray | None:
        """Retrieve the deserialized float32 embedding vector for an episode."""
        with self.engine.connection() as conn:
            row = conn.execute(
                "SELECT embedding FROM episodic_artifacts WHERE id = ?;",
                (int(episode_id),),
            ).fetchone()
        if row is None:
            return None
        return deserialize_f32_vector(row["embedding"])
