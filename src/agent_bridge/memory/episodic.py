"""Semantic & Episodic Memory Layer (`episodic_artifacts` + SQLite FTS5 + CPU Vectors).

Stores only validated (Task, Final_Artifact) pairs scoped by `(repo_identifier, git_branch)`,
discarding intermediate tool-calls and error traces. Provides sub-millisecond hybrid retrieval
combining SQLite FTS5 BM25, CPU-embedded float32 vector cosine similarity, and recency decay.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from agent_bridge.memory.engine import (
    LocalMemoryEngine,
    deserialize_f32_vector,
    serialize_f32_vector,
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


class EpisodicArtifact(BaseModel):
    """Validated episodic memory record combining task specification, branch scope, and final artifact."""

    id: int
    repo_identifier: str = Field(default="workspace:default")
    git_branch: str = Field(default="default")
    task: str
    final_artifact: str
    tags: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    pinned: bool = Field(default=False)
    created_at: str
    hybrid_score: float = Field(default=0.0, description="Combined BM25 + Vector + Recency relevance score")
    bm25_score: float = Field(default=0.0)
    vector_score: float = Field(default=0.0)
    recency_decay: float = Field(default=1.0, description="Temporal decay multiplier in (0.0, 1.0]")


class CPUEmbedder:
    """Zero-dependency deterministic CPU vector embedder using signed feature projections.

    Produces L2-normalized float32 vectors on CPU in <0.2ms without GPU/VRAM or network overhead,
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


def compute_recency_decay(created_at_iso: str, half_life_days: float = 30.0) -> float:
    """Compute smooth exponential recency multiplier in [0.5, 1.0] based on age in days."""
    try:
        dt = datetime.fromisoformat(created_at_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_seconds = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
        age_days = age_seconds / 86400.0
        decay = 0.5 + 0.5 * float(np.exp(-age_days / max(1.0, half_life_days)))
        return round(max(0.5, min(1.0, decay)), 4)
    except Exception:
        return 1.0


class EpisodicMemoryStore:
    """Manages Git-scoped validated episodic artifacts with FTS5 BM25 + CPU vector hybrid search."""

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
        git_branch: str | None = None,
        repo_identifier: str | None = None,
        pinned: bool = False,
        created_at: str | None = None,
    ) -> EpisodicArtifact:
        """Persist a validated (Task, Final_Artifact) pair indexed by FTS5, CPU vector, and Git branch."""
        self._validate_artifact_payload(task, final_artifact)

        clean_task = task.strip()
        clean_artifact = final_artifact.strip()
        clean_tags = [t.strip().lower() for t in (tags or []) if t.strip()]
        tags_json = json.dumps(clean_tags, ensure_ascii=False)
        clamped_score = max(0.0, min(1.0, float(quality_score)))
        target_repo = (repo_identifier or self.engine.git_context.repo_identifier).strip()
        target_branch = (git_branch or self.engine.git_context.git_branch).strip()

        combined_text = f"{clean_task}\n{clean_artifact}\n{' '.join(clean_tags)}"
        vec = self.embedder.embed(combined_text)
        vec_blob = serialize_f32_vector(vec)
        now_iso = created_at or datetime.now(timezone.utc).isoformat()

        with self.engine.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO episodic_artifacts (
                    repo_identifier, git_branch, task, final_artifact, tags_json,
                    quality_score, pinned, embedding, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    target_repo,
                    target_branch,
                    clean_task,
                    clean_artifact,
                    tags_json,
                    clamped_score,
                    1 if pinned else 0,
                    vec_blob,
                    now_iso,
                ),
            )
            row_id = int(cursor.lastrowid or 0)

            if self.engine.sqlite_vec_enabled:
                conn.execute(
                    "INSERT OR REPLACE INTO vec_episodic (artifact_id, embedding) VALUES (?, ?);",
                    (row_id, vec_blob),
                )

        return EpisodicArtifact(
            id=row_id,
            repo_identifier=target_repo,
            git_branch=target_branch,
            task=clean_task,
            final_artifact=clean_artifact,
            tags=clean_tags,
            quality_score=clamped_score,
            pinned=pinned,
            created_at=now_iso,
            hybrid_score=1.0,
            bm25_score=1.0,
            vector_score=1.0,
            recency_decay=1.0,
        )

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 4,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
        git_branch: str | None = None,
        all_branches: bool = False,
    ) -> list[EpisodicArtifact]:
        """Hybrid retrieval fusing SQLite FTS5 BM25, CPU vector cosine similarity, and Git branch scoping."""
        w_bm25 = self.engine.config.budget.bm25_weight if bm25_weight is None else bm25_weight
        w_vec = self.engine.config.budget.vector_weight if vector_weight is None else vector_weight
        target_repo = self.engine.git_context.repo_identifier
        active_branch = (git_branch or self.engine.git_context.git_branch).strip()

        if all_branches:
            visible_branches: list[str] | None = None
        elif git_branch is not None:
            visible_branches = [git_branch.strip()]
        else:
            visible_branches = self.engine.git_context.visible_branches()

        query_vec = self.embedder.embed(query)
        query_blob = serialize_f32_vector(query_vec)
        fts_query = self._sanitize_fts5_query(query)

        bm25_by_id: dict[int, float] = {}
        with self.engine.connection() as conn:
            if fts_query:
                fts_rows = conn.execute(
                    """
                    SELECT rowid AS id, (-1.0 * bm25(episodic_artifacts_fts, 2.0, 1.0, 0.5)) AS raw_bm25
                    FROM episodic_artifacts_fts
                    WHERE episodic_artifacts_fts MATCH ?
                    ORDER BY bm25(episodic_artifacts_fts)
                    LIMIT ?;
                    """,
                    (fts_query, max(top_k * 8, 40)),
                ).fetchall()
                for r in fts_rows:
                    bm25_by_id[int(r["id"])] = max(0.0, float(r["raw_bm25"]))

            if visible_branches is None:
                candidate_rows = conn.execute(
                    """
                    SELECT
                        id, repo_identifier, git_branch, task, final_artifact,
                        tags_json, quality_score, pinned, created_at,
                        vec_cosine_similarity(embedding, ?) AS vec_sim
                    FROM episodic_artifacts
                    WHERE repo_identifier = ?
                    ORDER BY vec_sim DESC, id DESC
                    LIMIT ?;
                    """,
                    (query_blob, target_repo, max(top_k * 10, 50)),
                ).fetchall()
            else:
                placeholders = ", ".join("?" for _ in visible_branches)
                params: list[Any] = [query_blob, target_repo, *visible_branches, max(top_k * 10, 50)]
                candidate_rows = conn.execute(
                    f"""
                    SELECT
                        id, repo_identifier, git_branch, task, final_artifact,
                        tags_json, quality_score, pinned, created_at,
                        vec_cosine_similarity(embedding, ?) AS vec_sim
                    FROM episodic_artifacts
                    WHERE repo_identifier = ? AND git_branch IN ({placeholders})
                    ORDER BY vec_sim DESC, id DESC
                    LIMIT ?;
                    """,
                    params,
                ).fetchall()

        if not candidate_rows:
            return []

        # Compute max_bm25 only over visible candidate rows so scores normalize cleanly
        visible_ids = {int(r["id"]) for r in candidate_rows}
        max_bm25 = max((v for k, v in bm25_by_id.items() if k in visible_ids), default=0.0)
        results: list[EpisodicArtifact] = []

        for r in candidate_rows:
            row_id = int(r["id"])
            raw_bm25 = bm25_by_id.get(row_id, 0.0)
            norm_bm25 = (raw_bm25 / max_bm25) if max_bm25 > 1e-9 else 0.0
            vec_sim = max(0.0, float(r["vec_sim"] or 0.0))
            quality = float(r["quality_score"] or 1.0)
            created_str = str(r["created_at"])
            recency = compute_recency_decay(created_str)
            row_branch = str(r["git_branch"])
            branch_boost = 1.0 if row_branch == active_branch else 0.92
            pin_boost = 1.05 if int(r["pinned"] or 0) == 1 else 1.0

            base_relevance = (w_bm25 * norm_bm25) + (w_vec * vec_sim)
            hybrid = base_relevance * (0.85 + 0.15 * quality) * recency * branch_boost * pin_boost
            hybrid = min(1.0, hybrid)

            results.append(
                EpisodicArtifact(
                    id=row_id,
                    repo_identifier=str(r["repo_identifier"]),
                    git_branch=row_branch,
                    task=str(r["task"]),
                    final_artifact=str(r["final_artifact"]),
                    tags=json.loads(r["tags_json"] or "[]"),
                    quality_score=quality,
                    pinned=bool(int(r["pinned"] or 0)),
                    created_at=created_str,
                    hybrid_score=round(hybrid, 4),
                    bm25_score=round(norm_bm25, 4),
                    vector_score=round(vec_sim, 4),
                    recency_decay=recency,
                )
            )

        results.sort(key=lambda item: (item.hybrid_score, item.id), reverse=True)
        return results[:top_k]

    def list_recent(
        self,
        limit: int = 20,
        git_branch: str | None = None,
        all_branches: bool = False,
    ) -> list[EpisodicArtifact]:
        """Return the most recently consolidated episodic artifacts respecting branch scoping."""
        target_repo = self.engine.git_context.repo_identifier
        if all_branches:
            branches: list[str] | None = None
        elif git_branch is not None:
            branches = [git_branch.strip()]
        else:
            branches = self.engine.git_context.visible_branches()

        with self.engine.connection() as conn:
            if branches is None:
                rows = conn.execute(
                    """
                    SELECT id, repo_identifier, git_branch, task, final_artifact,
                           tags_json, quality_score, pinned, created_at
                    FROM episodic_artifacts
                    WHERE repo_identifier = ?
                    ORDER BY id DESC
                    LIMIT ?;
                    """,
                    (target_repo, int(limit)),
                ).fetchall()
            else:
                placeholders = ", ".join("?" for _ in branches)
                rows = conn.execute(
                    f"""
                    SELECT id, repo_identifier, git_branch, task, final_artifact,
                           tags_json, quality_score, pinned, created_at
                    FROM episodic_artifacts
                    WHERE repo_identifier = ? AND git_branch IN ({placeholders})
                    ORDER BY id DESC
                    LIMIT ?;
                    """,
                    [target_repo, *branches, int(limit)],
                ).fetchall()

        return [
            EpisodicArtifact(
                id=int(r["id"]),
                repo_identifier=str(r["repo_identifier"]),
                git_branch=str(r["git_branch"]),
                task=str(r["task"]),
                final_artifact=str(r["final_artifact"]),
                tags=json.loads(r["tags_json"] or "[]"),
                quality_score=float(r["quality_score"]),
                pinned=bool(int(r["pinned"] or 0)),
                created_at=str(r["created_at"]),
                recency_decay=compute_recency_decay(str(r["created_at"])),
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
