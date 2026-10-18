"""Thought Journal Layer (`reasoning_journal` — inspired by NeuroCore `reasoning_book`).

Persists structured reasoning checkpoints and System-1 Noul confidence transitions
across macro-FSM execution steps without polluting `episodic_artifacts`.
Allows resuming or auditing multi-step reasoning trajectories per `thread_id`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from agent_bridge.memory.engine import LocalMemoryEngine


class JournalEntry(BaseModel):
    """Single reasoning checkpoint stored in `reasoning_journal`."""

    id: int
    repo_identifier: str
    git_branch: str
    thread_id: str
    task: str
    step_index: int
    node_name: str
    thought_summary: str
    noul_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: str


class ReasoningJournalStore:
    """Manages structured reasoning checkpoints in the unified SQLite WAL database."""

    def __init__(self, engine: LocalMemoryEngine) -> None:
        self.engine = engine

    def record_thought(
        self,
        thread_id: str,
        task: str,
        step_index: int,
        node_name: str,
        thought_summary: str,
        noul_confidence: float = 1.0,
        git_branch: str | None = None,
    ) -> JournalEntry:
        """Persist a single reasoning checkpoint for `thread_id`."""
        target_repo = self.engine.git_context.repo_identifier
        target_branch = (git_branch or self.engine.git_context.git_branch).strip()
        clamped_noul = max(0.0, min(1.0, float(noul_confidence)))
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.engine.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO reasoning_journal (
                    repo_identifier, git_branch, thread_id, task, step_index,
                    node_name, thought_summary, noul_confidence, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    target_repo,
                    target_branch,
                    thread_id.strip(),
                    task.strip(),
                    int(step_index),
                    node_name.strip(),
                    thought_summary.strip(),
                    clamped_noul,
                    now_iso,
                ),
            )
            row_id = int(cur.lastrowid or 0)

        return JournalEntry(
            id=row_id,
            repo_identifier=target_repo,
            git_branch=target_branch,
            thread_id=thread_id.strip(),
            task=task.strip(),
            step_index=int(step_index),
            node_name=node_name.strip(),
            thought_summary=thought_summary.strip(),
            noul_confidence=clamped_noul,
            created_at=now_iso,
        )

    def get_thread_journal(self, thread_id: str, limit: int = 50) -> list[JournalEntry]:
        """Retrieve ordered reasoning steps for a specific `thread_id`."""
        with self.engine.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, repo_identifier, git_branch, thread_id, task, step_index,
                       node_name, thought_summary, noul_confidence, created_at
                FROM reasoning_journal
                WHERE thread_id = ?
                ORDER BY step_index ASC, id ASC
                LIMIT ?;
                """,
                (thread_id.strip(), int(limit)),
            ).fetchall()

        return [
            JournalEntry(
                id=int(r["id"]),
                repo_identifier=str(r["repo_identifier"]),
                git_branch=str(r["git_branch"]),
                thread_id=str(r["thread_id"]),
                task=str(r["task"]),
                step_index=int(r["step_index"]),
                node_name=str(r["node_name"]),
                thought_summary=str(r["thought_summary"]),
                noul_confidence=float(r["noul_confidence"]),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]

    def list_recent_thoughts(self, limit: int = 25, git_branch: str | None = None) -> list[JournalEntry]:
        """List recent reasoning journal entries on the active repository."""
        target_repo = self.engine.git_context.repo_identifier
        branches = [git_branch.strip()] if git_branch else self.engine.git_context.visible_branches()
        placeholders = ", ".join("?" for _ in branches)

        with self.engine.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, repo_identifier, git_branch, thread_id, task, step_index,
                       node_name, thought_summary, noul_confidence, created_at
                FROM reasoning_journal
                WHERE repo_identifier = ? AND git_branch IN ({placeholders})
                ORDER BY id DESC
                LIMIT ?;
                """,
                [target_repo, *branches, int(limit)],
            ).fetchall()

        return [
            JournalEntry(
                id=int(r["id"]),
                repo_identifier=str(r["repo_identifier"]),
                git_branch=str(r["git_branch"]),
                thread_id=str(r["thread_id"]),
                task=str(r["task"]),
                step_index=int(r["step_index"]),
                node_name=str(r["node_name"]),
                thought_summary=str(r["thought_summary"]),
                noul_confidence=float(r["noul_confidence"]),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]
