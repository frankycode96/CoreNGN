"""Declarative Memory Layer (`declarative_facts`).

Implements deterministic Upsert semantics scoped by `(repo_identifier, git_branch, category, key)`
with immutability pinning (`pinned=1`) and branch isolation rules:
- Temporal conflicts are never delegated to an LLM: the latest write on the active branch is truth.
- Pinned facts (`pinned=1`) are protected against automatic overwrites.
- Unmerged feature-branch facts are strictly isolated from `main`/`master` queries.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agent_bridge.memory.engine import LocalMemoryEngine


class PinnedMemoryError(PermissionError):
    """Raised when attempting to overwrite or prune an immutable pinned memory entry."""


class DeclarativeFact(BaseModel):
    """Typed representation of a single row in `declarative_facts`."""

    id: int
    repo_identifier: str = Field(default="workspace:default")
    git_branch: str = Field(default="default")
    category: str
    key: str
    value: Any
    pinned: bool = Field(default=False, description="True if protected against automatic upserts/pruning")
    updated_at: str = Field(description="ISO-8601 UTC timestamp of latest deterministic upsert")


class DeclarativeMemoryStore:
    """Deterministic Git-scoped CRUD, pinning, and search operations for Declarative Memory."""

    def __init__(self, engine: LocalMemoryEngine) -> None:
        self.engine = engine

    @staticmethod
    def _row_to_fact(row: Any) -> DeclarativeFact:
        return DeclarativeFact(
            id=int(row["id"]),
            repo_identifier=str(row["repo_identifier"]),
            git_branch=str(row["git_branch"]),
            category=str(row["category"]),
            key=str(row["key"]),
            value=json.loads(row["value_json"]),
            pinned=bool(int(row["pinned"])),
            updated_at=str(row["updated_at"]),
        )

    def upsert_fact(
        self,
        category: str,
        key: str,
        value: Any,
        git_branch: str | None = None,
        repo_identifier: str | None = None,
        pinned: bool | None = None,
        respect_pin: bool = True,
    ) -> DeclarativeFact:
        """Atomically insert or overwrite a fact identified by (repo_identifier, git_branch, category, key).

        Raises:
            PinnedMemoryError: If `respect_pin=True` and the existing fact is marked `pinned=1`.
        """
        norm_category = category.strip().lower()
        norm_key = key.strip()
        if not norm_category or not norm_key:
            raise ValueError("Both 'category' and 'key' must be non-empty strings.")

        target_repo = (repo_identifier or self.engine.git_context.repo_identifier).strip()
        target_branch = (git_branch or self.engine.git_context.git_branch).strip()

        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.engine.connection() as conn:
            existing = conn.execute(
                """
                SELECT id, pinned, value_json
                FROM declarative_facts
                WHERE repo_identifier = ? AND git_branch = ? AND category = ? AND key = ?;
                """,
                (target_repo, target_branch, norm_category, norm_key),
            ).fetchone()

            if existing is not None and int(existing["pinned"]) == 1 and respect_pin and pinned is not False:
                raise PinnedMemoryError(
                    f"Declarative fact ({norm_category!r}, {norm_key!r}) on branch {target_branch!r} "
                    "is pinned (immutable) and cannot be overwritten without unpinning."
                )

            pin_int = (
                int(bool(pinned))
                if pinned is not None
                else (int(existing["pinned"]) if existing is not None else 0)
            )

            conn.execute(
                """
                INSERT INTO declarative_facts (repo_identifier, git_branch, category, key, value_json, pinned, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_identifier, git_branch, category, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    pinned = excluded.pinned,
                    updated_at = excluded.updated_at;
                """,
                (target_repo, target_branch, norm_category, norm_key, serialized, pin_int, now_iso),
            )
            row = conn.execute(
                """
                SELECT id, repo_identifier, git_branch, category, key, value_json, pinned, updated_at
                FROM declarative_facts
                WHERE repo_identifier = ? AND git_branch = ? AND category = ? AND key = ?;
                """,
                (target_repo, target_branch, norm_category, norm_key),
            ).fetchone()

        if row is None:  # pragma: no cover
            raise RuntimeError(f"Failed to retrieve upserted fact ({norm_category}, {norm_key})")

        return self._row_to_fact(row)

    def get_fact(
        self,
        category: str,
        key: str,
        git_branch: str | None = None,
        repo_identifier: str | None = None,
        include_trunk_fallback: bool = True,
    ) -> DeclarativeFact | None:
        """Retrieve a declarative fact scoped to the active branch (with optional trunk inheritance)."""
        norm_category = category.strip().lower()
        norm_key = key.strip()
        target_repo = (repo_identifier or self.engine.git_context.repo_identifier).strip()

        if git_branch is not None:
            branches = [git_branch.strip()]
        elif include_trunk_fallback:
            branches = self.engine.git_context.visible_branches()
        else:
            branches = [self.engine.git_context.git_branch]

        with self.engine.connection() as conn:
            for br in branches:
                row = conn.execute(
                    """
                    SELECT id, repo_identifier, git_branch, category, key, value_json, pinned, updated_at
                    FROM declarative_facts
                    WHERE repo_identifier = ? AND git_branch = ? AND category = ? AND key = ?;
                    """,
                    (target_repo, br, norm_category, norm_key),
                ).fetchone()
                if row is not None:
                    return self._row_to_fact(row)
        return None

    def list_facts(
        self,
        category: str | None = None,
        limit: int = 50,
        git_branch: str | None = None,
        all_branches: bool = False,
    ) -> list[DeclarativeFact]:
        """List declarative facts respecting Git branch isolation and feature-branch overrides."""
        target_repo = self.engine.git_context.repo_identifier

        if all_branches:
            branches: list[str] | None = None
        elif git_branch is not None:
            branches = [git_branch.strip()]
        else:
            branches = self.engine.git_context.visible_branches()

        with self.engine.connection() as conn:
            if branches is None:
                if category:
                    rows = conn.execute(
                        """
                        SELECT id, repo_identifier, git_branch, category, key, value_json, pinned, updated_at
                        FROM declarative_facts
                        WHERE repo_identifier = ? AND category = ?
                        ORDER BY pinned DESC, updated_at DESC, id DESC
                        LIMIT ?;
                        """,
                        (target_repo, category.strip().lower(), int(limit)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT id, repo_identifier, git_branch, category, key, value_json, pinned, updated_at
                        FROM declarative_facts
                        WHERE repo_identifier = ?
                        ORDER BY pinned DESC, updated_at DESC, id DESC
                        LIMIT ?;
                        """,
                        (target_repo, int(limit)),
                    ).fetchall()
                return [self._row_to_fact(r) for r in rows]

            placeholders = ", ".join("?" for _ in branches)
            params: list[Any] = [target_repo, *branches]
            cat_clause = ""
            if category:
                cat_clause = "AND category = ?"
                params.append(category.strip().lower())

            rows = conn.execute(
                f"""
                SELECT id, repo_identifier, git_branch, category, key, value_json, pinned, updated_at
                FROM declarative_facts
                WHERE repo_identifier = ? AND git_branch IN ({placeholders}) {cat_clause}
                ORDER BY updated_at DESC, id DESC;
                """,
                params,
            ).fetchall()

        # Deduplicate (category, key) so active branch overrides inherited trunk branch
        branch_priority = {b: idx for idx, b in enumerate(branches)}
        best_by_key: dict[tuple[str, str], tuple[int, DeclarativeFact]] = {}
        for r in rows:
            fact = self._row_to_fact(r)
            prio = branch_priority.get(fact.git_branch, 999)
            ck = (fact.category, fact.key)
            if ck not in best_by_key or prio < best_by_key[ck][0]:
                best_by_key[ck] = (prio, fact)

        deduped = [item[1] for item in best_by_key.values()]
        deduped.sort(key=lambda f: (f.pinned, f.updated_at, f.id), reverse=True)
        return deduped[: int(limit)]

    def search_facts_for_query(self, query: str, limit: int = 12) -> list[DeclarativeFact]:
        """Retrieve relevant branch-scoped declarative facts ranked by pin status, token overlap, and recency."""
        all_facts = self.list_facts(limit=200)
        if not query.strip():
            return all_facts[:limit]

        query_tokens = {
            tok.lower()
            for tok in query.replace("/", " ").replace("_", " ").replace("-", " ").split()
            if len(tok) >= 2
        }
        if not query_tokens:
            return all_facts[:limit]

        scored: list[tuple[float, int, DeclarativeFact]] = []
        for idx, fact in enumerate(all_facts):
            haystack = f"{fact.category} {fact.key} {json.dumps(fact.value)}".lower()
            matches = sum(1 for tok in query_tokens if tok in haystack)
            recency_bonus = 1.0 / (1.0 + idx * 0.1)
            pin_bonus = 2.0 if fact.pinned else 0.0
            branch_bonus = 0.5 if fact.git_branch == self.engine.git_context.git_branch else 0.0
            score = float(matches) + pin_bonus + branch_bonus + (0.25 * recency_bonus)
            scored.append((score, -idx, fact))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]

    def set_pin(self, category: str, key: str, pinned: bool = True, git_branch: str | None = None) -> DeclarativeFact:
        """Mark an existing declarative fact as pinned (immutable) or unpinned."""
        norm_category = category.strip().lower()
        norm_key = key.strip()
        target_repo = self.engine.git_context.repo_identifier
        target_branch = (git_branch or self.engine.git_context.git_branch).strip()

        with self.engine.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE declarative_facts
                SET pinned = ?
                WHERE repo_identifier = ? AND git_branch = ? AND category = ? AND key = ?;
                """,
                (1 if pinned else 0, target_repo, target_branch, norm_category, norm_key),
            )
            if cursor.rowcount == 0:
                raise KeyError(
                    f"Declarative fact ({norm_category!r}, {norm_key!r}) not found on branch {target_branch!r}."
                )
            row = conn.execute(
                """
                SELECT id, repo_identifier, git_branch, category, key, value_json, pinned, updated_at
                FROM declarative_facts
                WHERE repo_identifier = ? AND git_branch = ? AND category = ? AND key = ?;
                """,
                (target_repo, target_branch, norm_category, norm_key),
            ).fetchone()
        return self._row_to_fact(row)

    def delete_fact(
        self,
        category: str,
        key: str,
        git_branch: str | None = None,
        force: bool = False,
    ) -> bool:
        """Delete a fact by (category, key). Raises PinnedMemoryError if pinned and `force=False`."""
        norm_category = category.strip().lower()
        norm_key = key.strip()
        target_repo = self.engine.git_context.repo_identifier
        target_branch = (git_branch or self.engine.git_context.git_branch).strip()

        with self.engine.connection() as conn:
            row = conn.execute(
                """
                SELECT id, pinned FROM declarative_facts
                WHERE repo_identifier = ? AND git_branch = ? AND category = ? AND key = ?;
                """,
                (target_repo, target_branch, norm_category, norm_key),
            ).fetchone()
            if row is None:
                return False
            if int(row["pinned"]) == 1 and not force:
                raise PinnedMemoryError(
                    f"Cannot delete pinned fact ({norm_category!r}, {norm_key!r}) without force=True or unpinning."
                )
            cursor = conn.execute(
                "DELETE FROM declarative_facts WHERE id = ?;",
                (int(row["id"]),),
            )
            return cursor.rowcount > 0
