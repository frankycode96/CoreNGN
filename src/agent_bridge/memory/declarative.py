"""Declarative Memory Layer (`declarative_facts`).

Implements deterministic Upsert semantics scoped by `(repo_identifier, git_branch, category, key)`
with a 6-type semantic taxonomy inspired by NeuroCore:
- `RULE`: Immutable operational constraint (automatically pinned `pinned=True`).
- `FACT`: Verified objective statement (persistent).
- `PREFERENCE`: Developer/project stylistic preference (persistent).
- `IDENTITY`: System/project identity metadata (persistent).
- `EXPERIENCE`: Distilled rule-of-thumb from past episodes (persistent).
- `BELIEF`: Provisional or working hypothesis with a default **30-day TTL** (`expires_at`);
  automatically filtered out once expired and extended on active access.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_bridge.memory.engine import LocalMemoryEngine

MemoryTaxonomyType = Literal["FACT", "BELIEF", "PREFERENCE", "IDENTITY", "RULE", "EXPERIENCE"]
VALID_MEMORY_TYPES: frozenset[str] = frozenset({"FACT", "BELIEF", "PREFERENCE", "IDENTITY", "RULE", "EXPERIENCE"})
DEFAULT_BELIEF_TTL_DAYS = 30


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
    memory_type: MemoryTaxonomyType = Field(default="FACT")
    pinned: bool = Field(default=False, description="True if protected against automatic upserts/pruning")
    access_count: int = Field(default=0, ge=0)
    expires_at: str | None = Field(default=None, description="Optional ISO-8601 UTC expiration timestamp (for BELIEF)")
    updated_at: str = Field(description="ISO-8601 UTC timestamp of latest deterministic upsert")


def _infer_memory_type(category: str, explicit_type: str | None = None) -> MemoryTaxonomyType:
    """Resolve canonical MemoryTaxonomyType from explicit argument or category name."""
    if explicit_type:
        upper = explicit_type.strip().upper()
        if upper in VALID_MEMORY_TYPES:
            return upper  # type: ignore[return-value]
        raise ValueError(f"Unsupported memory_type {explicit_type!r}. Allowed: {sorted(VALID_MEMORY_TYPES)}")

    cat_l = category.strip().lower()
    if cat_l in {"rule", "rules", "constraint", "constraints", "security", "invariant"}:
        return "RULE"
    if cat_l in {"belief", "beliefs", "hypothesis", "temp", "provisional"}:
        return "BELIEF"
    if cat_l in {"preference", "preferences", "style"}:
        return "PREFERENCE"
    if cat_l in {"identity", "project", "persona"}:
        return "IDENTITY"
    if cat_l in {"experience", "heuristic", "lesson"}:
        return "EXPERIENCE"
    return "FACT"


class DeclarativeMemoryStore:
    """Deterministic Git-scoped CRUD, taxonomy TTL management, pinning, and search for Declarative Memory."""

    def __init__(self, engine: LocalMemoryEngine) -> None:
        self.engine = engine

    @staticmethod
    def _row_to_fact(row: Any) -> DeclarativeFact:
        raw_type = str(row["memory_type"] if "memory_type" in row.keys() else "FACT").upper()
        mem_type: MemoryTaxonomyType = raw_type if raw_type in VALID_MEMORY_TYPES else "FACT"  # type: ignore[assignment]
        return DeclarativeFact(
            id=int(row["id"]),
            repo_identifier=str(row["repo_identifier"]),
            git_branch=str(row["git_branch"]),
            category=str(row["category"]),
            key=str(row["key"]),
            value=json.loads(row["value_json"]),
            memory_type=mem_type,
            pinned=bool(int(row["pinned"])),
            access_count=int(row["access_count"] if "access_count" in row.keys() else 0),
            expires_at=str(row["expires_at"]) if ("expires_at" in row.keys() and row["expires_at"]) else None,
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _is_expired(expires_at: str | None, now_dt: datetime | None = None) -> bool:
        if not expires_at:
            return False
        try:
            exp = datetime.fromisoformat(expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            ref = now_dt or datetime.now(timezone.utc)
            return exp <= ref
        except Exception:
            return False

    def upsert_fact(
        self,
        category: str,
        key: str,
        value: Any,
        git_branch: str | None = None,
        repo_identifier: str | None = None,
        pinned: bool | None = None,
        respect_pin: bool = True,
        memory_type: str | None = None,
        ttl_days: float | None = None,
    ) -> DeclarativeFact:
        """Atomically insert or overwrite a fact identified by (repo_identifier, git_branch, category, key).

        - If `memory_type == 'RULE'` (and `pinned` is not explicitly set), `pinned` defaults to `False` unless
          explicitly requested or `memory_type='RULE'` is passed explicitly (`explicit_rule_pin`).
        - If `memory_type == 'BELIEF'` (or `ttl_days` is specified), computes `expires_at` (default 30 days).
        """
        norm_category = category.strip().lower()
        norm_key = key.strip()
        if not norm_category or not norm_key:
            raise ValueError("Both 'category' and 'key' must be non-empty strings.")

        resolved_type = _infer_memory_type(norm_category, memory_type)
        target_repo = (repo_identifier or self.engine.git_context.repo_identifier).strip()
        target_branch = (git_branch or self.engine.git_context.git_branch).strip()

        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()

        expires_iso: str | None = None
        if ttl_days is not None:
            expires_iso = (now_dt + timedelta(days=float(ttl_days))).isoformat()
        elif resolved_type == "BELIEF":
            expires_iso = (now_dt + timedelta(days=DEFAULT_BELIEF_TTL_DAYS)).isoformat()

        # If caller explicitly passed `memory_type="RULE"`, auto-pin unless `pinned=False`
        auto_pin_rule = (memory_type is not None and resolved_type == "RULE" and pinned is None)

        with self.engine.connection() as conn:
            existing = conn.execute(
                """
                SELECT id, pinned, value_json, access_count
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

            if pinned is not None:
                pin_int = int(bool(pinned))
            elif auto_pin_rule:
                pin_int = 1
            elif existing is not None:
                pin_int = int(existing["pinned"])
            else:
                pin_int = 0

            conn.execute(
                """
                INSERT INTO declarative_facts (
                    repo_identifier, git_branch, category, key, value_json,
                    memory_type, pinned, access_count, expires_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(repo_identifier, git_branch, category, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    memory_type = excluded.memory_type,
                    pinned = excluded.pinned,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at;
                """,
                (
                    target_repo,
                    target_branch,
                    norm_category,
                    norm_key,
                    serialized,
                    resolved_type,
                    pin_int,
                    expires_iso,
                    now_iso,
                ),
            )
            row = conn.execute(
                """
                SELECT id, repo_identifier, git_branch, category, key, value_json,
                       memory_type, pinned, access_count, expires_at, updated_at
                FROM declarative_facts
                WHERE repo_identifier = ? AND git_branch = ? AND category = ? AND key = ?;
                """,
                (target_repo, target_branch, norm_category, norm_key),
            ).fetchone()

        if row is None:  # pragma: no cover
            raise RuntimeError(f"Failed to retrieve upserted fact ({norm_category}, {norm_key})")

        self.engine.record_audit_event(
            action="upsert",
            target_tier="declarative",
            target_ref=f"{norm_category}:{norm_key}",
            details={
                "fact_id": int(row["id"]),
                "memory_type": resolved_type,
                "pinned": bool(pin_int),
                "expires_at": expires_iso,
            },
        )
        return self._row_to_fact(row)

    def get_fact(
        self,
        category: str,
        key: str,
        git_branch: str | None = None,
        repo_identifier: str | None = None,
        include_trunk_fallback: bool = True,
        bump_access: bool = True,
    ) -> DeclarativeFact | None:
        """Retrieve a non-expired declarative fact scoped to the active branch."""
        norm_category = category.strip().lower()
        norm_key = key.strip()
        target_repo = (repo_identifier or self.engine.git_context.repo_identifier).strip()

        if git_branch is not None:
            branches = [git_branch.strip()]
        elif include_trunk_fallback:
            branches = self.engine.git_context.visible_branches()
        else:
            branches = [self.engine.git_context.git_branch]

        now_dt = datetime.now(timezone.utc)
        with self.engine.connection() as conn:
            for br in branches:
                row = conn.execute(
                    """
                    SELECT id, repo_identifier, git_branch, category, key, value_json,
                           memory_type, pinned, access_count, expires_at, updated_at
                    FROM declarative_facts
                    WHERE repo_identifier = ? AND git_branch = ? AND category = ? AND key = ?;
                    """,
                    (target_repo, br, norm_category, norm_key),
                ).fetchone()
                if row is not None:
                    if self._is_expired(row["expires_at"], now_dt) and int(row["pinned"]) == 0:
                        continue
                    if bump_access:
                        conn.execute(
                            "UPDATE declarative_facts SET access_count = access_count + 1 WHERE id = ?;",
                            (int(row["id"]),),
                        )
                    fact = self._row_to_fact(row)
                    if bump_access:
                        fact.access_count += 1
                    return fact
        return None

    def list_facts(
        self,
        category: str | None = None,
        limit: int = 50,
        git_branch: str | None = None,
        all_branches: bool = False,
        include_expired: bool = False,
    ) -> list[DeclarativeFact]:
        """List declarative facts respecting Git branch isolation and TTL expiration."""
        target_repo = self.engine.git_context.repo_identifier
        now_dt = datetime.now(timezone.utc)

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
                        SELECT id, repo_identifier, git_branch, category, key, value_json,
                               memory_type, pinned, access_count, expires_at, updated_at
                        FROM declarative_facts
                        WHERE repo_identifier = ? AND category = ?
                        ORDER BY pinned DESC, updated_at DESC, id DESC
                        LIMIT ?;
                        """,
                        (target_repo, category.strip().lower(), int(limit) * 2),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT id, repo_identifier, git_branch, category, key, value_json,
                               memory_type, pinned, access_count, expires_at, updated_at
                        FROM declarative_facts
                        WHERE repo_identifier = ?
                        ORDER BY pinned DESC, updated_at DESC, id DESC
                        LIMIT ?;
                        """,
                        (target_repo, int(limit) * 2),
                    ).fetchall()
                facts = [
                    self._row_to_fact(r)
                    for r in rows
                    if include_expired or int(r["pinned"]) == 1 or not self._is_expired(r["expires_at"], now_dt)
                ]
                return facts[: int(limit)]

            placeholders = ", ".join("?" for _ in branches)
            params: list[Any] = [target_repo, *branches]
            cat_clause = ""
            if category:
                cat_clause = "AND category = ?"
                params.append(category.strip().lower())

            rows = conn.execute(
                f"""
                SELECT id, repo_identifier, git_branch, category, key, value_json,
                       memory_type, pinned, access_count, expires_at, updated_at
                FROM declarative_facts
                WHERE repo_identifier = ? AND git_branch IN ({placeholders}) {cat_clause}
                ORDER BY updated_at DESC, id DESC;
                """,
                params,
            ).fetchall()

        branch_priority = {b: idx for idx, b in enumerate(branches)}
        best_by_key: dict[tuple[str, str], tuple[int, DeclarativeFact]] = {}
        for r in rows:
            if not include_expired and int(r["pinned"]) == 0 and self._is_expired(r["expires_at"], now_dt):
                continue
            fact = self._row_to_fact(r)
            prio = branch_priority.get(fact.git_branch, 999)
            ck = (fact.category, fact.key)
            if ck not in best_by_key or prio < best_by_key[ck][0]:
                best_by_key[ck] = (prio, fact)

        deduped = [item[1] for item in best_by_key.values()]
        deduped.sort(key=lambda f: (f.pinned, f.updated_at, f.id), reverse=True)
        return deduped[: int(limit)]

    def search_facts_for_query(self, query: str, limit: int = 12) -> list[DeclarativeFact]:
        """Retrieve relevant branch-scoped declarative facts ranked by taxonomy weight, pin status, and token overlap."""
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

        type_weights = {
            "RULE": 1.5,
            "IDENTITY": 0.6,
            "PREFERENCE": 0.5,
            "FACT": 0.4,
            "EXPERIENCE": 0.4,
            "BELIEF": 0.15,
        }
        scored: list[tuple[float, int, DeclarativeFact]] = []
        for idx, fact in enumerate(all_facts):
            haystack = f"{fact.category} {fact.key} {json.dumps(fact.value)}".lower()
            matches = sum(1 for tok in query_tokens if tok in haystack)
            recency_bonus = 1.0 / (1.0 + idx * 0.1)
            pin_bonus = 2.0 if fact.pinned else 0.0
            tax_bonus = type_weights.get(fact.memory_type, 0.2)
            branch_bonus = 0.5 if fact.git_branch == self.engine.git_context.git_branch else 0.0
            score = float(matches) + pin_bonus + tax_bonus + branch_bonus + (0.25 * recency_bonus)
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
            row = conn.execute(
                """
                SELECT id FROM declarative_facts
                WHERE repo_identifier IN (?, 'workspace:default')
                  AND git_branch IN (?, 'default', 'main', 'master')
                  AND category = ? AND key = ?
                ORDER BY (repo_identifier = ?) DESC, (git_branch = ?) DESC, id DESC
                LIMIT 1;
                """,
                (target_repo, target_branch, norm_category, norm_key, target_repo, target_branch),
            ).fetchone()
            if row is None:
                raise KeyError(
                    f"Declarative fact ({norm_category!r}, {norm_key!r}) not found on branch {target_branch!r}."
                )
            fact_id = int(row["id"])
            conn.execute(
                "UPDATE declarative_facts SET pinned = ? WHERE id = ?;",
                (1 if pinned else 0, fact_id),
            )
            updated_row = conn.execute(
                """
                SELECT id, repo_identifier, git_branch, category, key, value_json,
                       memory_type, pinned, access_count, expires_at, updated_at
                FROM declarative_facts
                WHERE id = ?;
                """,
                (fact_id,),
            ).fetchone()

        self.engine.record_audit_event(
            action="pin" if pinned else "unpin",
            target_tier="declarative",
            target_ref=f"{norm_category}:{norm_key}",
            details={"fact_id": fact_id, "pinned": pinned},
        )
        return self._row_to_fact(updated_row)

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
            deleted = cursor.rowcount > 0

        if deleted:
            self.engine.record_audit_event(
                action="delete_fact",
                target_tier="declarative",
                target_ref=f"{norm_category}:{norm_key}",
                details={"fact_id": int(row["id"]), "forced": force},
            )
        return deleted
