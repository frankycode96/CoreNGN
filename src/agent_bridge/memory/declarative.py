"""Declarative Memory Layer (`declarative_facts`).

Implements deterministic Upsert semantics on `(category, key)`.
Temporal conflicts are never delegated to an LLM: the latest write is the ground truth
among stored memories (and subject to Repo Truth Hierarchy overrides at injection time).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agent_bridge.memory.engine import LocalMemoryEngine


class DeclarativeFact(BaseModel):
    """Typed representation of a single row in `declarative_facts`."""

    id: int
    category: str
    key: str
    value: Any
    updated_at: str = Field(description="ISO-8601 UTC timestamp of latest deterministic upsert")


class DeclarativeMemoryStore:
    """Deterministic CRUD and search operations for Declarative Memory."""

    def __init__(self, engine: LocalMemoryEngine) -> None:
        self.engine = engine

    def upsert_fact(self, category: str, key: str, value: Any) -> DeclarativeFact:
        """Atomically insert or overwrite a fact identified by (category, key)."""
        norm_category = category.strip().lower()
        norm_key = key.strip()
        if not norm_category or not norm_key:
            raise ValueError("Both 'category' and 'key' must be non-empty strings.")

        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.engine.connection() as conn:
            conn.execute(
                """
                INSERT INTO declarative_facts (category, key, value_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(category, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at;
                """,
                (norm_category, norm_key, serialized, now_iso),
            )
            row = conn.execute(
                """
                SELECT id, category, key, value_json, updated_at
                FROM declarative_facts
                WHERE category = ? AND key = ?;
                """,
                (norm_category, norm_key),
            ).fetchone()

        if row is None:  # pragma: no cover
            raise RuntimeError(f"Failed to retrieve upserted fact ({norm_category}, {norm_key})")

        return DeclarativeFact(
            id=int(row["id"]),
            category=str(row["category"]),
            key=str(row["key"]),
            value=json.loads(row["value_json"]),
            updated_at=str(row["updated_at"]),
        )

    def get_fact(self, category: str, key: str) -> DeclarativeFact | None:
        """Retrieve a single declarative fact by (category, key), or None if absent."""
        norm_category = category.strip().lower()
        norm_key = key.strip()
        with self.engine.connection() as conn:
            row = conn.execute(
                """
                SELECT id, category, key, value_json, updated_at
                FROM declarative_facts
                WHERE category = ? AND key = ?;
                """,
                (norm_category, norm_key),
            ).fetchone()

        if row is None:
            return None

        return DeclarativeFact(
            id=int(row["id"]),
            category=str(row["category"]),
            key=str(row["key"]),
            value=json.loads(row["value_json"]),
            updated_at=str(row["updated_at"]),
        )

    def list_facts(self, category: str | None = None, limit: int = 50) -> list[DeclarativeFact]:
        """List declarative facts ordered by most recently updated first."""
        with self.engine.connection() as conn:
            if category:
                rows = conn.execute(
                    """
                    SELECT id, category, key, value_json, updated_at
                    FROM declarative_facts
                    WHERE category = ?
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?;
                    """,
                    (category.strip().lower(), int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, category, key, value_json, updated_at
                    FROM declarative_facts
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?;
                    """,
                    (int(limit),),
                ).fetchall()

        return [
            DeclarativeFact(
                id=int(r["id"]),
                category=str(r["category"]),
                key=str(r["key"]),
                value=json.loads(r["value_json"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]

    def search_facts_for_query(self, query: str, limit: int = 12) -> list[DeclarativeFact]:
        """Retrieve relevant declarative facts ranked by token overlap and recency."""
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
            score = float(matches) + (0.25 * recency_bonus)
            scored.append((score, -idx, fact))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]

    def delete_fact(self, category: str, key: str) -> bool:
        """Delete a fact by (category, key). Returns True if a row was removed."""
        with self.engine.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM declarative_facts WHERE category = ? AND key = ?;",
                (category.strip().lower(), key.strip()),
            )
            return cursor.rowcount > 0
