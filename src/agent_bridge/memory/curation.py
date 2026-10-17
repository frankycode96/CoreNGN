"""Memory Hygiene & Curation Toolkit (`src/agent_bridge/memory/curation.py`).

Provides anti-memory-rot administrative operations:
- `pin(category, key)` / `unpin(category, key)`: Immutability protection against automatic overwrites.
- `forget(target_id_or_key)`: Atomic purge across declarative, episodic, FTS5, and sqlite-vec tables.
- `prune(older_than_days, min_salience_score, dry_run)`: Safe batch garbage collection preserving pinned entries.
- `explain(task_query)`: Exact mathematical breakdown of BM25, CPU vector cosine similarity, recency decay,
  branch scope priority, and < 800 token budget inclusion decisions.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field

from agent_bridge.memory.declarative import (
    DeclarativeFact,
    DeclarativeMemoryStore,
    PinnedMemoryError,
)
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore
from agent_bridge.memory.injector import ContextInjector, estimate_tokens


class ForgetResult(BaseModel):
    """Result of an atomic `forget` operation across memory tiers."""

    target: str
    deleted: bool
    tier: Literal["declarative", "episodic", "none"]
    deleted_id: int | None = None
    details: str


class PruneResult(BaseModel):
    """Summary of batch garbage collection (`prune`)."""

    dry_run: bool
    older_than_days: int
    min_salience_score: float
    pruned_declarative_ids: list[int] = Field(default_factory=list)
    pruned_episodic_ids: list[int] = Field(default_factory=list)
    skipped_pinned_count: int = Field(default=0)

    @property
    def total_pruned(self) -> int:
        return len(self.pruned_declarative_ids) + len(self.pruned_episodic_ids)


class ExplainedMemoryCandidate(BaseModel):
    """Granular score and token-budget breakdown for a single episodic or declarative candidate."""

    rank: int
    tier: Literal["repo_truth", "declarative", "episodic"]
    identifier: str
    git_branch: str
    pinned: bool = False
    bm25_raw_normalized: float = 0.0
    bm25_weighted: float = 0.0
    vector_cosine: float = 0.0
    vector_weighted: float = 0.0
    quality_score: float = 1.0
    recency_decay: float = 1.0
    branch_multiplier: float = 1.0
    final_hybrid_score: float = 0.0
    estimated_token_cost: int = 0
    cumulative_tokens_if_included: int = 0
    included_in_budget: bool = True
    justification: str


class ExplainRetrievalReport(BaseModel):
    """Complete retrieval explanation report for `agentic explain "<task>"`."""

    query: str
    active_git_branch: str
    repo_identifier: str
    max_token_budget: int = 799
    total_tokens_used: int = 0
    bm25_weight: float
    vector_weight: float
    candidates: list[ExplainedMemoryCandidate] = Field(default_factory=list)


class MemoryCurator:
    """Administrative toolkit for memory hygiene, immutability pinning, pruning, and retrieval auditing."""

    def __init__(
        self,
        engine: LocalMemoryEngine,
        declarative_store: DeclarativeMemoryStore | None = None,
        episodic_store: EpisodicMemoryStore | None = None,
    ) -> None:
        self.engine = engine
        self.declarative = declarative_store or DeclarativeMemoryStore(engine)
        self.episodic = episodic_store or EpisodicMemoryStore(engine)
        self.injector = ContextInjector(engine, self.declarative, self.episodic)

    def pin(self, category: str, key: str, git_branch: str | None = None) -> DeclarativeFact:
        """Mark a declarative fact as immutable (`pinned=True`), protecting it from automatic overwrites."""
        return self.declarative.set_pin(category=category, key=key, pinned=True, git_branch=git_branch)

    def unpin(self, category: str, key: str, git_branch: str | None = None) -> DeclarativeFact:
        """Restore mutable status (`pinned=False`) to a declarative fact."""
        return self.declarative.set_pin(category=category, key=key, pinned=False, git_branch=git_branch)

    def forget(self, target_id_or_key: str | int, force: bool = False) -> ForgetResult:
        """Atomically delete a declarative fact or episodic artifact across SQLite, FTS5, and sqlite-vec.

        Supported target formats:
        - Integer or `"12"` or `"ep:12"` / `"episode:12"`: Deletes episodic artifact #12 (or declarative #12 if `"fact:12"`).
        - `"category:key"`: Deletes declarative fact `(category, key)` on the active branch.
        - `"key"`: Deletes declarative fact matching `key` on the active branch.
        """
        raw = str(target_id_or_key).strip()
        if not raw:
            raise ValueError("Target identifier for forget() cannot be empty.")

        # 1. Explicit `fact:<id>`
        if raw.lower().startswith("fact:"):
            fact_id = int(raw.split(":", 1)[1])
            return self._delete_declarative_by_id(fact_id, raw, force=force)

        # 2. Explicit `ep:<id>` or `episode:<id>`
        if raw.lower().startswith(("ep:", "episode:")):
            ep_id = int(raw.split(":", 1)[1])
            return self._delete_episodic_by_id(ep_id, raw, force=force)

        # 3. Pure integer: check episodic_artifacts first, then declarative_facts by ID
        if raw.isdigit():
            target_id = int(raw)
            ep_res = self._delete_episodic_by_id(target_id, raw, force=force)
            if ep_res.deleted:
                return ep_res
            return self._delete_declarative_by_id(target_id, raw, force=force)

        # 4. `category:key` format
        if ":" in raw:
            cat, key = raw.split(":", 1)
            deleted = self.declarative.delete_fact(cat.strip(), key.strip(), force=force)
            if deleted:
                return ForgetResult(
                    target=raw,
                    deleted=True,
                    tier="declarative",
                    details=f"Purged declarative fact ({cat.strip().lower()!r}, {key.strip()!r}).",
                )

        # 5. Plain `key` lookup in declarative_facts on active repository
        with self.engine.connection() as conn:
            row = conn.execute(
                """
                SELECT id, category, key, pinned
                FROM declarative_facts
                WHERE repo_identifier = ? AND key = ?
                ORDER BY updated_at DESC
                LIMIT 1;
                """,
                (self.engine.git_context.repo_identifier, raw),
            ).fetchone()
            if row is not None:
                if int(row["pinned"]) == 1 and not force:
                    raise PinnedMemoryError(
                        f"Cannot forget pinned declarative fact ({row['category']!r}, {row['key']!r}) without force=True."
                    )
                conn.execute("DELETE FROM declarative_facts WHERE id = ?;", (int(row["id"]),))
                return ForgetResult(
                    target=raw,
                    deleted=True,
                    tier="declarative",
                    deleted_id=int(row["id"]),
                    details=f"Purged declarative fact id={row['id']} ({row['category']!r}, {row['key']!r}).",
                )

        return ForgetResult(
            target=raw,
            deleted=False,
            tier="none",
            details=f"No matching declarative fact or episodic artifact found for {raw!r}.",
        )

    def _delete_declarative_by_id(self, fact_id: int, raw_target: str, force: bool) -> ForgetResult:
        with self.engine.connection() as conn:
            row = conn.execute(
                "SELECT id, category, key, pinned FROM declarative_facts WHERE id = ?;",
                (fact_id,),
            ).fetchone()
            if row is None:
                return ForgetResult(
                    target=raw_target,
                    deleted=False,
                    tier="none",
                    details=f"Declarative fact id={fact_id} not found.",
                )
            if int(row["pinned"]) == 1 and not force:
                raise PinnedMemoryError(
                    f"Cannot forget pinned declarative fact id={fact_id} ({row['category']!r}, {row['key']!r}) without force=True."
                )
            conn.execute("DELETE FROM declarative_facts WHERE id = ?;", (fact_id,))
            return ForgetResult(
                target=raw_target,
                deleted=True,
                tier="declarative",
                deleted_id=fact_id,
                details=f"Purged declarative fact id={fact_id} ({row['category']!r}, {row['key']!r}).",
            )

    def _delete_episodic_by_id(self, episode_id: int, raw_target: str, force: bool) -> ForgetResult:
        with self.engine.connection() as conn:
            row = conn.execute(
                "SELECT id, task, pinned FROM episodic_artifacts WHERE id = ?;",
                (episode_id,),
            ).fetchone()
            if row is None:
                return ForgetResult(
                    target=raw_target,
                    deleted=False,
                    tier="none",
                    details=f"Episodic artifact id={episode_id} not found.",
                )
            if int(row["pinned"] or 0) == 1 and not force:
                raise PinnedMemoryError(
                    f"Cannot forget pinned episodic artifact id={episode_id} without force=True."
                )
            conn.execute("DELETE FROM episodic_artifacts WHERE id = ?;", (episode_id,))
            if self.engine.sqlite_vec_enabled:
                conn.execute("DELETE FROM vec_episodic WHERE artifact_id = ?;", (episode_id,))
            return ForgetResult(
                target=raw_target,
                deleted=True,
                tier="episodic",
                deleted_id=episode_id,
                details=f"Purged episodic artifact id={episode_id} and synchronized FTS5/vec indexes.",
            )

    def prune(
        self,
        older_than_days: int = 30,
        min_salience_score: float = 0.5,
        dry_run: bool = False,
    ) -> PruneResult:
        """Batch garbage-collect outdated or low-salience memories while preserving all pinned items.

        An episodic artifact is pruned if `pinned == 0` AND (`created_at < cutoff` OR `quality_score < min_salience_score`).
        A declarative fact is pruned if `pinned == 0` AND `updated_at < cutoff`.
        """
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=max(0, int(older_than_days)))
        cutoff_iso = cutoff_dt.isoformat()
        min_score = float(min_salience_score)

        pruned_decl_ids: list[int] = []
        pruned_ep_ids: list[int] = []
        skipped_pinned = 0

        with self.engine.connection() as conn:
            # Count pinned entries that would have matched age/score criteria
            pinned_decl = conn.execute(
                "SELECT COUNT(*) FROM declarative_facts WHERE pinned = 1 AND updated_at < ?;",
                (cutoff_iso,),
            ).fetchone()[0]
            pinned_ep = conn.execute(
                "SELECT COUNT(*) FROM episodic_artifacts WHERE pinned = 1 AND (created_at < ? OR quality_score < ?);",
                (cutoff_iso, min_score),
            ).fetchone()[0]
            skipped_pinned = int(pinned_decl) + int(pinned_ep)

            decl_rows = conn.execute(
                "SELECT id FROM declarative_facts WHERE pinned = 0 AND updated_at < ? ORDER BY id;",
                (cutoff_iso,),
            ).fetchall()
            pruned_decl_ids = [int(r["id"]) for r in decl_rows]

            ep_rows = conn.execute(
                """
                SELECT id FROM episodic_artifacts
                WHERE pinned = 0 AND (created_at < ? OR quality_score < ?)
                ORDER BY id;
                """,
                (cutoff_iso, min_score),
            ).fetchall()
            pruned_ep_ids = [int(r["id"]) for r in ep_rows]

            if not dry_run:
                if pruned_decl_ids:
                    ph_d = ", ".join("?" for _ in pruned_decl_ids)
                    conn.execute(f"DELETE FROM declarative_facts WHERE id IN ({ph_d});", pruned_decl_ids)
                if pruned_ep_ids:
                    ph_e = ", ".join("?" for _ in pruned_ep_ids)
                    conn.execute(f"DELETE FROM episodic_artifacts WHERE id IN ({ph_e});", pruned_ep_ids)
                    if self.engine.sqlite_vec_enabled:
                        conn.execute(f"DELETE FROM vec_episodic WHERE artifact_id IN ({ph_e});", pruned_ep_ids)

        return PruneResult(
            dry_run=dry_run,
            older_than_days=older_than_days,
            min_salience_score=min_score,
            pruned_declarative_ids=pruned_decl_ids,
            pruned_episodic_ids=pruned_ep_ids,
            skipped_pinned_count=skipped_pinned,
        )

    def explain(self, task_query: str, top_k: int = 8) -> ExplainRetrievalReport:
        """Explain hybrid retrieval scoring and < 800 token budget decisions for `task_query`."""
        cfg_budget = self.engine.config.budget
        w_bm25 = cfg_budget.bm25_weight
        w_vec = cfg_budget.vector_weight
        chars_per_tok = cfg_budget.chars_per_token
        ceiling = min(cfg_budget.max_context_tokens, ContextInjector.HARD_CEILING_TOKENS)
        active_branch = self.engine.git_context.git_branch

        bundle = self.injector.build_context(task=task_query, max_tokens=ceiling)
        included_fact_ids = {f.id for f in bundle.included_facts}
        included_ep_ids = {e.id for e in bundle.included_episodes}

        candidates: list[ExplainedMemoryCandidate] = []
        running_tokens = estimate_tokens(
            "<MEMORY_CONTEXT priority=\"REPO_TRUTH > DECLARATIVE > EPISODIC\">\n</MEMORY_CONTEXT>",
            chars_per_tok,
        )
        rank_counter = 0

        # 1. Repo Truth entries
        for truth_line in bundle.repo_truth_entries:
            rank_counter += 1
            cost = estimate_tokens(truth_line, chars_per_tok)
            running_tokens += cost
            candidates.append(
                ExplainedMemoryCandidate(
                    rank=rank_counter,
                    tier="repo_truth",
                    identifier=truth_line.split(":", 1)[0].lstrip("- ").strip(),
                    git_branch="disk_worktree",
                    pinned=True,
                    final_hybrid_score=1.0,
                    estimated_token_cost=cost,
                    cumulative_tokens_if_included=running_tokens,
                    included_in_budget=True,
                    justification="Priority 1 (Repo Truth Hierarchy): Live workspace disk state overrides stored memory.",
                )
            )

        # 2. Candidate Declarative Facts
        fact_candidates = self.declarative.search_facts_for_query(task_query, limit=top_k)
        for fact in fact_candidates:
            rank_counter += 1
            val_str = json.dumps(fact.value, ensure_ascii=False)
            fact_line = f"- ({fact.category}) {fact.key} = {val_str[:160]}"
            cost = estimate_tokens(fact_line, chars_per_tok)
            included = fact.id in included_fact_ids
            if included:
                running_tokens += cost
            branch_mult = 1.0 if fact.git_branch == active_branch else 0.92
            candidates.append(
                ExplainedMemoryCandidate(
                    rank=rank_counter,
                    tier="declarative",
                    identifier=f"{fact.category}:{fact.key}",
                    git_branch=fact.git_branch,
                    pinned=fact.pinned,
                    final_hybrid_score=1.0 if fact.pinned else round(0.85 * branch_mult, 4),
                    branch_multiplier=branch_mult,
                    estimated_token_cost=cost,
                    cumulative_tokens_if_included=running_tokens if included else running_tokens + cost,
                    included_in_budget=included,
                    justification=(
                        f"{'INCLUDED' if included else 'EXCLUDED'}: "
                        f"Declarative fact ({'pinned immutable' if fact.pinned else 'mutable'}, "
                        f"branch={fact.git_branch}, cost={cost} tok)."
                    ),
                )
            )

        # 3. Candidate Episodic Artifacts with full BM25 + Vector + Recency breakdown
        ep_candidates = self.episodic.retrieve_hybrid(task_query, top_k=top_k)
        for ep in ep_candidates:
            rank_counter += 1
            compact_line = f"- [ep#{ep.id} score={ep.hybrid_score:.2f}] Task: {ep.task[:100]} => Artifact: {ep.final_artifact[:220]}"
            cost = estimate_tokens(compact_line, chars_per_tok)
            included = ep.id in included_ep_ids
            if included:
                running_tokens += cost

            bm25_contrib = round(w_bm25 * ep.bm25_score, 4)
            vec_contrib = round(w_vec * ep.vector_score, 4)
            branch_mult = 1.0 if ep.git_branch == active_branch else 0.92

            candidates.append(
                ExplainedMemoryCandidate(
                    rank=rank_counter,
                    tier="episodic",
                    identifier=f"ep#{ep.id}",
                    git_branch=ep.git_branch,
                    pinned=ep.pinned,
                    bm25_raw_normalized=ep.bm25_score,
                    bm25_weighted=bm25_contrib,
                    vector_cosine=ep.vector_score,
                    vector_weighted=vec_contrib,
                    quality_score=ep.quality_score,
                    recency_decay=ep.recency_decay,
                    branch_multiplier=branch_mult,
                    final_hybrid_score=ep.hybrid_score,
                    estimated_token_cost=cost,
                    cumulative_tokens_if_included=running_tokens if included else running_tokens + cost,
                    included_in_budget=included,
                    justification=(
                        f"{'INCLUDED' if included else 'EXCLUDED (budget ceiling)'}: "
                        f"Hybrid={ep.hybrid_score:.4f} [BM25={ep.bm25_score:.2f}*{w_bm25} ({bm25_contrib:.3f}) + "
                        f"VecCos={ep.vector_score:.2f}*{w_vec} ({vec_contrib:.3f})] * "
                        f"Recency={ep.recency_decay:.2f} * Branch({ep.git_branch})={branch_mult:.2f}, cost={cost} tok."
                    ),
                )
            )

        return ExplainRetrievalReport(
            query=task_query,
            active_git_branch=active_branch,
            repo_identifier=self.engine.git_context.repo_identifier,
            max_token_budget=ceiling,
            total_tokens_used=bundle.estimated_tokens,
            bm25_weight=w_bm25,
            vector_weight=w_vec,
            candidates=candidates,
        )
