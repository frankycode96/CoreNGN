"""Bounded Context Injector (< 800 tokens) with RLM Repo Truth Hierarchy enforcement.

Guarantees that injected memory blocks never saturate the local model's KV-cache
and enforces the invariant: Repo Truth Hierarchy > Declarative Memory > Episodic Memory.
Uses `RLMWorkspaceToolkit` (`peek`, `search`) to extract surgical line-bounded slices
from workspace files rather than dumping full files into context.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from pydantic import BaseModel, Field

from agent_bridge.memory.declarative import DeclarativeFact, DeclarativeMemoryStore
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicArtifact, EpisodicMemoryStore
from agent_bridge.rlm.tools import RLMWorkspaceToolkit

_FILE_PATH_RE = re.compile(
    r"(?:[a-zA-Z0-9_\-\.]+/)+[a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+|[a-zA-Z0-9_\-]+\.(?:py|toml|md|json|yaml|yml|sql|sh)"
)
_SYMBOL_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b")


class InjectedContextBundle(BaseModel):
    """Structured result of bounded context assembly."""

    formatted_block: str = Field(description="Structured prompt block ready for LLM injection")
    estimated_tokens: int = Field(description="Conservative token count estimate (< 800)")
    max_token_budget: int = Field(default=800)
    repo_truth_entries: list[str] = Field(default_factory=list)
    included_facts: list[DeclarativeFact] = Field(default_factory=list)
    included_episodes: list[EpisodicArtifact] = Field(default_factory=list)
    truncated: bool = Field(default=False)


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Compute a conservative upper-bound token estimate for local BPE tokenizers."""
    if not text:
        return 0
    char_est = math.ceil(len(text) / max(1, chars_per_token))
    word_est = math.ceil(len(text.split()) * 1.35)
    return max(char_est, word_est)


class ContextInjector:
    """Assembles a strictly bounded (< 800 tokens) context block for System-2 execution."""

    HARD_CEILING_TOKENS = 799  # Strictly < 800 tokens per architectural pillar

    def __init__(
        self,
        engine: LocalMemoryEngine,
        declarative_store: DeclarativeMemoryStore | None = None,
        episodic_store: EpisodicMemoryStore | None = None,
        rlm_toolkit: RLMWorkspaceToolkit | None = None,
    ) -> None:
        self.engine = engine
        self.declarative = declarative_store or DeclarativeMemoryStore(engine)
        self.episodic = episodic_store or EpisodicMemoryStore(engine)
        self.rlm = rlm_toolkit or RLMWorkspaceToolkit(engine.config.repo_root)

    def _probe_repo_truth(self, task: str, facts: list[DeclarativeFact]) -> tuple[list[str], set[tuple[str, str]]]:
        """Verify live repository filesystem truth via RLM `peek`/`search` and invalidate stale file memories."""
        repo_root: Path = self.engine.config.repo_root
        self.rlm.repo_root = repo_root.resolve()
        referenced_paths: set[str] = set(_FILE_PATH_RE.findall(task))

        for fact in facts:
            if fact.category in {"repo_file", "filesystem", "file"}:
                referenced_paths.add(fact.key)

        task_symbols = [
            s
            for s in _SYMBOL_RE.findall(task)
            if s.lower()
            not in {"write", "create", "implement", "python", "function", "refactor", "check", "aligned", "with", "that"}
        ]

        repo_truth_lines: list[str] = []
        invalidated_keys: set[tuple[str, str]] = set()

        for rel_str in sorted(referenced_paths)[:6]:
            candidate = (repo_root / rel_str).resolve()
            try:
                candidate.relative_to(repo_root.resolve())
            except ValueError:
                continue

            if candidate.exists() and candidate.is_file():
                stat = candidate.stat()
                peek_res = self.rlm.peek(rel_str, start_line=1, end_line=4)
                first_non_empty = next(
                    (
                        ln.split(": ", 1)[1].strip()
                        for ln in peek_res.snippet.splitlines()
                        if ": " in ln and ln.split(": ", 1)[1].strip()
                    ),
                    "",
                )[:90]

                # Also run surgical RLM search if a task symbol matches inside the file
                matched_symbol_note = ""
                for sym in task_symbols[:3]:
                    hits = self.rlm.search(sym, rel_path=rel_str, max_matches=1)
                    if hits:
                        matched_symbol_note = f" | rlm_match L{hits[0].line_number}: {hits[0].line_text[:60]!r}"
                        break

                repo_truth_lines.append(
                    f"- {rel_str}: EXISTS ({stat.st_size}B, {peek_res.total_lines}L) | head: {first_non_empty!r}{matched_symbol_note}"
                )
            else:
                repo_truth_lines.append(f"- {rel_str}: MISSING_ON_DISK (Repo Truth overrides memory)")
                for fact in facts:
                    if fact.key == rel_str:
                        invalidated_keys.add((fact.category, fact.key))

        return repo_truth_lines, invalidated_keys

    def build_context(
        self,
        task: str,
        max_tokens: int | None = None,
        top_k_facts: int | None = None,
        top_k_episodes: int | None = None,
    ) -> InjectedContextBundle:
        """Retrieve top-N memories and pack them under a hard < 800 token budget."""
        cfg_budget = self.engine.config.budget
        chars_per_tok = cfg_budget.chars_per_token

        requested_max = max_tokens if max_tokens is not None else cfg_budget.max_context_tokens
        effective_ceiling = min(int(requested_max), self.HARD_CEILING_TOKENS)

        k_facts = top_k_facts if top_k_facts is not None else cfg_budget.top_k_declarative
        k_episodes = top_k_episodes if top_k_episodes is not None else cfg_budget.top_k_episodic

        raw_facts = self.declarative.search_facts_for_query(task, limit=k_facts)
        raw_episodes = self.episodic.retrieve_hybrid(task, top_k=k_episodes)
        repo_truth_lines, invalidated_keys = self._probe_repo_truth(task, raw_facts)

        header = "<MEMORY_CONTEXT priority=\"REPO_TRUTH > DECLARATIVE > EPISODIC\">"
        footer = "</MEMORY_CONTEXT>"
        lines: list[str] = [header]

        def _fits_with(candidate_lines: list[str]) -> bool:
            trial = "\n".join([*lines, *candidate_lines, footer])
            return estimate_tokens(trial, chars_per_tok) <= effective_ceiling

        truncated = False
        included_truth: list[str] = []
        included_facts: list[DeclarativeFact] = []
        included_episodes: list[EpisodicArtifact] = []

        # 1. Repo Truth Hierarchy (Highest Precedence)
        if repo_truth_lines:
            section_header = ["[REPO_TRUTH]"]
            if _fits_with(section_header):
                lines.extend(section_header)
                for truth_line in repo_truth_lines:
                    if _fits_with([truth_line]):
                        lines.append(truth_line)
                        included_truth.append(truth_line)
                    else:
                        truncated = True
                        break

        # 2. Declarative Memory Facts
        valid_facts = [f for f in raw_facts if (f.category, f.key) not in invalidated_keys]
        if valid_facts:
            section_header = ["[DECLARATIVE_FACTS]"]
            if _fits_with(section_header):
                lines.extend(section_header)
                for fact in valid_facts:
                    val_str = json.dumps(fact.value, ensure_ascii=False)
                    if len(val_str) > 160:
                        val_str = val_str[:157] + "..."
                    fact_line = f"- [{fact.memory_type}] ({fact.category}) {fact.key} = {val_str}"
                    if _fits_with([fact_line]):
                        lines.append(fact_line)
                        included_facts.append(fact)
                    else:
                        truncated = True
                        break

        # 3. Validated Episodic Artifacts
        if raw_episodes:
            section_header = ["[EPISODIC_ARTIFACTS]"]
            if _fits_with(section_header):
                lines.extend(section_header)
                for ep in raw_episodes:
                    compact_task = " ".join(ep.task.split())[:100]
                    compact_art = " ".join(ep.final_artifact.split())[:220]
                    ep_line = (
                        f"- [ep#{ep.id} score={ep.hybrid_score:.2f} rrf={ep.rrf_score:.2f}] "
                        f"Task: {compact_task} => Artifact: {compact_art}"
                    )
                    if _fits_with([ep_line]):
                        lines.append(ep_line)
                        included_episodes.append(ep)
                    else:
                        truncated = True
                        break

        lines.append(footer)
        formatted = "\n".join(lines)
        token_count = estimate_tokens(formatted, chars_per_tok)

        if token_count > effective_ceiling:
            max_chars = max(32, effective_ceiling * chars_per_tok - len(footer) - 8)
            formatted = formatted[:max_chars].rstrip() + "\n" + footer
            token_count = estimate_tokens(formatted, chars_per_tok)
            truncated = True

        return InjectedContextBundle(
            formatted_block=formatted,
            estimated_tokens=token_count,
            max_token_budget=effective_ceiling,
            repo_truth_entries=included_truth,
            included_facts=included_facts,
            included_episodes=included_episodes,
            truncated=truncated,
        )


def inject_bounded_context(
    engine: LocalMemoryEngine,
    task: str,
    max_tokens: int = 799,
) -> InjectedContextBundle:
    """Convenience entry point to assemble bounded memory context (< 800 tokens)."""
    injector = ContextInjector(engine=engine)
    return injector.build_context(task=task, max_tokens=max_tokens)
