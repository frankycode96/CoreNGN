"""Recursive Language Model (RLM) Workspace Navigation Toolkit (`src/agent_bridge/rlm/tools.py`).

Implements deterministic, zero-LLM-overhead RLM inspection primitives inspired by NeuroCore:
- `peek(rel_path, start_line, end_line)`: Surgical line-range slicing confined to `repo_root`.
- `search(pattern, rel_path, max_matches)`: Regex/symbol search across workspace files.
- `chunk(text_or_rel_path, chunk_lines, overlap_lines)`: Overlapping window chunking with token estimates.
- `set_variable(key, value)` / `get_variable(key)`: Ephemeral variable store for multi-step RLM inspection.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RLMPeekResult(BaseModel):
    """Line-bounded slice of a workspace file."""

    rel_path: str
    exists: bool
    total_lines: int = 0
    start_line: int = 1
    end_line: int = 1
    snippet: str = ""


class RLMSearchMatch(BaseModel):
    """Single regex or symbol match inside a repository file."""

    rel_path: str
    line_number: int
    line_text: str
    context_snippet: str


class RLMChunk(BaseModel):
    """Overlapping chunk of a large text or source file."""

    chunk_index: int
    start_line: int
    end_line: int
    content: str
    estimated_chars: int


class RLMWorkspaceToolkit:
    """Surgical file navigation primitives enforcing Repo Truth without saturating the < 800 token budget."""

    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self._variables: dict[str, Any] = {}

    def _resolve_safe_path(self, rel_path: str) -> Path | None:
        """Resolve `rel_path` and guarantee it remains strictly inside `self.repo_root`."""
        candidate = (self.repo_root / rel_path).resolve()
        try:
            candidate.relative_to(self.repo_root)
            return candidate
        except ValueError:
            return None

    def peek(self, rel_path: str, start_line: int = 1, end_line: int = 30) -> RLMPeekResult:
        """Read lines `[start_line, end_line]` (1-indexed, inclusive) from `rel_path`."""
        safe_path = self._resolve_safe_path(rel_path)
        if safe_path is None or not safe_path.exists() or not safe_path.is_file():
            return RLMPeekResult(rel_path=rel_path, exists=False)

        try:
            lines = safe_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return RLMPeekResult(rel_path=rel_path, exists=False)

        total = len(lines)
        if total == 0:
            return RLMPeekResult(rel_path=rel_path, exists=True, total_lines=0, start_line=1, end_line=1, snippet="")

        s_idx = max(1, min(int(start_line), total))
        e_idx = max(s_idx, min(int(end_line), total))
        numbered = [f"L{ln}: {lines[ln - 1]}" for ln in range(s_idx, e_idx + 1)]

        return RLMPeekResult(
            rel_path=rel_path,
            exists=True,
            total_lines=total,
            start_line=s_idx,
            end_line=e_idx,
            snippet="\n".join(numbered),
        )

    def search(
        self,
        pattern: str,
        rel_path: str | None = None,
        max_matches: int = 8,
        context_lines: int = 1,
    ) -> list[RLMSearchMatch]:
        """Find regex or identifier matches inside `rel_path` (or top-level Python files in `repo_root`)."""
        if not pattern.strip():
            return []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        target_files: list[tuple[str, Path]] = []
        if rel_path:
            sp = self._resolve_safe_path(rel_path)
            if sp and sp.exists() and sp.is_file():
                target_files.append((rel_path, sp))
        else:
            for p in sorted(self.repo_root.rglob("*.py"))[:40]:
                if ".venv" in p.parts or "__pycache__" in p.parts or ".git" in p.parts:
                    continue
                rel = str(p.relative_to(self.repo_root))
                target_files.append((rel, p))

        matches: list[RLMSearchMatch] = []
        for rel, file_path in target_files:
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for idx, line in enumerate(lines):
                if regex.search(line):
                    line_no = idx + 1
                    c_start = max(0, idx - context_lines)
                    c_end = min(len(lines), idx + context_lines + 1)
                    ctx = " | ".join(f"L{i + 1}:{lines[i].strip()}" for i in range(c_start, c_end))
                    matches.append(
                        RLMSearchMatch(
                            rel_path=rel,
                            line_number=line_no,
                            line_text=line.strip()[:120],
                            context_snippet=ctx[:180],
                        )
                    )
                    if len(matches) >= max_matches:
                        return matches
        return matches

    def chunk(
        self,
        text_or_rel_path: str,
        chunk_lines: int = 30,
        overlap_lines: int = 5,
    ) -> list[RLMChunk]:
        """Split a workspace file or raw text into overlapping line chunks."""
        safe_path = self._resolve_safe_path(text_or_rel_path) if "\n" not in text_or_rel_path else None
        if safe_path is not None and safe_path.exists() and safe_path.is_file():
            raw_text = safe_path.read_text(encoding="utf-8", errors="replace")
        else:
            raw_text = text_or_rel_path

        lines = raw_text.splitlines()
        if not lines:
            return []

        window = max(5, int(chunk_lines))
        overlap = max(0, min(int(overlap_lines), window - 1))
        step = max(1, window - overlap)

        chunks: list[RLMChunk] = []
        idx = 0
        chunk_num = 0
        while idx < len(lines):
            chunk_num += 1
            end_idx = min(len(lines), idx + window)
            slice_text = "\n".join(lines[idx:end_idx])
            chunks.append(
                RLMChunk(
                    chunk_index=chunk_num,
                    start_line=idx + 1,
                    end_line=end_idx,
                    content=slice_text,
                    estimated_chars=len(slice_text),
                )
            )
            if end_idx >= len(lines):
                break
            idx += step
        return chunks

    def set_variable(self, key: str, value: Any) -> None:
        """Store an intermediate result in the RLM session scratchpad."""
        self._variables[key.strip()] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored intermediate result from the RLM session scratchpad."""
        return self._variables.get(key.strip(), default)
