"""Git Context Resolution & Branch-Scoping Semantics (`git_context.py`).

Inspects the active workspace via deterministic subprocess calls (`git rev-parse`,
`git status --porcelain`) with strict timeouts, resolving repository identity,
active branch, commit SHA, and dirty status. Gracefully degrades to
`workspace:default` / branch `default` when executed outside a Git repository.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

PRIMARY_BRANCHES: frozenset[str] = frozenset({"main", "master", "trunk", "default"})


class GitContextInfo(BaseModel):
    """Immutable snapshot of the workspace Git state used to scope memory operations."""

    repo_root: Path = Field(description="Resolved filesystem root of the repository or workspace")
    repo_identifier: str = Field(
        default="workspace:default",
        description="Canonical repository identifier (e.g. 'git:ai-memory:a1b2c3d4' or 'workspace:default')",
    )
    git_branch: str = Field(
        default="default",
        description="Active Git branch name (or 'default' outside Git)",
    )
    commit_sha: str = Field(
        default="none",
        description="Short 7-12 char commit SHA of HEAD, or 'none' if unborn/non-git",
    )
    is_dirty: bool = Field(
        default=False,
        description="True if working tree or index has uncommitted modifications",
    )
    is_git_repo: bool = Field(
        default=False,
        description="True if workspace resides inside a valid Git work tree",
    )

    @property
    def is_primary_branch(self) -> bool:
        """Return True if the active branch is a primary trunk branch (main, master, trunk, default)."""
        return self.git_branch.lower() in PRIMARY_BRANCHES

    def visible_branches(self) -> list[str]:
        """Return ordered list of branches visible from the current branch context.

        Branch Isolation Invariant:
        - On `main`/`master`/`default`: ONLY `main`, `master`, and `default` memories are visible.
          Unmerged feature branch memories are strictly excluded to prevent cross-branch pollution.
        - On a feature branch (e.g. `feat/x`): the active feature branch has priority 1,
          while baseline trunk branches (`main`, `master`, `default`) are inherited as priority 2.
        """
        active = self.git_branch.strip() or "default"
        if active.lower() in PRIMARY_BRANCHES:
            ordered = [active]
            for trunk in ("main", "master", "default"):
                if trunk not in ordered:
                    ordered.append(trunk)
            return ordered

        # Feature branch: active branch first, then trunk branches
        return [active, "main", "master", "default"]


def _run_git(cwd: Path, args: list[str], timeout: float = 2.0) -> str | None:
    """Execute a read-only git command and return stripped stdout, or None on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_git_context(
    workspace_path: Path | str | None = None,
    branch_override: str | None = None,
    repo_id_override: str | None = None,
) -> GitContextInfo:
    """Deterministically resolve the active Git context for `workspace_path`.

    Parameters:
        workspace_path: Directory to inspect (defaults to `Path.cwd()`).
        branch_override: Explicit branch name override for testing or pinned contexts.
        repo_id_override: Explicit repository identifier override.

    Returns:
        Populated `GitContextInfo` instance; falls back to `workspace:default` outside Git.
    """
    base_path = Path(workspace_path or Path.cwd()).expanduser().resolve()
    if base_path.is_file():
        base_path = base_path.parent

    if not base_path.exists():
        return GitContextInfo(
            repo_root=base_path,
            repo_identifier=repo_id_override or "workspace:default",
            git_branch=branch_override or "default",
            commit_sha="none",
            is_dirty=False,
            is_git_repo=False,
        )

    toplevel = _run_git(base_path, ["rev-parse", "--show-toplevel"])
    if not toplevel:
        return GitContextInfo(
            repo_root=base_path,
            repo_identifier=repo_id_override or "workspace:default",
            git_branch=branch_override or "default",
            commit_sha="none",
            is_dirty=False,
            is_git_repo=False,
        )

    resolved_root = Path(toplevel).resolve()
    path_digest = hashlib.blake2b(str(resolved_root).encode("utf-8"), digest_size=4).hexdigest()
    default_repo_id = f"git:{resolved_root.name}:{path_digest}"
    repo_identifier = repo_id_override or default_repo_id

    raw_branch = _run_git(resolved_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if not raw_branch:
        # Handle newly initialized repository with unborn HEAD
        symbolic = _run_git(resolved_root, ["symbolic-ref", "--short", "HEAD"])
        raw_branch = symbolic or "main"

    short_sha = _run_git(resolved_root, ["rev-parse", "--short", "HEAD"]) or "uncommitted"
    if raw_branch == "HEAD":
        raw_branch = f"detached:{short_sha}"

    active_branch = (branch_override or raw_branch).strip() or "default"
    porcelain = _run_git(resolved_root, ["status", "--porcelain"])
    is_dirty = bool(porcelain)

    return GitContextInfo(
        repo_root=resolved_root,
        repo_identifier=repo_identifier,
        git_branch=active_branch,
        commit_sha=short_sha,
        is_dirty=is_dirty,
        is_git_repo=True,
    )
