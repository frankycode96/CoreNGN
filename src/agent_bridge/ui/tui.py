"""Zero-Bloat Terminal UI (`src/agent_bridge/ui/tui.py`).

Provides both:
1. An interactive full-screen `textual` application (`AgentBridgeTUI`) with 3 tabs:
   - Tab 1: Memory Explorer (Declarative Facts, Episodic Artifacts, Working Checkpoints, Pin/Forget hotkeys)
   - Tab 2: Live Graph Tracer (4-Node LangGraph FSM execution timeline & System-1 telemetry)
   - Tab 3: Workspace & Repo Truth Diff (In-memory representation vs live disk files)
2. A lightweight `rich` snapshot renderer (`render_rich_dashboard_snapshot`) for non-interactive terminals.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from agent_bridge.memory.curation import MemoryCurator
from agent_bridge.memory.declarative import DeclarativeMemoryStore, PinnedMemoryError
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore
from agent_bridge.orchestrator.graph import HybridAgentBridgeRunner


def compute_repo_truth_diffs(engine: LocalMemoryEngine) -> list[dict[str, Any]]:
    """Compare stored `repo_file`/`filesystem` memories against live workspace files on disk."""
    decl = DeclarativeMemoryStore(engine)
    facts = decl.list_facts(limit=100, all_branches=True)
    repo_root: Path = engine.config.repo_root
    diffs: list[dict[str, Any]] = []

    file_facts = [f for f in facts if f.category in {"repo_file", "filesystem", "file"}]
    for fact in file_facts:
        rel_path = fact.key
        disk_path = (repo_root / rel_path).resolve()
        mem_text = str(fact.value) if not isinstance(fact.value, (dict, list)) else json.dumps(fact.value, indent=2)

        try:
            disk_path.relative_to(repo_root.resolve())
            inside_repo = True
        except ValueError:
            inside_repo = False

        if not inside_repo or not disk_path.exists() or not disk_path.is_file():
            status = "DRIFT_MISSING_ON_DISK"
            disk_text = "<MISSING_ON_DISK: Repo Truth overrides stale memory>"
        else:
            disk_text = disk_path.read_text(encoding="utf-8", errors="replace")
            status = "IN_SYNC" if disk_text.strip() == mem_text.strip() else "DRIFT_MODIFIED_ON_DISK"

        unified = "\n".join(
            difflib.unified_diff(
                mem_text.splitlines(),
                disk_text.splitlines()[:40],
                fromfile=f"memory://{fact.git_branch}/{rel_path}",
                tofile=f"disk://{rel_path}",
                lineterm="",
            )
        )
        diffs.append(
            {
                "file": rel_path,
                "git_branch": fact.git_branch,
                "status": status,
                "memory_preview": mem_text[:120],
                "disk_preview": disk_text[:120],
                "unified_diff": unified or "--- Identical ---",
            }
        )

    # If no explicit file facts exist yet, inspect pyproject.toml or README.md as live workspace truth
    if not diffs:
        for default_name in ("pyproject.toml", "README.md"):
            candidate = repo_root / default_name
            if candidate.exists():
                head_text = "\n".join(candidate.read_text(encoding="utf-8", errors="ignore").splitlines()[:8])
                diffs.append(
                    {
                        "file": default_name,
                        "git_branch": engine.git_context.git_branch,
                        "status": "DISK_VERIFIED",
                        "memory_preview": "<Unindexed in declarative_facts>",
                        "disk_preview": head_text[:120],
                        "unified_diff": f"+++ disk://{default_name}\n{head_text}",
                    }
                )
    return diffs


def render_rich_dashboard_snapshot(engine: LocalMemoryEngine, query: str = "") -> str:
    """Render a zero-bloat Rich dashboard snapshot covering Memory Explorer, Graph Trace, and Repo Truth Diff."""
    console = Console(record=True, width=115)
    stats = engine.inspect_stats()
    decl = DeclarativeMemoryStore(engine)
    episodic = EpisodicMemoryStore(engine)
    curator = MemoryCurator(engine, decl, episodic)

    # Header summary panel
    header_text = (
        f"[bold cyan]DB:[/bold cyan] {stats['db_path']} ({stats['journal_mode']}) | "
        f"[bold green]Repo:[/bold green] {stats['repo_identifier']} | "
        f"[bold yellow]Branch:[/bold yellow] {stats['active_git_branch']} | "
        f"[bold magenta]Facts:[/bold magenta] {stats['declarative_facts_count']} (Pinned: {stats['pinned_facts_count']}) | "
        f"[bold blue]Episodes:[/bold blue] {stats['episodic_artifacts_count']}"
    )
    console.print(Panel(header_text, title="Hybrid Agent Bridge — Zero-Bloat Terminal Inspector", border_style="cyan"))

    # Tab 1: Declarative Facts Table
    fact_table = Table(title="Tab 1A: Declarative Facts (Branch-Scoped)", expand=True)
    fact_table.add_column("ID", style="dim", width=5)
    fact_table.add_column("Branch", style="yellow", width=14)
    fact_table.add_column("Pin", width=5)
    fact_table.add_column("Category", style="cyan", width=14)
    fact_table.add_column("Key", style="bold white", width=22)
    fact_table.add_column("Value", style="green")

    for f in decl.list_facts(limit=12):
        fact_table.add_row(
            str(f.id),
            f.git_branch,
            "PIN" if f.pinned else "",
            f.category,
            f.key,
            json.dumps(f.value, ensure_ascii=False)[:60],
        )
    console.print(fact_table)

    # Tab 1B: Episodic Artifacts Table
    ep_table = Table(title="Tab 1B: Episodic Artifacts (FTS5 BM25 + CPU Vectors)", expand=True)
    ep_table.add_column("ID", style="dim", width=5)
    ep_table.add_column("Branch", style="yellow", width=14)
    ep_table.add_column("Score", style="magenta", width=7)
    ep_table.add_column("Task", style="bold white", width=36)
    ep_table.add_column("Final Artifact Preview", style="green")

    for ep in episodic.list_recent(limit=8):
        ep_table.add_row(
            f"ep#{ep.id}",
            ep.git_branch,
            f"{ep.quality_score:.2f}",
            " ".join(ep.task.split())[:36],
            " ".join(ep.final_artifact.split())[:55],
        )
    console.print(ep_table)

    # Tab 3: Workspace & Repo Truth Diff
    diff_table = Table(title="Tab 3: Workspace & Repo Truth Diff (Disk > Memory)", expand=True)
    diff_table.add_column("File", style="bold cyan", width=26)
    diff_table.add_column("Branch", style="yellow", width=12)
    diff_table.add_column("Status", style="bold", width=24)
    diff_table.add_column("In-Memory vs Disk Diff Preview")

    for d in compute_repo_truth_diffs(engine)[:6]:
        diff_table.add_row(
            str(d["file"]),
            str(d["git_branch"]),
            str(d["status"]),
            " ".join(str(d["unified_diff"]).split())[:65],
        )
    console.print(diff_table)

    if query.strip():
        report = curator.explain(query, top_k=5)
        exp_table = Table(
            title=f"Explain Retrieval Breakdown for Query: {query!r} "
            f"(Tokens: {report.total_tokens_used}/{report.max_token_budget})",
            expand=True,
        )
        exp_table.add_column("Rank", width=5)
        exp_table.add_column("Tier", width=12)
        exp_table.add_column("ID", width=18)
        exp_table.add_column("Hybrid", width=8)
        exp_table.add_column("Budget", width=10)
        exp_table.add_column("Justification")
        for c in report.candidates:
            exp_table.add_row(
                str(c.rank),
                c.tier,
                c.identifier,
                f"{c.final_hybrid_score:.3f}",
                "INCLUDED" if c.included_in_budget else "EXCLUDED",
                c.justification[:70],
            )
        console.print(exp_table)

    return console.export_text()


class AgentBridgeTUI(App[None]):
    """Interactive Zero-Bloat Textual Dashboard for Hybrid Agent Bridge."""

    TITLE = "Hybrid Agent Bridge — Local-First 4-Tier Memory & LangGraph TUI"
    CSS = """
    Screen {
        background: $surface;
    }
    #status-banner {
        height: 3;
        padding: 0 1;
        background: $boost;
        color: $text;
        border: solid $primary;
    }
    .panel-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }
    DataTable {
        height: 1fr;
        border: tall $primary-background;
    }
    RichLog {
        height: 1fr;
        border: tall $secondary;
    }
    #controls-bar {
        height: 3;
        margin: 1 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_all", "Refresh"),
        Binding("p", "toggle_pin", "Pin/Unpin Selected Fact"),
        Binding("d", "forget_selected", "Forget Selected Fact"),
        Binding("t", "run_sample_trace", "Run Live FSM Trace"),
    ]

    def __init__(self, engine: LocalMemoryEngine) -> None:
        super().__init__()
        self.engine = engine
        self.decl = DeclarativeMemoryStore(engine)
        self.episodic = EpisodicMemoryStore(engine)
        self.curator = MemoryCurator(engine, self.decl, self.episodic)
        self._fact_row_keys: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="status-banner")
        with TabbedContent(initial="tab-explorer"):
            with TabPane("1. Memory Explorer", id="tab-explorer"):
                yield Input(
                    placeholder="Filter memories or enter query (press Enter to filter, 'p' to pin, 'd' to forget)...",
                    id="search-input",
                )
                with Horizontal():
                    with Vertical():
                        yield Label("Declarative Facts (Git-Scoped)", classes="panel-title")
                        yield DataTable(id="facts-table", cursor_type="row")
                    with Vertical():
                        yield Label("Episodic Artifacts (FTS5 + Vector)", classes="panel-title")
                        yield DataTable(id="episodes-table", cursor_type="row")
                yield Label("Working Memory Checkpoints (LangGraph SqliteSaver)", classes="panel-title")
                yield DataTable(id="checkpoints-table", cursor_type="row")

            with TabPane("2. Live Graph Tracer", id="tab-tracer"):
                with Horizontal(id="controls-bar"):
                    yield Input(
                        value="Write a Python function validate_schema_version for SQLite migrations",
                        id="trace-task-input",
                    )
                    yield Button("Execute 4-Node FSM Trace (t)", id="btn-run-trace", variant="primary")
                yield Label(
                    "Macro-FSM Pipeline: Context_Loader -> SubWorker_Node -> System1_Gatekeeper -> Episodic_Consolidator",
                    classes="panel-title",
                )
                yield DataTable(id="trace-nodes-table", cursor_type="row")
                yield Label("System-1 Decision & Audit Stream", classes="panel-title")
                yield RichLog(id="trace-log", highlight=True, markup=True)

            with TabPane("3. Workspace & Repo Truth Diff", id="tab-diff"):
                yield Label("Side-by-Side In-Memory vs Live Workspace Disk Truth", classes="panel-title")
                yield DataTable(id="diff-table", cursor_type="row")
                yield Label("Unified Diff Inspector (Repo Truth > Memory)", classes="panel-title")
                yield RichLog(id="diff-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._setup_tables()
        self.action_refresh_all()

    def _setup_tables(self) -> None:
        facts_t = self.query_one("#facts-table", DataTable)
        facts_t.add_columns("ID", "Branch", "Pin", "Category", "Key", "Value")

        ep_t = self.query_one("#episodes-table", DataTable)
        ep_t.add_columns("ID", "Branch", "Score", "Task", "Artifact")

        cp_t = self.query_one("#checkpoints-table", DataTable)
        cp_t.add_columns("Thread ID", "Namespace", "Checkpoint ID", "Parent ID")

        tr_t = self.query_one("#trace-nodes-table", DataTable)
        tr_t.add_columns("Step", "Node", "Event", "Metrics / Details")

        df_t = self.query_one("#diff-table", DataTable)
        df_t.add_columns("File", "Branch", "Sync Status", "Memory Preview", "Disk Preview")

    def action_refresh_all(self, filter_query: str = "") -> None:
        """Reload all tables and Repo Truth diffs from the SQLite WAL engine."""
        stats = self.engine.inspect_stats()
        banner = self.query_one("#status-banner", Static)
        banner.update(
            f"DB: {stats['db_path']} [{stats['journal_mode']}] | Repo: {stats['repo_identifier']} | "
            f"Branch: {stats['active_git_branch']} | Facts: {stats['declarative_facts_count']} "
            f"(Pinned: {stats['pinned_facts_count']}) | Episodes: {stats['episodic_artifacts_count']}"
        )

        # Populate Declarative Facts
        facts_t = self.query_one("#facts-table", DataTable)
        facts_t.clear()
        self._fact_row_keys.clear()
        facts = (
            self.decl.search_facts_for_query(filter_query, limit=30)
            if filter_query.strip()
            else self.decl.list_facts(limit=30)
        )
        for f in facts:
            self._fact_row_keys.append((f.category, f.key))
            facts_t.add_row(
                str(f.id),
                f.git_branch,
                "PIN" if f.pinned else "-",
                f.category,
                f.key,
                json.dumps(f.value, ensure_ascii=False)[:48],
            )

        # Populate Episodic Artifacts
        ep_t = self.query_one("#episodes-table", DataTable)
        ep_t.clear()
        episodes = (
            self.episodic.retrieve_hybrid(filter_query, top_k=20)
            if filter_query.strip()
            else self.episodic.list_recent(limit=20)
        )
        for ep in episodes:
            ep_t.add_row(
                f"ep#{ep.id}",
                ep.git_branch,
                f"{ep.quality_score:.2f}",
                " ".join(ep.task.split())[:35],
                " ".join(ep.final_artifact.split())[:45],
            )

        # Populate Working Memory Checkpoints
        cp_t = self.query_one("#checkpoints-table", DataTable)
        cp_t.clear()
        for cp in self.engine.list_working_checkpoints(limit=15):
            cp_t.add_row(
                str(cp["thread_id"]),
                str(cp["checkpoint_ns"]),
                str(cp["checkpoint_id"])[:24],
                str(cp["parent_checkpoint_id"])[:24],
            )

        # Populate Repo Truth Diff
        df_t = self.query_one("#diff-table", DataTable)
        df_log = self.query_one("#diff-log", RichLog)
        df_t.clear()
        df_log.clear()
        for d in compute_repo_truth_diffs(self.engine):
            df_t.add_row(
                str(d["file"]),
                str(d["git_branch"]),
                str(d["status"]),
                str(d["memory_preview"])[:40],
                str(d["disk_preview"])[:40],
            )
            df_log.write(f"[bold cyan]=== {d['file']} ({d['status']}) ===[/bold cyan]\n{d['unified_diff']}\n")

    def action_toggle_pin(self) -> None:
        """Toggle immutability pin on the currently highlighted declarative fact."""
        facts_t = self.query_one("#facts-table", DataTable)
        row_idx = facts_t.cursor_row
        if 0 <= row_idx < len(self._fact_row_keys):
            cat, key = self._fact_row_keys[row_idx]
            current = self.decl.get_fact(cat, key, include_trunk_fallback=False)
            if current is not None:
                self.decl.set_pin(cat, key, pinned=not current.pinned)
                self.action_refresh_all()

    def action_forget_selected(self) -> None:
        """Forget the currently highlighted declarative fact (if not pinned)."""
        facts_t = self.query_one("#facts-table", DataTable)
        row_idx = facts_t.cursor_row
        if 0 <= row_idx < len(self._fact_row_keys):
            cat, key = self._fact_row_keys[row_idx]
            try:
                self.curator.forget(f"{cat}:{key}")
                self.action_refresh_all()
            except PinnedMemoryError as exc:
                trace_log = self.query_one("#trace-log", RichLog)
                trace_log.write(f"[bold red]{exc}[/bold red]")

    def action_run_sample_trace(self) -> None:
        """Execute a live 4-Node LangGraph trace and stream node telemetry to Tab 2."""
        task_input = self.query_one("#trace-task-input", Input)
        task_text = task_input.value.strip() or "Write a Python function check_wal_mode"
        runner = HybridAgentBridgeRunner(config=self.engine.config, engine=self.engine)
        state = runner.run(task=task_text)

        tr_t = self.query_one("#trace-nodes-table", DataTable)
        tr_log = self.query_one("#trace-log", RichLog)
        tr_t.clear()
        tr_log.clear()

        for idx, entry in enumerate(state.get("audit_trail") or [], start=1):
            tr_t.add_row(
                str(idx),
                entry["node"],
                entry["event"],
                json.dumps(entry["details"], ensure_ascii=False),
            )
            tr_log.write(
                f"[bold green]Step {idx} [{entry['node']}][/bold green] -> {entry['event']}: "
                f"{json.dumps(entry['details'])}"
            )
        tr_log.write(f"\n[bold cyan]Final Consolidated Artifact:[/bold cyan]\n{state.get('final_artifact', '')}")
        self.action_refresh_all()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input":
            self.action_refresh_all(filter_query=event.value)
        elif event.input.id == "trace-task-input":
            self.action_run_sample_trace()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run-trace":
            self.action_run_sample_trace()
