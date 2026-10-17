"""Zero-Bloat Terminal UI (Textual & Rich Dashboard) Subsystem."""

from agent_bridge.ui.tui import (
    AgentBridgeTUI,
    compute_repo_truth_diffs,
    render_rich_dashboard_snapshot,
)

__all__ = [
    "AgentBridgeTUI",
    "compute_repo_truth_diffs",
    "render_rich_dashboard_snapshot",
]
