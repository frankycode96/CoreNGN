"""Phase 7 Unit Tests: On-Device Quantitative Benchmark Harness (`BenchmarkRunner`)."""

from __future__ import annotations

from pathlib import Path

from agent_bridge.benchmark.runner import BenchmarkRunner
from agent_bridge.cli import main as cli_main
from agent_bridge.config import BridgeConfig


def test_benchmark_runner_100_synthetic_runs_and_markdown_report(tmp_path: Path) -> None:
    cfg = BridgeConfig(db_path=tmp_path / "bench_suite.db", repo_root=tmp_path)
    runner = BenchmarkRunner(config=cfg)

    report = runner.run_suite(synthetic_iterations=100, system1_samples=25)

    # 1. System-1 latency must be sub-50ms and significantly faster than System-2 reference
    assert report.system1_latency.samples == 25
    assert report.system1_latency.mean_ms < 50.0
    assert report.system1_speedup_factor > 50.0

    # 2. Throughput & Context Budget (< 800 tokens vs > 2000 unbounded tokens)
    assert report.throughput.bounded_injected_tokens < 800
    assert report.throughput.unbounded_history_tokens > 1500
    assert report.throughput.token_savings_pct > 50.0
    assert report.throughput.generation_tokens_per_sec > 0.0

    # 3. Footprint over 100 synthetic runs & Zero-VRAM Persist Invariant
    assert report.footprint.synthetic_runs == 100
    assert report.footprint.vram_persistence_mb == 0.0
    assert report.footprint.db_size_after_kb >= report.footprint.db_size_before_kb
    assert report.footprint.hybrid_retrieval_mean_ms < 50.0

    md = runner.format_markdown(report)
    assert "### Hybrid Agent Bridge — On-Device Quantitative Benchmark" in md
    assert "Zero-VRAM" in md


def test_cli_benchmark_and_doctor_commands(tmp_path: Path) -> None:
    db_str = str(tmp_path / "cli_bench.db")
    assert cli_main(["--db-path", db_str, "benchmark", "--iterations", "20", "--format", "both"]) == 0
    assert (
        cli_main(
            [
                "--db-path",
                db_str,
                "doctor",
                "--ollama-url",
                "http://127.0.0.1:59997",
                "--llamacpp-url",
                "http://127.0.0.1:59998",
                "--openai-url",
                "http://127.0.0.1:59999/v1",
            ]
        )
        == 0
    )
