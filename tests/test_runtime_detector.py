"""Phase 5 Unit Tests: Universal Local Runtime Auto-Discovery (`RuntimeDetector`)."""

from __future__ import annotations

import httpx

from agent_bridge.runtime.detector import RuntimeDetector


def test_runtime_detector_discovers_ollama_llamacpp_and_vllm() -> None:
    def _mock_router(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "http://localhost:11434/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3.2:3b"}]},
            )
        if url == "http://localhost:11434/api/show":
            return httpx.Response(
                200,
                json={
                    "model_info": {"qwen2.context_length": 16384},
                    "details": {"parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
                },
            )
        if url == "http://localhost:8080/props":
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {"n_ctx": 4096},
                    "model_alias": "deepseek-coder-6.7b-q4.gguf",
                    "total_slots": 2,
                },
            )
        if url == "http://localhost:8080/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "deepseek-coder-6.7b-q4.gguf"}]},
            )
        if url == "http://localhost:1234/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "max_model_len": 8192}]},
            )
        return httpx.Response(404, json={"error": "not found"})

    detector = RuntimeDetector(transport=httpx.MockTransport(_mock_router))
    report = detector.discover_all_sync()

    assert len(report.providers) == 3
    assert all(p.reachable for p in report.providers)
    assert report.active_provider is not None
    assert report.recommended_context_budget_tokens <= 799
    assert report.zero_vram_persistence_verified is True

    by_type = {p.provider_type: p for p in report.providers}
    assert by_type["ollama"].primary_model == "qwen2.5-coder:7b"
    assert by_type["ollama"].detected_context_window == 16384
    assert by_type["ollama"].details.get("quantization_level") == "Q4_K_M"

    assert by_type["llamacpp"].detected_context_window == 4096
    assert "deepseek-coder-6.7b-q4.gguf" in by_type["llamacpp"].models

    assert by_type["openai_compat"].detected_context_window == 8192


def test_runtime_detector_graceful_offline_handling() -> None:
    def _offline_router(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    detector = RuntimeDetector(transport=httpx.MockTransport(_offline_router))
    report = detector.discover_all_sync()

    assert len(report.providers) == 3
    assert all(not p.reachable for p in report.providers)
    assert report.active_provider is None
    assert report.recommended_context_budget_tokens == 600
    assert all(p.error is not None for p in report.providers)
