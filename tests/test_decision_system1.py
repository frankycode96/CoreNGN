"""Deterministic unit tests for Phase 2: System-1 Decision Controller (`Choice`, `Score`, `Noul`)."""

from __future__ import annotations

import httpx
import pytest

from agent_bridge.decision.jev_client import JevClientError, TypeSafeJevClient
from agent_bridge.decision.local_heuristics import LocalHeuristicDecisionEngine


def test_local_heuristics_choice_score_and_noul_primitives() -> None:
    engine = LocalHeuristicDecisionEngine()

    # 1. Choice operator: probability distribution sums to 1.0
    choice_fact = engine.Choice(
        state={"category": "runtime", "key": "gpu_vram_gb", "value": 8},
        question="Route incoming observation to memory tier",
        options=["declarative", "episodic", "discard"],
    )
    assert choice_fact.chosen == "declarative"
    assert abs(sum(choice_fact.distribution.values()) - 1.0) < 1e-4

    choice_discard = engine.Choice(
        state={"artifact": "def broken(:\n    # TODO fix syntax error"},
        question="Route broken code",
        options=["declarative", "episodic", "discard"],
    )
    assert choice_discard.chosen == "discard"

    # 2. Score operator: penalizes syntax errors and TODO stubs, rewards valid AST
    valid_code = (
        "```python\n"
        "def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:\n"
        "    dot = sum(a * b for a, b in zip(vec_a, vec_b))\n"
        "    return float(dot)\n"
        "```"
    )
    score_good = engine.Score(
        state={"task": "Write function compute_cosine_similarity for vectors", "artifact": valid_code},
        question="Does the artifact satisfy constraints?",
        scale_rubric={"syntax": "valid AST", "completeness": "no stubs"},
    )
    assert score_good.score >= 0.80
    assert score_good.rubric_breakdown["syntax_integrity"] == 1.0
    assert score_good.rubric_breakdown["no_placeholders"] == 1.0

    stub_code = "```python\ndef compute_cosine_similarity(vec_a, vec_b):\n    # TODO implement\n    pass\n```"
    score_stub = engine.Score(
        state={"task": "Write function compute_cosine_similarity", "artifact": stub_code},
        question="Does the artifact satisfy constraints?",
        scale_rubric={},
    )
    assert score_stub.score < 0.35

    # 3. Noul operator: epistemic truth probability in [0.0, 1.0]
    noul_good = engine.Noul(
        state={"task": "Write function compute_cosine_similarity for vectors", "artifact": valid_code},
        proposition="The candidate artifact is complete, syntactically valid, and fulfills the task requirements.",
    )
    assert noul_good.probability >= 0.80

    noul_bad = engine.Noul(
        state={"task": "Write function compute_cosine_similarity", "artifact": stub_code},
        proposition="The candidate artifact is complete, syntactically valid, and fulfills the task requirements.",
    )
    assert noul_bad.probability < 0.35


def test_typesafe_jev_client_transport_and_fallback() -> None:
    def _mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/choice"):
            return httpx.Response(
                200,
                json={
                    "chosen": "episodic",
                    "distribution": {"declarative": 0.1, "episodic": 0.85, "discard": 0.05},
                    "confidence": 0.85,
                },
            )
        if request.url.path.endswith("/score"):
            return httpx.Response(
                200,
                json={
                    "score": 0.92,
                    "rubric_breakdown": {"syntax_integrity": 1.0},
                    "rationale": "Verified by Jev TypeSafe",
                },
            )
        if request.url.path.endswith("/noul"):
            return httpx.Response(
                200,
                json={"probability": 0.94, "signals": {"ast_valid": 1.0}},
            )
        return httpx.Response(500, json={"error": "unexpected path"})

    mock_transport = httpx.MockTransport(_mock_handler)
    http_client = httpx.Client(transport=mock_transport)
    jev = TypeSafeJevClient(
        api_url="https://api.jev-typesafe.local/v1",
        api_key="test-token",
        fallback_to_local=False,
        http_client=http_client,
    )

    c_res = jev.Choice({"artifact": "ok"}, "route", ["declarative", "episodic", "discard"])
    assert c_res.chosen == "episodic"
    assert c_res.backend == "typesafe_jev"

    s_res = jev.Score({"artifact": "ok"}, "quality", {})
    assert s_res.score == 0.92
    assert s_res.backend == "typesafe_jev"

    n_res = jev.Noul({"artifact": "ok"}, "is valid")
    assert n_res.probability == 0.94
    assert n_res.backend == "typesafe_jev"

    # Verify offline fallback when endpoint fails
    failing_transport = httpx.MockTransport(lambda req: httpx.Response(503, text="Service Unavailable"))
    jev_fallback = TypeSafeJevClient(
        api_url="https://api.jev-typesafe.local/v1",
        fallback_to_local=True,
        http_client=httpx.Client(transport=failing_transport),
    )
    fallback_noul = jev_fallback.Noul(
        state={"task": "create add function", "artifact": "def add(a: int, b: int) -> int:\n    return a + b\n"},
        proposition="The artifact is valid",
    )
    assert fallback_noul.backend == "local_embedded"
    assert fallback_noul.probability >= 0.70

    # Verify strict error raising when fallback_to_local=False
    jev_strict = TypeSafeJevClient(
        api_url="https://api.jev-typesafe.local/v1",
        fallback_to_local=False,
        http_client=httpx.Client(transport=failing_transport),
    )
    with pytest.raises(JevClientError):
        jev_strict.Noul({"artifact": "x"}, "is valid")
