"""Jev TypeSafe API Driver (`typesafe_jev`).

Provides low-latency HTTP transport to a Jev TypeSafe decision sidecar or endpoint
implementing the `Choice`, `Score`, and `Noul` primitives, with optional deterministic
fallback to `LocalHeuristicDecisionEngine` when operating offline.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import httpx

from agent_bridge.config import BridgeConfig, DecisionConfig
from agent_bridge.decision.base import ChoiceResult, DecisionEngine, NoulResult, ScoreResult
from agent_bridge.decision.local_heuristics import LocalHeuristicDecisionEngine

logger = logging.getLogger(__name__)


class JevClientError(RuntimeError):
    """Raised when the Jev TypeSafe endpoint fails and local fallback is disabled."""


class TypeSafeJevClient(DecisionEngine):
    """HTTP client driver for Jev TypeSafe (`typesafe_jev` backend)."""

    def __init__(
        self,
        api_url: str = "https://api.jev-typesafe.local/v1",
        api_key: str | None = None,
        timeout_seconds: float = 5.0,
        fallback_to_local: bool = True,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.fallback_to_local = fallback_to_local
        self._local_fallback = LocalHeuristicDecisionEngine()
        self._client = http_client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        try:
            if self._client is not None:
                resp = self._client.post(url, json=payload, headers=self._headers(), timeout=self.timeout_seconds)
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise JevClientError(f"Expected JSON object from Jev endpoint {url}, got {type(data).__name__}")
            return data
        except Exception as exc:
            raise JevClientError(f"Jev TypeSafe request to {url} failed: {exc}") from exc

    def choice(
        self,
        state: Mapping[str, Any] | str,
        question: str,
        options: Sequence[str],
    ) -> ChoiceResult:
        """Execute remote Jev `Choice(state, question, options)` primitive."""
        payload = {
            "operator": "Choice",
            "state": state if isinstance(state, str) else dict(state),
            "question": question,
            "options": list(options),
        }
        try:
            data = self._post("/choice", payload)
            dist = {str(k): float(v) for k, v in data["distribution"].items()}
            chosen = str(data["chosen"])
            return ChoiceResult(
                chosen=chosen,
                distribution=dist,
                confidence=float(data.get("confidence", dist.get(chosen, 1.0))),
                backend="typesafe_jev",
            )
        except JevClientError as exc:
            if not self.fallback_to_local:
                raise
            logger.debug("Falling back to local_embedded for Choice: %s", exc)
            return self._local_fallback.choice(state=state, question=question, options=options)

    def score(
        self,
        state: Mapping[str, Any] | str,
        question: str,
        scale_rubric: Mapping[str, str] | str,
    ) -> ScoreResult:
        """Execute remote Jev `Score(state, question, scale_rubric)` primitive."""
        payload = {
            "operator": "Score",
            "state": state if isinstance(state, str) else dict(state),
            "question": question,
            "scale_rubric": scale_rubric if isinstance(scale_rubric, str) else dict(scale_rubric),
        }
        try:
            data = self._post("/score", payload)
            return ScoreResult(
                score=float(data["score"]),
                rubric_breakdown={str(k): float(v) for k, v in data.get("rubric_breakdown", {}).items()},
                rationale=str(data.get("rationale", "")),
                backend="typesafe_jev",
            )
        except JevClientError as exc:
            if not self.fallback_to_local:
                raise
            logger.debug("Falling back to local_embedded for Score: %s", exc)
            return self._local_fallback.score(state=state, question=question, scale_rubric=scale_rubric)

    def noul(
        self,
        state: Mapping[str, Any] | str,
        proposition: str,
    ) -> NoulResult:
        """Execute remote Jev `Noul(state, proposition)` primitive."""
        payload = {
            "operator": "Noul",
            "state": state if isinstance(state, str) else dict(state),
            "proposition": proposition,
        }
        try:
            data = self._post("/noul", payload)
            return NoulResult(
                probability=float(data["probability"]),
                proposition=proposition,
                signals={str(k): float(v) for k, v in data.get("signals", {}).items()},
                backend="typesafe_jev",
            )
        except JevClientError as exc:
            if not self.fallback_to_local:
                raise
            logger.debug("Falling back to local_embedded for Noul: %s", exc)
            return self._local_fallback.noul(state=state, proposition=proposition)


def create_decision_engine(config: BridgeConfig | DecisionConfig | None = None) -> DecisionEngine:
    """Instantiate the configured System-1 Decision Controller."""
    if isinstance(config, BridgeConfig):
        dec_cfg = config.decision
        vec_dim = config.sqlite.vector_dim
    elif isinstance(config, DecisionConfig):
        dec_cfg = config
        vec_dim = 64
    else:
        dec_cfg = DecisionConfig()
        vec_dim = 64

    if dec_cfg.backend == "typesafe_jev":
        return TypeSafeJevClient(
            api_url=dec_cfg.jev_api_url,
            api_key=dec_cfg.jev_api_key,
            timeout_seconds=dec_cfg.jev_timeout_seconds,
            fallback_to_local=True,
        )
    return LocalHeuristicDecisionEngine(onnx_model_path=dec_cfg.onnx_model_path, vector_dim=vec_dim)
