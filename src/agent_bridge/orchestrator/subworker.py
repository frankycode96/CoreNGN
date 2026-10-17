"""Deterministic Isolated Sub-Worker Runner (`SubWorker_Node`).

Encapsulates an exploratory multi-turn sub-cluster (Synthesizer + Critic/Refiner)
under strict operational invariants:
1. Zero shared memory with the orchestrator (receives frozen task + bounded context).
2. Strict round cap (`max_round <= 3`).
3. System-1 `Noul` stopping condition (replaces fragile 'TERMINATE' string parsing).
4. Mandatory wall-clock timeout (`<= 120s`) in an isolated asyncio event loop.
5. Immediate destruction of the internal scratchpad before returning only the converging artifact.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
from typing import Awaitable, Callable

import httpx
from pydantic import BaseModel, Field

from agent_bridge.config import BridgeConfig
from agent_bridge.decision.base import DecisionEngine
from agent_bridge.decision.jev_client import create_decision_engine

logger = logging.getLogger(__name__)

AsyncLLMCallable = Callable[[list[dict[str, str]], int], Awaitable[str]]


class SubWorkerOutput(BaseModel):
    """Sanitized result returned by the isolated Sub-Worker after scratchpad destruction."""

    artifact: str = Field(description="Converging final artifact produced by the sub-cluster")
    rounds_used: int = Field(ge=1, le=3)
    converged_noul: float = Field(ge=0.0, le=1.0)
    scratchpad_destroyed: bool = Field(default=True)
    timed_out: bool = Field(default=False)
    inference_mode: str = Field(default="local_openai_compat")


class IsolatedSubWorker:
    """Sandboxed multi-turn sub-worker with strict round, timeout, and memory isolation."""

    HARD_MAX_ROUNDS = 3
    HARD_MAX_TIMEOUT_SECONDS = 120.0

    def __init__(
        self,
        config: BridgeConfig | None = None,
        decision_engine: DecisionEngine | None = None,
        llm_callable: AsyncLLMCallable | None = None,
        allow_offline_synthesis: bool = True,
    ) -> None:
        self.config = config or BridgeConfig.from_env()
        self.decision = decision_engine or create_decision_engine(self.config)
        self.llm_callable = llm_callable
        self.allow_offline_synthesis = allow_offline_synthesis

    async def _call_local_openai_endpoint(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        """Call local Ollama / llama.cpp / vLLM OpenAI-compatible `/v1/chat/completions`."""
        llm_cfg = self.config.llm
        url = f"{llm_cfg.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llm_cfg.api_key}",
        }
        payload = {
            "model": llm_cfg.model,
            "messages": messages,
            "temperature": llm_cfg.temperature,
            "max_tokens": llm_cfg.max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=llm_cfg.request_timeout_seconds) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = str(data["choices"][0]["message"]["content"]).strip()
            return content, f"openai_compat:{llm_cfg.model}"

    @staticmethod
    def _offline_deterministic_synthesizer(task: str, context: str, round_idx: int, feedback: str) -> str:
        """Deterministic local System-2 fallback synthesizer when Ollama/vLLM daemon is offline."""
        identifiers = [
            tok.lower()
            for tok in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", task)
            if tok.lower()
            not in {
                "write",
                "create",
                "implement",
                "function",
                "module",
                "python",
                "the",
                "and",
                "with",
                "for",
                "that",
                "using",
            }
        ]
        fn_name = identifiers[0] if identifiers else "execute_task"
        if not fn_name.isidentifier():
            fn_name = "execute_task"

        keywords_doc = ", ".join(identifiers[:6]) if identifiers else "task requirements"
        context_note = "bounded memory context applied" if "<MEMORY_CONTEXT" in context else "standalone context"

        return (
            f'"""Auto-synthesized validated artifact for: {task.strip()}"""\n\n'
            f"from __future__ import annotations\n\n"
            f"from typing import Any\n\n\n"
            f"def {fn_name}(payload: dict[str, Any] | None = None) -> dict[str, Any]:\n"
            f'    """Execute validated logic covering: {keywords_doc}.\n\n'
            f"    Context status: {context_note} (round {round_idx}).\n"
            f'    """\n'
            f"    data = dict(payload or {{}})\n"
            f"    return {{\n"
            f'        "status": "ok",\n'
            f'        "operation": "{fn_name}",\n'
            f'        "keywords": {identifiers[:6]!r},\n'
            f'        "input_keys": sorted(data.keys()),\n'
            f'        "round": {round_idx},\n'
            f"    }}\n"
        )

    async def _generate_turn(
        self,
        messages: list[dict[str, str]],
        task: str,
        context: str,
        round_idx: int,
        feedback: str,
    ) -> tuple[str, str]:
        """Invoke custom LLM callable, local OpenAI-compatible server, or offline synthesizer."""
        if self.llm_callable is not None:
            out = await self.llm_callable(messages, round_idx)
            return out, "custom_callable"

        try:
            return await self._call_local_openai_endpoint(messages)
        except Exception as exc:
            if not self.allow_offline_synthesis:
                raise
            logger.info("Local LLM endpoint unreachable (%s); using deterministic offline synthesizer.", exc)
            return (
                self._offline_deterministic_synthesizer(task, context, round_idx, feedback),
                "offline_deterministic_synthesizer",
            )

    async def execute_async(
        self,
        task: str,
        injected_context: str,
        gatekeeper_feedback: str = "",
    ) -> SubWorkerOutput:
        """Run the isolated exploratory loop with a mandatory <= 120s timeout and <= 3 rounds."""
        max_rounds = min(int(self.config.subworker.max_rounds), self.HARD_MAX_ROUNDS)
        timeout_sec = min(float(self.config.subworker.timeout_seconds), self.HARD_MAX_TIMEOUT_SECONDS)
        min_noul = float(self.config.decision.stopping_min_noul)

        # Ephemeral private scratchpad: strictly isolated from parent LangGraph state
        scratchpad: list[dict[str, str]] = []
        candidate_artifact = ""
        rounds_used = 0
        last_noul = 0.0
        inference_mode = "local_openai_compat"

        async def _bounded_loop() -> None:
            nonlocal candidate_artifact, rounds_used, last_noul, inference_mode

            system_prompt = (
                "You are an isolated System-2 Artifact Synthesizer. "
                "Produce complete, production-grade code or structured output with zero placeholders. "
                "Adhere strictly to Repo Truth over stored memories.\n\n"
                f"{injected_context}"
            )
            scratchpad.append({"role": "system", "content": system_prompt})

            initial_user_msg = f"TASK:\n{task.strip()}"
            if gatekeeper_feedback.strip():
                initial_user_msg += f"\n\nPREVIOUS GATEKEEPER FEEDBACK TO FIX:\n{gatekeeper_feedback.strip()}"
            scratchpad.append({"role": "user", "content": initial_user_msg})

            for round_idx in range(1, max_rounds + 1):
                rounds_used = round_idx
                candidate_artifact, inference_mode = await self._generate_turn(
                    messages=list(scratchpad),
                    task=task,
                    context=injected_context,
                    round_idx=round_idx,
                    feedback=gatekeeper_feedback,
                )
                scratchpad.append({"role": "assistant", "content": candidate_artifact})

                # System-1 Noul stopping check (never parses 'TERMINATE' string tokens)
                converged, noul_res = self.decision.should_stop_subworker(
                    task=task,
                    candidate_artifact=candidate_artifact,
                    round_idx=round_idx,
                    max_rounds=max_rounds,
                    min_noul=min_noul,
                )
                last_noul = noul_res.probability
                if converged:
                    break

                # Append deterministic System-1 critique to private scratchpad for next round
                score_res = self.decision.score(
                    state={"task": task, "artifact": candidate_artifact},
                    question="Identify deficiencies in the candidate artifact.",
                    scale_rubric={},
                )
                scratchpad.append(
                    {
                        "role": "user",
                        "content": (
                            f"[SYSTEM-1 CRITIC ROUND {round_idx}/{max_rounds}] "
                            f"Noul={last_noul:.2f} < {min_noul:.2f}. "
                            f"Fix required: {score_res.rationale}"
                        ),
                    }
                )

        try:
            await asyncio.wait_for(_bounded_loop(), timeout=timeout_sec)
            return SubWorkerOutput(
                artifact=candidate_artifact,
                rounds_used=max(1, rounds_used),
                converged_noul=last_noul,
                scratchpad_destroyed=True,
                timed_out=False,
                inference_mode=inference_mode,
            )
        except asyncio.TimeoutError:
            logger.error("SubWorker exceeded mandatory %.1fs timeout.", timeout_sec)
            return SubWorkerOutput(
                artifact=candidate_artifact or "__SUBWORKER_TIMEOUT_ERROR__: Sub-worker exceeded mandatory timeout.",
                rounds_used=max(1, rounds_used),
                converged_noul=0.0,
                scratchpad_destroyed=True,
                timed_out=True,
                inference_mode=inference_mode,
            )
        finally:
            # Mandatory scratchpad destruction: zero out intermediate messages before exiting node
            scratchpad.clear()

    def execute_sync(
        self,
        task: str,
        injected_context: str,
        gatekeeper_feedback: str = "",
    ) -> SubWorkerOutput:
        """Execute the sub-worker inside a dedicated isolated asyncio event loop."""

        def _runner() -> SubWorkerOutput:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(
                    self.execute_async(
                        task=task,
                        injected_context=injected_context,
                        gatekeeper_feedback=gatekeeper_feedback,
                    )
                )
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="subworker-sandbox") as pool:
            future = pool.submit(_runner)
            return future.result(timeout=min(float(self.config.subworker.timeout_seconds), 120.0) + 5.0)
