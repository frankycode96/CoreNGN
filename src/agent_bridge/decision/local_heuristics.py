"""Offline Local System-1 Driver (`local_embedded`).

Executes deterministic AST syntax verification, 5-layer security sandbox validation,
placeholder detection, lexical/vector alignment, and optional quantized ONNX Runtime
CPU classification in sub-millisecond time. Never invokes the generative System-2 LLM.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from agent_bridge.decision.base import ChoiceResult, DecisionEngine, NoulResult, ScoreResult
from agent_bridge.memory.episodic import CPUEmbedder
from agent_bridge.sandbox.validator import CodeSandboxValidator

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r"\b(?:TODO|FIXME|XXX|HACK|NotImplementedError|PLACEHOLDER|DUMMY_IMPLEMENTATION)\b",
    re.IGNORECASE,
)
_KV_FACT_RE = re.compile(r"^\s*([a-zA-Z0-9_\-\.]+)\s*[:=]\s*(.+)$")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{3,}")


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    """Compute numerically stable softmax probability distribution over option keys."""
    if not logits:
        return {}
    max_logit = max(logits.values())
    exps = {k: math.exp(v - max_logit) for k, v in logits.items()}
    total = sum(exps.values()) or 1.0
    return {k: round(v / total, 6) for k, v in exps.items()}


class LocalHeuristicDecisionEngine(DecisionEngine):
    """Offline CPU-based System-1 Decision Controller (`local_embedded` driver)."""

    def __init__(
        self,
        onnx_model_path: Path | None = None,
        vector_dim: int = 64,
        sandbox_validator: CodeSandboxValidator | None = None,
    ) -> None:
        self.onnx_model_path = onnx_model_path
        self.embedder = CPUEmbedder(dim=vector_dim)
        self.sandbox = sandbox_validator or CodeSandboxValidator()
        self._onnx_session: Any = None
        if onnx_model_path and onnx_model_path.exists():
            self._try_load_onnx(onnx_model_path)

    def _try_load_onnx(self, model_path: Path) -> None:
        """Optionally initialize an ONNX Runtime CPUInferenceSession if installed."""
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            self._onnx_session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception:  # pragma: no cover
            self._onnx_session = None

    @staticmethod
    def _normalize_state(state: Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(state, str):
            return {"text": state, "artifact": state}
        return dict(state)

    @staticmethod
    def inspect_python_syntax(text: str) -> tuple[bool, str]:
        """Verify Python AST validity for fenced Python blocks or raw Python code."""
        cleaned = text.strip()
        if not cleaned:
            return False, "Empty artifact text"

        fenced_blocks = _CODE_FENCE_RE.findall(cleaned)
        candidates = fenced_blocks if fenced_blocks else []

        if not candidates and any(
            cleaned.startswith(prefix)
            for prefix in ("def ", "class ", "import ", "from ", "async def ", "@", '"""', "#")
        ):
            candidates = [cleaned]

        for idx, block in enumerate(candidates):
            try:
                tree = ast.parse(block)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        body = [
                            stmt
                            for stmt in node.body
                            if not (
                                isinstance(stmt, ast.Expr)
                                and isinstance(getattr(stmt, "value", None), ast.Constant)
                                and isinstance(getattr(stmt.value, "value", None), str)
                            )
                        ]
                        if len(body) == 1 and isinstance(body[0], ast.Pass):
                            return False, f"Function '{node.name}' contains only a stub 'pass' statement"
                        if (
                            len(body) == 1
                            and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and body[0].value.value is Ellipsis
                        ):
                            return False, f"Function '{node.name}' contains only an ellipsis '...' stub"
            except SyntaxError as exc:
                return False, f"Python AST SyntaxError in block #{idx + 1} (line {exc.lineno}): {exc.msg}"

        return True, "AST syntax valid"

    def _compute_alignment(self, task: str, artifact: str) -> float:
        """Compute hybrid lexical + CPU vector alignment between task and artifact."""
        if not artifact.strip():
            return 0.0
        if not task.strip():
            return 0.85

        task_tokens = {
            t.lower()
            for t in _TOKEN_RE.findall(task)
            if t.lower() not in {"the", "and", "for", "with", "that", "this", "from", "into", "write", "create"}
        }
        art_lower = artifact.lower()
        if task_tokens:
            matched = sum(1 for tok in task_tokens if tok in art_lower)
            lexical_cov = matched / len(task_tokens)
        else:
            lexical_cov = 0.8

        v_task = self.embedder.embed(task)
        v_art = self.embedder.embed(artifact)
        norm_t = float(np.linalg.norm(v_task))
        norm_a = float(np.linalg.norm(v_art))
        vec_sim = float(np.dot(v_task, v_art) / (norm_t * norm_a)) if (norm_t > 0 and norm_a > 0) else 0.0
        vec_sim = max(0.0, vec_sim)

        length_factor = min(1.0, len(artifact.strip()) / 40.0)
        combined = (0.55 * min(1.0, lexical_cov * 1.35) + 0.45 * min(1.0, vec_sim * 1.4)) * length_factor
        return round(max(0.0, min(1.0, combined)), 4)

    def choice(
        self,
        state: Mapping[str, Any] | str,
        question: str,
        options: Sequence[str],
    ) -> ChoiceResult:
        """Deterministic System-1 multi-class choice with calibrated Softmax distribution."""
        unique_opts = [str(o) for o in options if str(o).strip()]
        if not unique_opts:
            raise ValueError("Choice requires at least one non-empty option.")

        st = self._normalize_state(state)
        text = str(st.get("artifact") or st.get("text") or st.get("value") or json.dumps(st, default=str))
        task = str(st.get("task") or "")
        logits: dict[str, float] = {opt: 0.0 for opt in unique_opts}

        lowered_text = text.lower().strip()
        syntax_ok, _ = self.inspect_python_syntax(text)
        sandbox_res = self.sandbox.validate_artifact(text)
        has_placeholders = bool(_PLACEHOLDER_RE.search(text))
        has_error = any(
            err in lowered_text
            for err in ("traceback (most recent call last)", "syntaxerror:", "exception:", "__subworker_timeout_error__")
        )

        for opt in unique_opts:
            opt_l = opt.lower()
            if opt_l in {"discard", "abort", "reject"}:
                if not lowered_text or has_error or has_placeholders or not syntax_ok or not sandbox_res.safe:
                    logits[opt] += 3.2
                else:
                    logits[opt] -= 1.5
            elif opt_l in {"declarative", "fact", "config"}:
                is_short_fact = (len(text) < 280 and ("\n" not in text.strip())) or bool(_KV_FACT_RE.match(text))
                has_fact_keys = "category" in st and "key" in st
                if is_short_fact or has_fact_keys:
                    logits[opt] += 2.6
                else:
                    logits[opt] -= 0.4
            elif opt_l in {"episodic", "artifact", "consolidate", "accept"}:
                if syntax_ok and sandbox_res.safe and not has_placeholders and not has_error and len(text.strip()) >= 20:
                    logits[opt] += 2.8
                else:
                    logits[opt] -= 1.2
            elif opt_l == "retry":
                if (
                    (not syntax_ok or not sandbox_res.safe or has_placeholders)
                    and int(st.get("retry_count", 0)) < int(st.get("max_retries", 2))
                ):
                    logits[opt] += 2.5
                else:
                    logits[opt] -= 0.8
            else:
                sim = self._compute_alignment(f"{question} {opt}", f"{task} {text}")
                logits[opt] += sim * 2.0

        dist = _softmax(logits)
        chosen = max(dist.items(), key=lambda kv: kv[1])[0]
        drift = round(1.0 - sum(dist.values()), 6)
        dist[chosen] = round(max(0.0, min(1.0, dist[chosen] + drift)), 6)

        return ChoiceResult(
            chosen=chosen,
            distribution=dist,
            confidence=dist[chosen],
            backend="local_embedded",
        )

    def score(
        self,
        state: Mapping[str, Any] | str,
        question: str,
        scale_rubric: Mapping[str, str] | str,
    ) -> ScoreResult:
        """Deterministic System-1 rubric scorer returning a normalized score in [0.0, 1.0]."""
        st = self._normalize_state(state)
        artifact = str(st.get("artifact") or st.get("text") or "")
        task = str(st.get("task") or question or "")

        if not artifact.strip():
            return ScoreResult(
                score=0.0,
                rubric_breakdown={
                    "syntax_integrity": 0.0,
                    "no_placeholders": 0.0,
                    "sandbox_safety": 0.0,
                    "task_alignment": 0.0,
                },
                rationale="Artifact is empty.",
                backend="local_embedded",
            )

        syntax_ok, syntax_msg = self.inspect_python_syntax(artifact)
        syntax_score = 1.0 if syntax_ok else 0.0

        sandbox_res = self.sandbox.validate_artifact(artifact)
        sandbox_score = sandbox_res.safety_score

        placeholder_matches = _PLACEHOLDER_RE.findall(artifact)
        placeholder_score = 1.0 if not placeholder_matches else max(0.0, 1.0 - 0.5 * len(placeholder_matches))

        alignment_score = self._compute_alignment(task, artifact)

        lowered = artifact.lower()
        has_fatal_error = any(
            marker in lowered
            for marker in ("traceback (most recent call last):", "__subworker_timeout_error__", "unhandled exception:")
        )

        if has_fatal_error or not syntax_ok or not sandbox_res.safe:
            total_score = round(0.15 * alignment_score, 4)
        else:
            total_score = round(
                0.35 * syntax_score
                + 0.25 * placeholder_score
                + 0.15 * sandbox_score
                + 0.25 * max(0.65, alignment_score),
                4,
            )

        reasons: list[str] = []
        if not syntax_ok:
            reasons.append(syntax_msg)
        if not sandbox_res.safe:
            reasons.extend(sandbox_res.violations)
        if placeholder_matches:
            reasons.append(f"Contains forbidden placeholders: {', '.join(sorted(set(placeholder_matches)))}")
        if has_fatal_error:
            reasons.append("Contains runtime error or traceback marker")
        if not reasons:
            reasons.append("All AST syntax, 5-layer sandbox, anti-placeholder, and task alignment checks passed")

        return ScoreResult(
            score=max(0.0, min(1.0, total_score)),
            rubric_breakdown={
                "syntax_integrity": round(syntax_score, 4),
                "no_placeholders": round(placeholder_score, 4),
                "sandbox_safety": round(sandbox_score, 4),
                "task_alignment": round(alignment_score, 4),
            },
            rationale="; ".join(reasons),
            backend="local_embedded",
        )

    def noul(
        self,
        state: Mapping[str, Any] | str,
        proposition: str,
    ) -> NoulResult:
        """Compute scalar probability in [0.0, 1.0] that `proposition` holds true in `state`."""
        st = self._normalize_state(state)
        artifact = str(st.get("artifact") or st.get("text") or "")
        task = str(st.get("task") or "")

        score_res = self.score(state=st, question=proposition, scale_rubric={})
        prop_lower = proposition.lower()

        asks_negative = any(w in prop_lower for w in ("error", "invalid", "fail", "incomplete", "missing"))
        base_prob = score_res.score

        if "verified" in st and isinstance(st["verified"], bool):
            base_prob = (base_prob + (1.0 if st["verified"] else 0.0)) / 2.0

        final_prob = (1.0 - base_prob) if asks_negative else base_prob
        final_prob = round(max(0.0, min(1.0, final_prob)), 4)

        return NoulResult(
            probability=final_prob,
            proposition=proposition,
            signals={
                **score_res.rubric_breakdown,
                "artifact_length": float(len(artifact)),
                "task_length": float(len(task)),
            },
            backend="local_embedded",
        )
