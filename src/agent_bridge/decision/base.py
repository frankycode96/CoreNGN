"""System-1 Decision Controller Abstract Base Contract (`JevDecisionAdapter`).

Defines the typed primitives of the Jev-Mem pattern:
- Choice(state, question, options) -> chosen option + probability distribution
- Score(state, question, scale_rubric) -> normalized scalar in [0.0, 1.0]
- Noul(state, proposition) -> scalar truth probability in [0.0, 1.0]

All routing decisions, memory classification, compliance validation, and sub-worker
stopping criteria flow through this System-1 contract—never through the generative LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, Field, model_validator


class ChoiceResult(BaseModel):
    """Typed output of the System-1 `Choice` operator."""

    chosen: str = Field(description="Selected option from the allowed set")
    distribution: dict[str, float] = Field(
        description="Normalized probability distribution over all provided options (sums to 1.0)"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the chosen option")
    backend: str = Field(default="local_embedded")

    @model_validator(mode="after")
    def _validate_choice_in_distribution(self) -> ChoiceResult:
        if self.chosen not in self.distribution:
            raise ValueError(f"Chosen option {self.chosen!r} missing from distribution keys.")
        return self


class ScoreResult(BaseModel):
    """Typed output of the System-1 `Score` operator."""

    score: float = Field(ge=0.0, le=1.0, description="Normalized rubric evaluation score in [0.0, 1.0]")
    rubric_breakdown: dict[str, float] = Field(default_factory=dict)
    rationale: str = Field(default="")
    backend: str = Field(default="local_embedded")


class NoulResult(BaseModel):
    """Typed output of the System-1 `Noul` epistemic truth operator."""

    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Scalar probability in [0.0, 1.0] that the proposition holds true in the given state",
    )
    proposition: str
    signals: dict[str, float] = Field(default_factory=dict)
    backend: str = Field(default="local_embedded")


class GatekeeperVerdict(BaseModel):
    """Composite verdict produced by `System1_Gatekeeper` node."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    compliance_probability: float = Field(ge=0.0, le=1.0)
    action: Literal["accept", "retry", "abort"]
    feedback: str


class DecisionEngine(ABC):
    """Abstract System-1 Decision Controller implementing Choice, Score, and Noul."""

    @abstractmethod
    def choice(
        self,
        state: Mapping[str, Any] | str,
        question: str,
        options: Sequence[str],
    ) -> ChoiceResult:
        """Select an option from `options` with a full probability distribution."""

    @abstractmethod
    def score(
        self,
        state: Mapping[str, Any] | str,
        question: str,
        scale_rubric: Mapping[str, str] | str,
    ) -> ScoreResult:
        """Evaluate `state` against `scale_rubric` returning a normalized score in [0.0, 1.0]."""

    @abstractmethod
    def noul(
        self,
        state: Mapping[str, Any] | str,
        proposition: str,
    ) -> NoulResult:
        """Compute scalar probability in [0.0, 1.0] that `proposition` is true given `state`."""

    # --- Standard PascalCase aliases matching Jev-Mem specification ---

    def Choice(
        self,
        state: Mapping[str, Any] | str,
        question: str,
        options: Sequence[str],
    ) -> ChoiceResult:
        """Jev-Mem canonical alias for `choice(state, question, options)`."""
        return self.choice(state=state, question=question, options=options)

    def Score(
        self,
        state: Mapping[str, Any] | str,
        question: str,
        scale_rubric: Mapping[str, str] | str,
    ) -> ScoreResult:
        """Jev-Mem canonical alias for `score(state, question, scale_rubric)`."""
        return self.score(state=state, question=question, scale_rubric=scale_rubric)

    def Noul(
        self,
        state: Mapping[str, Any] | str,
        proposition: str,
    ) -> NoulResult:
        """Jev-Mem canonical alias for `noul(state, proposition)`."""
        return self.noul(state=state, proposition=proposition)

    # --- High-Level Operational Gatekeepers ---

    def route_memory_ingestion(
        self,
        state: Mapping[str, Any],
    ) -> ChoiceResult:
        """Route a candidate memory payload at ingestion time."""
        return self.choice(
            state=state,
            question="Which memory tier should store this information?",
            options=["declarative", "episodic", "discard"],
        )

    def should_stop_subworker(
        self,
        task: str,
        candidate_artifact: str,
        round_idx: int,
        max_rounds: int = 3,
        min_noul: float = 0.75,
    ) -> tuple[bool, NoulResult]:
        """Determine whether the isolated sub-worker loop has converged without parsing 'TERMINATE'."""
        state = {
            "task": task,
            "artifact": candidate_artifact,
            "round_idx": round_idx,
            "max_rounds": max_rounds,
        }
        noul_res = self.noul(
            state=state,
            proposition="The candidate artifact is complete, syntactically valid, and fulfills the task requirements.",
        )
        converged = (noul_res.probability >= min_noul) or (round_idx >= max_rounds)
        return converged, noul_res

    def evaluate_gatekeeper(
        self,
        task: str,
        artifact: str,
        retry_count: int = 0,
        max_retries: int = 2,
        min_score: float = 0.70,
    ) -> GatekeeperVerdict:
        """System-1 Gatekeeper validation for generated artifacts."""
        eval_state = {
            "task": task,
            "artifact": artifact,
            "retry_count": retry_count,
        }
        score_res = self.score(
            state=eval_state,
            question="Does the artifact satisfy syntax, completeness, and task alignment constraints?",
            scale_rubric={
                "syntax_integrity": "Code or structured output parses cleanly with zero syntax errors.",
                "no_placeholders": "Free of TODO, FIXME, pass placeholders, or unimplemented stubs.",
                "task_alignment": "Directly addresses the core concepts and deliverables in the task.",
            },
        )
        noul_res = self.noul(
            state=eval_state,
            proposition="The artifact complies with all structural and syntactic requirements and is ready for consolidation.",
        )

        combined_quality = round(0.6 * score_res.score + 0.4 * noul_res.probability, 4)
        passed = combined_quality >= min_score and noul_res.probability >= (min_score - 0.05)

        if passed:
            action: Literal["accept", "retry", "abort"] = "accept"
            feedback = f"Passed System-1 Gatekeeper (score={score_res.score:.2f}, noul={noul_res.probability:.2f})."
        elif retry_count < max_retries:
            action = "retry"
            feedback = (
                f"Rejected by System-1 Gatekeeper (score={score_res.score:.2f} < {min_score:.2f}). "
                f"Issues: {score_res.rationale or 'insufficient completeness or syntax errors'}."
            )
        else:
            action = "abort"
            feedback = (
                f"Aborted after {retry_count} retries (score={score_res.score:.2f}, "
                f"rationale={score_res.rationale})."
            )

        return GatekeeperVerdict(
            passed=passed,
            score=score_res.score,
            compliance_probability=noul_res.probability,
            action=action,
            feedback=feedback,
        )
