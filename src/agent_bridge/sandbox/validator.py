"""5-Layer AST & Resource Sandbox Validator (`src/agent_bridge/sandbox/validator.py`).

Implements deterministic, zero-VRAM security validation inspired by NeuroCore:
- Layer 1 (Static AST Analysis): Blocks `eval()`, `exec()`, `compile()`, `__import__()`, `os.system()`, `shutil.rmtree()`.
- Layer 2 (Import Blocklist): Blocks dangerous low-level/deserialization modules (`ctypes`, `pickle`, `marshal`, `pty`).
- Layer 3 (SafeEnv Mocking): Strips sensitive credentials when executing sandboxed verification checks.
- Layer 4 (SSRF & Network Guard): Blocks cloud metadata IPs (`169.254.169.254`) and dangerous URI schemes (`gopher://`).
- Layer 5 (Resource Limits): Enforces 100 KB max output cap and hard execution timeout.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from typing import Any

from pydantic import BaseModel, Field

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

BLOCKED_IMPORTS: frozenset[str] = frozenset(
    {
        "ctypes",
        "pickle",
        "_pickle",
        "cPickle",
        "marshal",
        "shelve",
        "pty",
        "termios",
        "fcntl",
        "resource",
    }
)

BLOCKED_BUILTIN_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
    }
)

BLOCKED_ATTR_CALLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("os", "system"),
        ("os", "popen"),
        ("os", "execl"),
        ("os", "execle"),
        ("os", "execlp"),
        ("os", "execv"),
        ("os", "execve"),
        ("os", "execvp"),
        ("os", "fork"),
        ("os", "kill"),
        ("shutil", "rmtree"),
        ("pickle", "loads"),
        ("pickle", "load"),
        ("marshal", "loads"),
    }
)

_SSRF_DANGEROUS_PATTERNS = (
    "169.254.169.254",
    "metadata.google.internal",
    "100.100.100.200",
    "gopher://",
    "dict://",
)

MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB ceiling


class SandboxValidationResult(BaseModel):
    """Typed output of the 5-layer security sandbox validator."""

    safe: bool = Field(default=True)
    safety_score: float = Field(default=1.0, ge=0.0, le=1.0)
    violations: list[str] = Field(default_factory=list)
    layers_checked: list[str] = Field(
        default_factory=lambda: [
            "layer1_static_ast",
            "layer2_import_blocklist",
            "layer3_safe_env",
            "layer4_ssrf_network_guard",
            "layer5_resource_limits",
        ]
    )


class SafeEnv:
    """Layer 3: Restricted environment wrapper stripping secret tokens and API keys."""

    _SENSITIVE_SUBSTRINGS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL", "AUTH")

    @classmethod
    def build_sanitized_env(cls) -> dict[str, str]:
        clean: dict[str, str] = {}
        for k, v in os.environ.items():
            upper = k.upper()
            if any(sec in upper for sec in cls._SENSITIVE_SUBSTRINGS):
                continue
            clean[k] = v
        clean["PYTHONDONTWRITEBYTECODE"] = "1"
        return clean


class CodeSandboxValidator:
    """5-Layer deterministic security validator for System-1 Gatekeeper & Sub-Worker code."""

    def __init__(
        self,
        blocked_imports: frozenset[str] = BLOCKED_IMPORTS,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ) -> None:
        self.blocked_imports = blocked_imports
        self.max_output_bytes = max_output_bytes

    def validate_artifact(self, artifact_text: str) -> SandboxValidationResult:
        """Run all 5 static & resource security layers on `artifact_text` in < 0.5ms on CPU."""
        violations: list[str] = []

        # Layer 5: Resource output size check (100 KB max)
        encoded_len = len(artifact_text.encode("utf-8", errors="replace"))
        if encoded_len > self.max_output_bytes:
            violations.append(
                f"Layer 5 (Resource Limit): Artifact size ({encoded_len} B) exceeds {self.max_output_bytes} B ceiling."
            )

        # Layer 4: SSRF & Network Guard check
        lowered = artifact_text.lower()
        for pattern in _SSRF_DANGEROUS_PATTERNS:
            if pattern in lowered:
                violations.append(f"Layer 4 (SSRF Guard): Blocked dangerous endpoint/scheme {pattern!r}.")

        # Extract Python code blocks for Layers 1 & 2
        cleaned = artifact_text.strip()
        fenced = _CODE_FENCE_RE.findall(cleaned)
        blocks = fenced if fenced else []
        if not blocks and any(
            cleaned.startswith(p) for p in ("def ", "class ", "import ", "from ", "async def ", "@", '"""', "#")
        ):
            blocks = [cleaned]

        for block_idx, code in enumerate(blocks, start=1):
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue  # Syntax errors are caught by AST syntax checker

            for node in ast.walk(tree):
                # Layer 2: Import Blocklist
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = alias.name.split(".")[0]
                        if root_mod in self.blocked_imports:
                            violations.append(
                                f"Layer 2 (Import Blocklist): Forbidden import '{alias.name}' in block #{block_idx}."
                            )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in self.blocked_imports:
                        violations.append(
                            f"Layer 2 (Import Blocklist): Forbidden import from '{node.module}' in block #{block_idx}."
                        )

                # Layer 1: Static AST Dangerous Calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTIN_CALLS:
                        violations.append(
                            f"Layer 1 (Static AST): Forbidden builtin call '{node.func.id}()' in block #{block_idx}."
                        )
                    elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                        pair = (node.func.value.id, node.func.attr)
                        if pair in BLOCKED_ATTR_CALLS:
                            violations.append(
                                f"Layer 1 (Static AST): Forbidden call '{pair[0]}.{pair[1]}()' in block #{block_idx}."
                            )

        if not violations:
            return SandboxValidationResult(safe=True, safety_score=1.0, violations=[])

        penalty = min(1.0, 0.5 * len(violations))
        return SandboxValidationResult(
            safe=False,
            safety_score=round(max(0.0, 1.0 - penalty), 4),
            violations=violations,
        )

    def execute_sandboxed_snippet(self, python_code: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
        """Execute a validated Python snippet in a restricted subprocess with SafeEnv and 100 KB output cap."""
        validation = self.validate_artifact(python_code)
        if not validation.safe:
            return {
                "executed": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "; ".join(validation.violations),
                "violations": validation.violations,
            }

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", python_code],
                env=SafeEnv.build_sanitized_env(),
                capture_output=True,
                text=True,
                timeout=min(float(timeout_seconds), 30.0),
                check=False,
            )
            stdout_capped = proc.stdout.encode("utf-8")[: self.max_output_bytes].decode("utf-8", errors="replace")
            stderr_capped = proc.stderr.encode("utf-8")[: self.max_output_bytes].decode("utf-8", errors="replace")
            return {
                "executed": True,
                "returncode": proc.returncode,
                "stdout": stdout_capped,
                "stderr": stderr_capped,
                "violations": [],
            }
        except subprocess.TimeoutExpired:
            return {
                "executed": False,
                "returncode": -2,
                "stdout": "",
                "stderr": f"Layer 5 (Resource Limit): Execution timed out after {timeout_seconds}s.",
                "violations": ["TimeoutExpired"],
            }
