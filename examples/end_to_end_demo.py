"""End-to-End Executable Demonstration of Hybrid Agent Bridge (Phase 4).

Demonstrates the complete 4-tier memory and hierarchical LangGraph workflow:
1. Initializes the unified SQLite WAL database (`memory.db`) with FTS5 and CPU vectors.
2. Seeds Declarative Memory facts (demonstrating deterministic Upsert: latest-is-truth)
   and verifies Repo Truth Hierarchy (> Memory).
3. Connects to a live local Ollama / llama.cpp / vLLM `/v1/chat/completions` endpoint
   if reachable on `localhost:11434`, or launches a lightweight local OpenAI-compatible
   HTTP server on `127.0.0.1` so the full HTTP wire protocol executes deterministically.
4. Executes Task 1 (including a simulated System-1 Gatekeeper rejection on an initial
   draft containing a `TODO` placeholder, followed by automatic retry and consolidation).
5. Executes Task 2, retrieving the consolidated episode from Task 1 via hybrid FTS5 BM25 +
   CPU vector search under the strict < 800 token Context Injector budget.
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent_bridge.config import BridgeConfig
from agent_bridge.memory.declarative import DeclarativeMemoryStore
from agent_bridge.memory.engine import LocalMemoryEngine
from agent_bridge.memory.episodic import EpisodicMemoryStore
from agent_bridge.orchestrator.graph import HybridAgentBridgeRunner


class _LocalOpenAICompatHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible `/v1/chat/completions` server simulating a local quantized model."""

    call_counter = 0

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return  # Keep demo stdout clean

    def do_POST(self) -> None:  # noqa: N802
        content_len = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_len).decode("utf-8")
        payload = json.loads(raw_body)
        messages = payload.get("messages", [])
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        _LocalOpenAICompatHandler.call_counter += 1
        call_num = _LocalOpenAICompatHandler.call_counter

        # On the very first call, simulate a typical quantized local LLM mistake (TODO stub)
        # so we can observe System-1 Noul/Gatekeeper catching and repairing it deterministically.
        if call_num == 1 and "SYSTEM-1 CRITIC" not in last_user and "PREVIOUS GATEKEEPER FEEDBACK" not in last_user:
            reply_code = (
                "```python\n"
                "def calculate_token_budget(text: str, limit: int = 800) -> dict:\n"
                "    # TODO: implement token budget calculation\n"
                "    pass\n"
                "```"
            )
        else:
            reply_code = (
                "```python\n"
                "from __future__ import annotations\n\n"
                "import math\n\n\n"
                "def calculate_token_budget(text: str, limit: int = 800) -> dict[str, int | bool]:\n"
                '    """Compute conservative token estimate and enforce < 800 token budget."""\n'
                "    tokens = max(math.ceil(len(text) / 4), math.ceil(len(text.split()) * 1.35))\n"
                "    return {\n"
                '        "estimated_tokens": tokens,\n'
                '        "limit": limit,\n'
                '        "within_budget": tokens < limit,\n'
                "    }\n"
                "```"
            )

        response_body = json.dumps(
            {
                "id": f"chatcmpl-local-{call_num}",
                "object": "chat.completion",
                "model": payload.get("model", "qwen2.5-coder:7b"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply_code},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def _is_ollama_running(host: str = "127.0.0.1", port: int = 11434) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def run_demo() -> None:
    print("=" * 78)
    print("HYBRID AGENT BRIDGE - PHASE 4 END-TO-END WORKFLOW DEMO")
    print("=" * 78)

    with tempfile.TemporaryDirectory(prefix="agent_bridge_demo_") as tmp_dir:
        db_path = Path(tmp_dir) / "memory.db"
        repo_root = Path(__file__).resolve().parent.parent

        config = BridgeConfig(db_path=db_path, repo_root=repo_root)
        server: HTTPServer | None = None

        if _is_ollama_running():
            print(f"[Inference] Detected live local inference server at {config.llm.base_url}")
        else:
            server = HTTPServer(("127.0.0.1", 0), _LocalOpenAICompatHandler)
            port = server.server_port
            threading.Thread(target=server.serve_forever, daemon=True).start()
            config.llm.base_url = f"http://127.0.0.1:{port}/v1"
            print(f"[Inference] Started local OpenAI-compatible endpoint at {config.llm.base_url}")

        try:
            # 1. Initialize Unified SQLite Engine (WAL + FTS5 + Vectors)
            engine = LocalMemoryEngine(config=config)
            init_meta = engine.initialize()
            print(f"\n[Phase 1] Initialized SQLite Engine: {json.dumps(init_meta)}")

            # 2. Declarative Memory Upsert (Latest-is-Truth)
            decl = DeclarativeMemoryStore(engine)
            decl.upsert_fact("runtime", "target_hardware", "Apple Silicon M-Series 16GB Unified RAM")
            decl.upsert_fact("constraints", "max_context_tokens", 512)
            # Immediate deterministic overwrite without LLM conflict resolution
            updated_fact = decl.upsert_fact("constraints", "max_context_tokens", 800)
            print(
                f"[Phase 1] Deterministic Upsert ('constraints', 'max_context_tokens') -> {updated_fact.value} "
                f"(updated_at={updated_fact.updated_at})"
            )

            # Seed a stale file fact for a non-existent file to verify Repo Truth Hierarchy override
            decl.upsert_fact("repo_file", "src/agent_bridge/non_existent_legacy.py", "Exists in old memory")

            # 3. Execute Task 1 through the 4-Node LangGraph Bridge
            runner = HybridAgentBridgeRunner(config=config, engine=engine, allow_offline_synthesis=False)
            task_1 = (
                "Write a Python function `calculate_token_budget` aligned with pyproject.toml and "
                "src/agent_bridge/non_existent_legacy.py that estimates token budget under limit=800."
            )
            print(f"\n[Phase 3] Executing Task 1 on LangGraph Macro-FSM:\n  Task: {task_1}")
            state_1 = runner.run(task=task_1, thread_id="demo-thread-1", tags=["token-budget", "core"])

            print(f"  -> Final FSM Status       : {state_1['status']}")
            print(f"  -> Context Budget Used    : {state_1['context_tokens']} / 800 tokens")
            print(f"  -> Repo Truth Entries     : {state_1['repo_truth_entries']}")
            print(f"  -> SubWorker Rounds Used  : {state_1['subworker_rounds_used']} (max_round <= 3)")
            print(f"  -> Scratchpad Destroyed   : {state_1['scratchpad_destroyed']}")
            print(
                f"  -> System-1 Gatekeeper    : passed={state_1['gatekeeper_passed']} "
                f"(Score={state_1['gatekeeper_score']}, Noul={state_1['gatekeeper_noul']})"
            )
            print(f"  -> Consolidated Episode # : {state_1['consolidated_episode_id']}")

            # 4. Execute Task 2 verifying hybrid FTS5 BM25 + CPU Vector retrieval
            episodic = EpisodicMemoryStore(engine)
            hits = episodic.retrieve_hybrid("calculate_token_budget token limit 800", top_k=2)
            print(
                f"\n[Phase 4] Hybrid Retrieval (FTS5 BM25 + CPU Vector) found {len(hits)} episode(s): "
                f"top_id={hits[0].id}, hybrid_score={hits[0].hybrid_score}, "
                f"bm25={hits[0].bm25_score}, vec={hits[0].vector_score}"
            )

            print("\n[Validated Artifact Persisted in SQLite]:")
            print(state_1["final_artifact"])
            print("=" * 78)
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    run_demo()
