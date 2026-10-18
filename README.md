# CoreNGN (hybrid-memory-engine)

**Zero-Container Local-First 4-Tier Memory & System-1 / System-2 Hierarchical LangGraph Orchestration Framework.**

Designed for real-time reliability on consumer/prosumer hardware running local quantized inference servers (**Ollama**, **llama.cpp**, **vLLM**) under strict VRAM and KV-cache constraints.

---

## Architectural Pillars

1. **Zero-Container Local-First Persistence (`LocalMemoryEngine`)**
   - Unifies all 4 memory tiers into a single embedded SQLite file (`~/.agent_bridge/memory.db`) running in **WAL (Write-Ahead Logging)** mode (`synchronous=NORMAL`, `busy_timeout=5000ms`).
   - **Git-Branch Scoped Persistence**: Automatically resolves the active Git repository and branch (`git rev-parse --abbrev-ref HEAD`), isolating unmerged feature-branch memories so they never pollute `main`/`master`.
   - **Working Memory**: Atomic graph checkpoints via LangGraph's `SqliteSaver`.
   - **Declarative Memory**: `declarative_facts` scoped by `(repo_identifier, git_branch, category, key)` with deterministic **Upsert** semantics (*latest write is ground truth*) and immutability pinning (`pinned=1`).
   - **Semantic & Episodic Memory**: `episodic_artifacts` indexed via **SQLite FTS5 (BM25)** external-content triggers and CPU-embedded `float32` vectors (`sqlite-vec` + deterministic SQLite cosine similarity UDF). Stores **only** validated `(Task, Final_Artifact)` pairs.
   - **Bounded Context Injector**: Guarantees a hard context budget (`< 800 tokens`) and enforces **Repo Truth Hierarchy > Declarative Memory > Episodic Memory**.

2. **System-1 / System-2 Dualism (`JevDecisionAdapter`)**
   - Generative LLMs (*System 2*) are reserved **exclusively** for artifact synthesis.
   - Routing, memory ingestion classification, compliance validation, and sub-worker stopping conditions are delegated to a typed *System 1* controller implementing:
     - `Choice(state, question, options)` $\rightarrow$ chosen option + normalized Softmax distribution.
     - `Score(state, question, scale_rubric)` $\rightarrow$ normalized scalar in $[0.0, 1.0]$ with AST & anti-placeholder rubric breakdown.
     - `Noul(state, proposition)` $\rightarrow$ epistemic truth probability in $[0.0, 1.0]$.

3. **Deterministic Sub-Worker Isolation (`IsolatedSubWorker`)**
   - Encapsulates exploratory multi-turn synthesis inside `SubWorker_Node` with:
     - Zero shared state with the parent orchestrator.
     - Strict round cap (`max_round <= 3`).
     - Mandatory `120s` wall-clock timeout inside an isolated `asyncio` event loop.
     - Immediate destruction of the internal scratchpad (`scratchpad.clear()`) before returning only the converging artifact.

4. **Anti-Memory-Rot Curation, Zero-Bloat TUI & On-Device Benchmark**
   - **Memory Curation (`MemoryCurator`)**: `pin`, `unpin`, `forget`, `prune` (with `--dry-run`), and `explain` (transparent BM25 + CPU Vector + Recency + Branch score decomposition).
   - **Zero-Bloat Terminal UI (`AgentBridgeTUI`)**: Full-screen `textual` and `rich` terminal dashboard with 3 tabs (*Memory Explorer*, *Live Graph Tracer*, and *Workspace & Repo Truth Diff*).
   - **Universal Runtime Auto-Discovery (`RuntimeDetector`)**: Non-blocking probing of Ollama (`11434`), llama.cpp (`8080`), and vLLM/LM Studio (`1234`).
   - **On-Device Benchmark (`BenchmarkRunner`)**: Profiles System-1 vs System-2 latency, TTFT, tokens/sec, and RSS RAM / SQLite disk growth over 100 synthetic runs with 0.0 MB VRAM overhead.

---

## Macro-FSM Topology

```mermaid
flowchart LR
    START((START)) --> CL["1. Context_Loader<br/>(Repo Truth + <800 tok)"]
    CL --> SW["2. SubWorker_Node<br/>(<=3 rounds, 120s timeout)"]
    SW --> GK{"3. System1_Gatekeeper<br/>(Score & Noul)"}
    GK -- "accept" --> EC["4. Episodic_Consolidator<br/>(SQLite FTS5 + Vector)"]
    GK -- "retry (<= max_retries)" --> SW
    GK -- "abort" --> END((END))
    EC --> END
```

---

## Quick Start (Local-First)

### 1. Installation

```bash
uv venv --python python3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev,all]"
```

### 2. Initialize & Run Pre-Flight Diagnostics

```bash
agentic init --seed-defaults
agentic doctor
```

### 3. Memory Curation, Retrieval Explanation & Execution

```bash
# Upsert and pin an immutable architectural constraint on the active Git branch
agentic memory upsert runtime inference_backend "ollama/qwen2.5-coder:7b" --pin
agentic pin architecture context_budget_tokens

# Execute the 4-node LangGraph FSM
agentic run "Write a Python function calculate_token_budget that estimates token usage" \
  --fact "constraints:max_tokens=800"

# Explain exact BM25 + CPU Vector + Recency scoring and < 800 token budget decisions
agentic explain "calculate_token_budget"

# Prune outdated or low-salience memories (protecting pinned rules)
agentic prune --older-than 30 --min-score 0.50 --dry-run

# Launch the Zero-Bloat Terminal UI (or render a Rich snapshot)
agentic tui
agentic tui --snapshot --query "calculate_token_budget"

# Execute the 100-run on-device latency & footprint benchmark
agentic benchmark --iterations 100 --format both
```

### 4. Run the Deterministic Test Suite

```bash
pytest -v
```

## Core Capabilities & Architectural Advantages

### 1. Deterministic Auto-Repair & Sandboxed Sub-Workers
* **Isolated Multi-Agent Sandbox:** Encapsulates exploratory multi-agent collaboration (such as Coder and Critic workflows) within isolated sub-worker nodes[cite: 1, 2].
* **Execution Guardrails & Circuit Breakers:** Restricts conversational loops with strict turn limits (`max_round`) and execution timeouts to eliminate token explosion and prevent orchestrator blocking[cite: 1, 2].
* **Clean State Hand-Off:** Ensures the sub-worker extracts and returns only the final converged, validated artifact to the primary state machine[cite: 1, 2].
* **Immediate Scratchpad Eviction:** Completely flushes and discards the internal conversational buffer upon node completion, preventing intermediate noise from causing context stuffing[cite: 1, 2].

### 2. Multi-Tier Memory Persistence & Zero Semantic Drift
* **Decoupled 4-Tier Memory:** Organizes state across Working, Declarative, Semantic, and Episodic persistence layers to eliminate monolithic prompt bloating[cite: 1, 3].
* **Clean Episodic Storage:** Persists exclusively converged problem-solution pairs while aggressively pruning intermediate trial-and-error traces and failed attempts[cite: 1, 3].
* **Mitigation of the "Telephone Game":** Prevents recursive summarization loss and semantic drift by avoiding continuous lossy compression of active chat dialogues[cite: 3].
* **Ingestion Garbage Collection:** Automatically strips raw tool invocations, syntax errors, and internal reasoning traces before records reach long-term storage[cite: 1, 3, 5].

### 3. Absolute Truth Hierarchy (`Repo State > Memory`)
* **Background Context Demotion:** Treats retrieved memories strictly as background context rather than rigid, overriding instructions[cite: 1, 5].
* **Disk Primacy:** Enforces that the live state of code files on disk within the repository always takes precedence over historical memory whenever conflicts arise[cite: 1, 5].
* **Hallucination Suppression:** Eliminates phantom memories and outdated architectural assumptions when codebase implementations evolve across sessions[cite: 1, 5].

### 4. Sub-Millisecond System-1 Decision Controller
* **Deterministic Pre-Flight Checks:** Executes syntax validation via AST parsing and typed heuristic evaluations before dispatching generative LLM queries.
* **Instantaneous Gating:** Resolves state routing and stopping criteria in sub-millisecond cycles on CPU, delivering significant speedups over generative round-trips.
* **GPU Compute Reservation:** Leaves GPU cores and VRAM unburdened by routine classification tasks, reserving compute strictly for generative code synthesis.

### 5. Resource Efficiency & Zero-VRAM Footprint
* **Zero-VRAM Persistence:** Runs persistence operations, lexical search (BM25), and CPU embeddings entirely in-process without allocating GPU VRAM[cite: 1].
* **KV-Cache Optimization:** Enforces strict context budget ceilings (< 800 tokens), preventing context window saturation and attention degradation[cite: 1, 3].
* **Container-Free Deployment:** Eliminates mandatory external server stacks (e.g., PostgreSQL, Redis, standalone vector DBs) in local-first environments in favor of embedded storage[cite: 1].

### 6. Git-Awareness & Branch-Scoped Isolation
* **Branch-Context Tracking:** Automatically associates declarative rules and episodic solutions with the active Git branch and repository identifier.
* **Cross-Branch Quarantine:** Isolates unmerged feature-branch memories from contaminating the main trunk branch.
* **Asymmetric Inheritance:** Permits experimental feature branches to inherit baseline trunk conventions while preventing unverified assumptions from polluting global state.

### 7. Explicit Memory Curation & Human-in-the-Loop Control
* **Native Checkpointing & Time Travel:** Leverages state checkpointing to allow pausing execution, inspecting states, and rolling back steps retroactively[cite: 1, 2, 3].
* **Immutable Pinning:** Allows developers to pin critical architectural rules to guarantee they cannot be overwritten by automatic compaction or LLM updates[cite: 1].
* **Deterministic Deletion:** Provides explicit deletion mechanisms to atomically purge obsolete memories across relational and search indices[cite: 1, 5].
* **Retrieval Explainability:** Exposes exact score breakdowns detailing lexical BM25 matching, vector cosine similarity, and recency decay contributions.

### 8. Flexible Deployment & OMP Ecosystem Interoperability
* **Standalone Local Operation:** Runs self-contained on a single developer machine with zero network dependencies[cite: 1].
* **Asymmetric Exobrain Setup:** Operates as a centralized memory daemon serving remote execution environments and coding agents over lightweight HTTP APIs[cite: 1, 5].
* **OMP (Oh My Pi) Integration:** Integrates with terminal workflows by mirroring OMP native memory principles (`mnemopi`, `local`, and `hindsight`)[cite: 1, 5].
