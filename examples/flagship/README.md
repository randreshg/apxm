# APXM Flagship Demo — 3-Agent Collaborative Research

This demo showcases APXM's core differentiators vs Python agent frameworks:

| Feature | APXM | LangGraph | AutoGen | CrewAI |
|---------|------|-----------|---------|--------|
| Compiled binary (.apxmobj) | ✓ | ✗ | ✗ | ✗ |
| Static analysis before run | ✓ | ✗ | ✗ | ✗ |
| Compiler optimizer (fuse-ask-ops) | ✓ | ✗ | ✗ | ✗ |
| KPN formal parallelism | ✓ | ✗ | ✗ | ✗ |
| PAUSE/RESUME checkpoint | ✓ | ✗ | ✗ | ✗ |
| Execution metrics (p50/p99) | ✓ | ✗ | ✗ | ✗ |

## Architecture

```
Coordinator Agent (port 18800)
    │
    ├─ COMMUNICATE → Researcher Agent (port 18801)  ─┐ parallel
    └─ COMMUNICATE → Critic Agent    (port 18802)  ─┘ (KPN dataflow)
    │
    ├─ PAUSE → human review checkpoint
    │
    └─ REASON → final synthesized report
```

## Quick Start

```bash
# 1. Build binaries
cargo build --release -p apxm-server -p apxm

# 2. Set LLM key (or use any OpenAI-compatible backend)
export OCP_APIM_KEY="your-key-here"

# 3. Run the demo
bash demo/run_demo.sh "XDNA 2 NPU: architecture, performance, and software stack"
```

## Files

```
demo/
├── agents/
│   ├── coordinator.ais   # orchestrator: plans, dispatches, pauses, synthesizes
│   ├── researcher.ais    # deep research + LTM persistence
│   └── critic.ais        # gap analysis + counterarguments
├── artifacts/            # compiled .apxmobj files (created by run_demo.sh)
├── run_demo.sh           # end-to-end demo script
└── README.md             # this file
```

## LLM Backend

All agents use `backend: "apxm"` which maps to any On-Premises LLM API.
Configure in `apxm.toml`:

```toml
[[llm_backends]]
name = "apxm"
provider = "openai"
model = "your-model-name"
api_key = "dummy"
base_url = "env:APXM_LLM_BASE_URL"  # set APXM_LLM_BASE_URL to your LLM gateway URL

[llm_backends.extra_headers]
Ocp-Apim-Subscription-Key = "env:OCP_APIM_KEY"
user = "env:USERNAME"
```

## Expected Output

The demo script walks through 8 steps with colored terminal output:

1. **Compile** — 3 `.apxmobj` files generated and sized
2. **Diagnostics** — compiler passes, fused ops, speedup estimate
3. **Servers** — 3 apxm-server instances start (health-checked)
4. **Registry** — researcher + critic registered with coordinator
5. **Execute** — coordinator starts, researcher+critic run in parallel
6. **Stream** — SSE events show node-by-node progress, PAUSE fires
7. **Human review** — checkpoint fetched, human notes injected, execution resumes
8. **Metrics** — parallelism_factor, llm_calls, p50/p99 latency displayed
