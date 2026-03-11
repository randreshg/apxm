# APXM – Agent Programming eXecution Model

APXM is a full toolchain for building autonomous agents:

- **ApxmGraph IR** as the canonical frontend format
- **AIS DSL frontend** support via AST normalization to `ApxmGraph`
- **Compiler** that lowers graph → AIS MLIR → executable artifacts
- **Runtime** with scheduler, memory system, and LLM registry
- **CLI** for compile/run workflows

---

## Quick Start

**New installation:**
```bash
git clone --recursive https://github.com/randreshg/apxm
cd apxm
pip install -e external/sniff
python3 tools/apxm_cli.py install
source ~/.bashrc  # or ~/.zshrc - restart shell
apxm doctor
```

**Already cloned? Just add the submodule:**
```bash
cd apxm
git submodule update --init --recursive
pip install -e external/sniff
python3 tools/apxm_cli.py install
source ~/.bashrc  # or ~/.zshrc - restart shell
apxm doctor
```

**What the installer does:**
- ✓ Detects platform and package manager
- ✓ Creates conda environment with MLIR/LLVM 21
- ✓ Installs Rust nightly if needed
- ✓ Builds the APXM binary
- ✓ Installs `apxm` to `bin/` and updates shell PATH

See [docs/getting-started.md](docs/getting-started.md) for detailed instructions and troubleshooting.

## Prerequisites

- **Python** 3.10+ (for the CLI)
- **mamba** or **conda** ([miniforge](https://github.com/conda-forge/miniforge) recommended)
- **Git**

Optional (installer can set these up automatically):
- Rust nightly
- CMake >= 3.20

Run `apxm doctor` to check your environment automatically.

---

## CLI Commands

```bash
apxm doctor                     # Check environment (powered by sniff)
apxm install                    # Install/update conda environment
apxm build                      # Build full project
apxm build --compiler           # Build compiler only
apxm build --runtime            # Build runtime only
apxm build --no-trace           # Build without tracing (zero overhead)
apxm execute <file.json>         # Compile and run an ApxmGraph file
apxm execute <file.json> --trace debug  # Run with debug tracing
apxm compile <file.json> -o out.apxmobj # Compile to artifact
apxm run <file.apxmobj>         # Run pre-compiled artifact
apxm workloads list             # List benchmark workloads
apxm workloads check            # Validate workloads compile
apxm workloads run <name>       # Run a specific workload
apxm benchmarks run --workloads # Run all benchmark workloads
```

See [docs/AGENTS.md](docs/AGENTS.md) for complete reference.

---

## Runtime Tracing

APXM includes a two-tier tracing system for debugging and performance analysis:

**Build-time control:**
```bash
apxm build              # Default: tracing compiled in
apxm build --no-trace   # Zero-overhead build (tracing compiled out)
```

**Runtime control** (when built with tracing):
```bash
apxm execute workflow.json                  # Silent execution
apxm execute workflow.json --trace info     # High-level execution flow
apxm execute workflow.json --trace debug    # Detailed worker/operation info
apxm execute workflow.json --trace trace    # Full verbosity (tokens, LLM calls)
```

Trace targets: `apxm::scheduler`, `apxm::ops`, `apxm::llm`, `apxm::tokens`, `apxm::dag`

---

## LLM Operations

APXM provides three core LLM operations with different reasoning characteristics:

| Operation | Purpose | Example |
|-----------|---------|---------|
| `ask` | Simple Q&A with LLM | `ask("What is 2+2?") -> answer` |
| `think` | Extended thinking with token budget | `think("Analyze this problem", budget: 1000) -> analysis` |
| `reason` | Structured reasoning with belief updates | `reason("Solve step by step", context) -> solution` |

```json
{
  "name": "llm_ops",
  "nodes": [
    { "id": 1, "name": "ask", "op": "ASK", "attributes": { "template_str": "Explain the domain background of {0}" } },
    { "id": 2, "name": "think", "op": "THINK", "attributes": { "template_str": "Analyze the implications", "budget": 2000 } },
    { "id": 3, "name": "reason", "op": "REASON", "attributes": { "template_str": "Execute step 1: {0}" } }
  ],
  "edges": [
    { "from": 1, "to": 2, "dependency": "Data" },
    { "from": 2, "to": 3, "dependency": "Data" }
  ],
  "parameters": [{ "name": "topic", "type_name": "str" }],
  "metadata": { "is_entry": true }
}
```

Context operands (comma-separated) are appended at runtime as:

```
<template>

Context 1: <value>
Context 2: <value>
```

The `+` operator merges tokens; use the comma form for context operands.

---

## IR Debugging

```bash
# Write per-pass MLIR snapshots to the given directory.
APXM_PRINT_IR_DIR=/tmp/apxm-ir apxm compiler compile file.json -o output.apxmobj

# Optional: print a one-line trace of IR printing config.
APXM_PRINT_IR_TRACE=1 APXM_PRINT_IR_DIR=/tmp/apxm-ir apxm compiler compile file.json -o output.apxmobj
```

---

## Configuration

Create `.apxm/config.toml`:

```toml
[chat]
providers = ["ollama"]
default_model = "ollama"

[[llm_backends]]
name = "ollama"
provider = "ollama"
model = "gpt-oss:20b-cloud"
endpoint = "http://localhost:11434"
```

---

## Project Layout

```text
crates/
  apxm-ais        # AIS operation definitions
  apxm-cli        # CLI tool
  apxm-compiler   # MLIR compiler
  apxm-driver     # Compiler+runtime orchestration
  runtime/
    apxm-artifact # Artifact format
    apxm-backends # LLM and storage backends
    apxm-core     # Shared types
    apxm-runtime  # Execution engine
examples/         # Sample ApxmGraph programs
docs/             # Documentation
```

---

## Environment Diagnostics

The `apxm doctor` command uses [sniff](https://github.com/randres/sniff) for comprehensive environment detection:

```bash
apxm doctor
```

**What it checks:**
- **Platform** -- OS, architecture, Linux distro, WSL, containers
- **Dependencies** -- Rust (nightly), Cargo, CMake, Ninja, Git, LLVM
- **Conda environment** -- `apxm` env activation, Python version, MLIR/LLVM 21.x
- **Build status** -- whether the compiler binary has been built
- **Credentials** -- registered LLM provider API keys
- **CI environment** -- GitHub Actions, GitLab CI, Jenkins, and other providers (auto-detected)

Each check provides actionable fix suggestions when issues are found.

---

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - Installation, setup, and first steps
- [AGENTS.md](docs/AGENTS.md) - CLI reference and AI agent guide
- [Benchmark Workloads](papers/CF26/benchmarks/workloads/README.md) - workload benchmarks and examples
