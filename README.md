# APXM – Agent Programming eXecution Model

APXM is a full toolchain for building autonomous agents. It combines:

- **A high-level DSL** (AIS) for declaring memory, flows, handlers, and tool invocations.
- **A compiler** that lowers AIS → MLIR → executable artifacts.
- **A runtime/linker** that wires artifacts to capabilities, an LLM registry, and execution memory.
- **A driver + CLI** for compile/run workflows.

The project is inspired by the “Agent Programming eXecution Model” paper: agents maintain belief/goal structures, compile their plans into deterministic flows, and execute them under a scheduler with long‑term/short‑term memory.

---

## Docs Map

- `docs/getting_started.md`: setup and toolchain
- `docs/hello_ais.md`: minimal end-to-end example
- `docs/architecture.md`: system architecture
- `docs/contract.md`: compiler/runtime contract
- `docs/diagrams.md`: flow diagrams
- `docs/glossary.md`: terminology
- `docs/testing.md`: test matrix

## Getting Started

### Quick Setup (Recommended)

```bash
git clone https://github.com/randreshg/apxm
cd apxm

# Create the conda environment (installs MLIR/LLVM toolchain)
conda env create -f environment.yaml
conda activate apxm
source scripts/apxm-activate.sh

# Build the Rust workspace (compiler excluded by default)
cargo build --workspace --exclude apxm-compiler
```

### Prerequisites

- **mamba** or **conda** (install [miniforge](https://github.com/conda-forge/miniforge) recommended)
- **git**
- **Rust nightly** (managed via `rust-toolchain.toml`)

### Development Commands

Use the Makefile:

```bash
make build      # Build project
make test       # Run tests
make help       # Show all targets
```

APXM keeps workspace state under `.apxm/` (artifacts, sessions, logs). A minimal CLI is available for compile/run workflows.

---

## Configuration

Runtime and driver configuration live in `crates/apxm-driver`. A minimal example TOML:

```toml
[chat]
providers = ["ollama", "gemini"]
default_model = "ollama"
planning_model = "ollama"

[[llm_backends]]
name = "ollama"
provider = "ollama"
model = "qwen3-coder:480b-cloud"
endpoint = "http://localhost:11434"

[[llm_backends]]
name = "gemini"
provider = "google"
model = "gemini-flash-latest"
api_key = "env:GEMINI_API_KEY"
```

Key sections:

- `[chat]`: default/planning models, session storage path, system prompts.
- `[[llm_backends]]`: register providers (OpenAI, Anthropic, Google, Ollama, …). Each entry may specify `model`, `endpoint`, `api_key` (string or `env:VAR`), and arbitrary `options`.
- `capabilities`, `tools`, `execpolicy`: declare available external actions and sandboxing policies.

The driver instantiates each backend, registers them with the LLM registry, and sets defaults. Ollama endpoints skip API-key validation; cloud providers must provide `api_key`.

---

## Project Layout

```
crates/
  apxm-ais        # AIS operation definitions + validation
  apxm-cli        # Minimal compile/run CLI
  apxm-compiler   # MLIR passes + codegen
  apxm-driver     # Compiler+runtime wiring + config
  runtime/
    apxm-artifact # Artifact serialization
    apxm-backends # Runtime subsystem: LLM and storage backends
    apxm-core     # Shared types + errors
    apxm-runtime  # Scheduler, memory, execution engine
examples/         # Sample AIS programs
.apxm/            # Generated artifacts, sessions, logs (gitignored)
```

## Frequently Used Commands

```bash
# Build runtime and driver
cargo build -p apxm-runtime -p apxm-driver

# Run runtime examples
cargo run -p apxm-runtime --example substrate_demo
cargo run -p apxm-runtime --example ollama_llm_demo

# Enable metrics (LLM tokens + timing) in runtime/driver
cargo test -p apxm-runtime --features metrics

# CLI (compile/run)
cargo run -p apxm-cli --features driver -- compile examples/hello_world.ais
cargo run -p apxm-cli --features driver -- run examples/hello_world.ais

# CLI diagnostics
cargo run -p apxm-cli -- doctor

# CLI env exports (after conda activate)
eval "$(cargo run -p apxm-cli -- activate)"

# Install/update env (mamba)
cargo run -p apxm-cli -- install
```

---

## Resources

- `docs/README.md`: documentation index
- `docs/getting_started.md`: step-by-step setup guide
- `docs/CRATES.md`: per-crate documentation index
- `docs/architecture.md`: system architecture
- `docs/architecture_summary.md`: architecture summary
- `docs/quick_reference.md`: quick reference
- `docs/diagrams.md`: system diagrams

Questions? Open an issue or ping the maintainers on the project chat.
