# APXM – Agent Programming eXecution Model

APXM is a full toolchain for building autonomous agents. It combines:

- **A high-level DSL** (AIS) for declaring memory, flows, handlers, and tool invocations.
- **A compiler** that lowers AIS → MLIR → executable artifacts.
- **A runtime/linker** that wires artifacts to capabilities, an LLM registry, and execution memory.
- **A driver + CLI** for compile/run workflows.

The project is inspired by the “Agent Programming eXecution Model” paper: agents maintain belief/goal structures, compile their plans into deterministic flows, and execute them under a scheduler with long‑term/short‑term memory.

---

## Docs Map

- `docs/AGENTS.md`: **Python CLI reference** (recommended)
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
cargo run -p apxm-cli -- install

# Install Python CLI dependencies
pip install typer rich
```

### Using the Python CLI (Recommended)

The Python CLI automatically handles environment setup and MLIR configuration:

```bash
# Check environment status
python tools/apxm_cli.py doctor

# Build the compiler (auto-sets MLIR environment)
python tools/apxm_cli.py compiler build

# Run an AIS file
python tools/apxm_cli.py compiler run examples/hello_world.ais

# Check all benchmark workloads
python tools/apxm_cli.py workloads check
```

See `docs/AGENTS.md` for the complete CLI reference.

### Prerequisites

- **mamba** or **conda** (install [miniforge](https://github.com/conda-forge/miniforge) recommended)
- **git**
- **Rust nightly** (managed via `rust-toolchain.toml`)
- **Python 3.10+** with `typer` and `rich`

### Development Commands

Use the Makefile:

```bash
make build      # Build project
make test       # Run tests
make help       # Show all targets
```

APXM keeps workspace state under `.apxm/` (artifacts, sessions, logs).

---

## Configuration

Runtime and driver configuration live in `crates/apxm-driver`. A minimal example TOML:

```toml
[chat]
providers = ["ollama"]
default_model = "ollama"
planning_model = "ollama"

[[llm_backends]]
name = "ollama"
provider = "ollama"
model = "gpt-oss:120b-cloud" # or: "gpt-oss:20b-cloud"
endpoint = "http://localhost:11434"
```

Key sections:

- `[chat]`: default/planning models, session storage path, system prompts.
- `[[llm_backends]]`: register providers (OpenAI, Anthropic, Google, Ollama, …). Each entry may specify `model`, `endpoint`, `api_key` (string or `env:VAR`), and arbitrary `options`.
- `capabilities`, `tools`, `execpolicy`: declare available external actions and sandboxing policies.

The driver instantiates each backend, registers them with the LLM registry, and sets defaults. Ollama endpoints skip API-key validation; cloud providers must provide `api_key`.

**MVP note**: for now we assume a single default backend/model. The runtime can register multiple backends, but the DSL does not yet expose per-operation model selection.

See `docs/getting_started.md` for ready-to-copy examples (Ollama cloud, OpenAI, Gemini).

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

### Python CLI (Recommended)

```bash
# Environment check
python tools/apxm_cli.py doctor

# Build compiler
python tools/apxm_cli.py compiler build

# Run AIS file
python tools/apxm_cli.py compiler run examples/hello_world.ais

# Compile only
python tools/apxm_cli.py compiler compile file.ais -o output.apxmobj

# Workload validation
python tools/apxm_cli.py workloads list
python tools/apxm_cli.py workloads check
```

### Manual Commands (Alternative)

For advanced users who prefer manual environment management:

```bash
# Activate environment first
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"

# Build and use the compiled binary
cargo build -p apxm-cli --features driver --release
./target/release/apxm run examples/hello_world.ais
./target/release/apxm compile examples/hello_world.ais -o output.apxmobj

# Runtime examples (no compiler required)
cargo run -p apxm-runtime --example substrate_demo
cargo run -p apxm-runtime --example ollama_llm_demo

# Install/update conda environment
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
