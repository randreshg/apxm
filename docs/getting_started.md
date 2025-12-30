# Getting Started

This guide walks you through installing dependencies, setting up the MLIR toolchain, and building APXM.

## 1) Install Prerequisites

- Rust toolchain (nightly is pinned via `rust-toolchain.toml`)
- `mamba` (recommended) or `conda` (for the MLIR toolchain)
- Python 3.10+ with `typer` and `rich`

## 2) Create the Conda Environment

```bash
cd /path/to/apxm
cargo run -p apxm-cli -- install
pip install typer rich
```

This uses `mamba` to create/update the `apxm` environment from `environment.yaml`.

## 3) Verify the Toolchain

```bash
python tools/apxm_cli.py doctor
```

You should see all checkmarks for APXM directory, Rust, Cargo, Conda environment, MLIR, and LLVM.

## 4) Build the Compiler

```bash
# Using Python CLI (recommended - auto-handles environment)
python tools/apxm_cli.py compiler build
```

## 5) Run Examples

```bash
# Using Python CLI (recommended)
python tools/apxm_cli.py compiler run examples/hello_world.ais

# Runtime examples (no compiler required)
cargo run -p apxm-runtime --example substrate_demo
cargo run -p apxm-runtime --example ollama_llm_demo
```

## 6) Check Workloads

```bash
# List available workloads
python tools/apxm_cli.py workloads list

# Verify all workloads compile
python tools/apxm_cli.py workloads check
```

---

## Alternative: Manual Environment Setup

For advanced users who prefer manual environment management:

### Activate the Environment

```bash
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"
```

### Build the Workspace

```bash
cargo build --workspace
```

### Run Examples

```bash
# Build the CLI with driver feature
cargo build -p apxm-cli --features driver --release

# Run an AIS file (requires activated environment)
./target/release/apxm run examples/hello_world.ais
```

**Note:** The Python CLI handles all of this automatically. Use it instead unless you have specific needs.

---

## 7) Running Tests

APXM has tests organized by crate. For a quick check during development:

```bash
# Core library tests (no MLIR required)
cargo test -p apxm-core
cargo test -p apxm-artifact
cargo test -p apxm-ais
cargo test -p apxm-runtime

# Driver tests
cargo test -p apxm-driver
```

For full workspace tests:

```bash
cargo test --workspace
```

For compiler tests (requires activated MLIR environment):

```bash
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"
cargo test -p apxm-compiler
```

For feature-gated tests:

```bash
cargo test -p apxm-runtime --features metrics
cargo test -p apxm-driver --features metrics
```

See [Testing Matrix](testing.md) for more details.

### Running Benchmarks

The benchmark suite compares A-PXM against LangGraph.

**Using Python CLI (Recommended):**

```bash
# Build compiler first
python tools/apxm_cli.py compiler build

# List available workloads
python tools/apxm_cli.py workloads list

# Check all workloads compile
python tools/apxm_cli.py workloads check

# Run a specific workload
python tools/apxm_cli.py workloads run 10_multi_agent
```

**Or manually:**

```bash
cd papers/cf26/benchmarks

# Run a specific workload (e.g., multi-agent)
python workloads/10_multi_agent/run.py

# Run all DSL comparison workloads
python run_all.py --workloads

# Run in quick mode (fewer iterations)
python run_all.py --workloads --quick

# Run paper benchmarks (for CF'26)
python run_all.py --paper

# JSON output for analysis
python run_all.py --workloads --json > results.json
```

**Prerequisites for benchmarks:**
- Ollama running with `phi3:mini` model: `ollama serve && ollama pull phi3:mini`
- Python deps: `pip install langgraph langchain-ollama`

See [Benchmark README](../papers/cf26/benchmarks/workloads/README.md) for full details.

## 8) Configure the LLM Model/Provider

AIS programs that use `RSN`, `PLAN`, `REFLECT`, `VERIFY`, or `COMMUNICATE` require an LLM backend. APXM discovers configuration from:

1. Project `.apxm/config.toml` (walking up from the current directory)
2. Global `~/.apxm/config.toml`

### MVP recommendation (single model)

For the current MVP we assume a single default backend/model (later we can extend the DSL to select among multiple backends/models).

Create `.apxm/config.toml` at the repo root:

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

Then run Ollama:

```bash
ollama serve
ollama list   # verify the model name matches config
```

### Optional: use cloud APIs (OpenAI / Gemini)

If you prefer direct cloud APIs instead of Ollama:

```toml
[chat]
providers = ["openai"]
default_model = "openai"
planning_model = "openai"

[[llm_backends]]
name = "openai"
provider = "openai"
model = "gpt-4o-mini"
api_key = "env:OPENAI_API_KEY"
```

```toml
[chat]
providers = ["gemini"]
default_model = "gemini"
planning_model = "gemini"

[[llm_backends]]
name = "gemini"
provider = "google"
model = "gemini-2.5-flash"
api_key = "env:GEMINI_API_KEY"
```

## Troubleshooting

- If `doctor` reports missing MLIR tools, re-run:
  - `cargo run -p apxm-cli -- install`
  - `conda activate apxm`
  - `eval "$(cargo run -p apxm-cli -- activate)"`
- If MLIR is installed but the compiler fails to link, check that `MLIR_DIR` and `LLVM_DIR` are set to `$CONDA_PREFIX/lib/cmake/{mlir,llvm}`.

### Environment Variables

After activating the conda environment, the following must be set for MLIR compilation:

```bash
export MLIR_DIR=$CONDA_PREFIX/lib/cmake/mlir
export LLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
```

The `eval "$(cargo run -p apxm-cli -- activate)"` command sets these automatically.

For non-interactive shells or CI, set them manually:

```bash
export CONDA_PREFIX=/path/to/miniforge3/envs/apxm
export MLIR_DIR=$CONDA_PREFIX/lib/cmake/mlir
export LLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH  # macOS
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH      # Linux
```

### Common Build Errors

**"Library not loaded: @rpath/libapxm_compiler_c.dylib"**

The compiler library needs to be in the library path. Either:
1. Use the activate command: `eval "$(cargo run -p apxm-cli -- activate)"`
2. Or set `DYLD_LIBRARY_PATH` (macOS) / `LD_LIBRARY_PATH` (Linux) to include `target/release/lib`

**"MLIR toolchain not detected"**

Ensure `CONDA_PREFIX` points to the apxm conda environment:
```bash
conda activate apxm
echo $CONDA_PREFIX  # Should show .../envs/apxm
```
