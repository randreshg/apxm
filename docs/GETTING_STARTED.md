# Getting Started with APXM

This guide will help you install, configure, and run your first APXM program.

## Prerequisites

Before installing APXM, ensure you have:

- **mamba** or **conda** ([miniforge](https://github.com/conda-forge/miniforge) recommended)
- **Rust nightly** (managed automatically via `rust-toolchain.toml`)
- **Ollama** (optional, for LLM backend)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/randreshg/apxm
cd apxm
```

### Step 2: Create Conda Environment

```bash
# Create the 'apxm' environment with Python, MLIR/LLVM, and all dependencies
mamba env create -f environment.yaml
```

This creates a self-contained environment with:
- Python 3.11 + typer + rich (CLI dependencies)
- LLVM/MLIR 21 toolchain
- CMake, Ninja, and build tools

### Step 3: Add APXM to PATH

```bash
# Add to ~/.zshrc or ~/.bashrc for persistence
export PATH="$PATH:$(pwd)/bin"
```

### Step 4: Build the Project

```bash
apxm build
```

### Step 5: Verify Installation

```bash
apxm doctor
```

You should see green checkmarks for all components.

---

## Building the Project

```bash
# Build full project (compiler + runtime)
apxm build

# Build compiler only
apxm build --compiler

# Build runtime only
apxm build --runtime

# Build without tracing (zero overhead for benchmarks)
apxm build --no-trace
```

---

## Your First AIS Program

### Create a Simple Agent

Create a file named `hello.ais`:

```ais
agent HelloWorld {
    @entry flow main() -> str {
        ask("Generate a friendly greeting for a new developer") -> greeting
        return greeting
    }
}
```

### Run the Program

```bash
apxm execute hello.ais
```

### With Tracing (for debugging)

```bash
apxm execute hello.ais --trace debug
```

### Compile and Run Separately

```bash
# Compile to artifact
apxm compile hello.ais -o hello.apxmobj

# Run the artifact
apxm run hello.apxmobj
```

---

## LLM Configuration

AIS programs that use `ask`, `think`, `reason`, `plan`, or `reflect` operations require an LLM backend.

### Create Configuration File

Create `.apxm/config.toml` in your project directory:

```toml
[chat]
providers = ["ollama"]
default_model = "ollama"
planning_model = "ollama"

[[llm_backends]]
name = "ollama"
provider = "ollama"
model = "gpt-oss:20b-cloud"
endpoint = "http://localhost:11434"
```

### Start Ollama

```bash
ollama serve
ollama pull gpt-oss:20b-cloud
```

### Alternative: Cloud APIs

**OpenAI:**

```toml
[chat]
providers = ["openai"]
default_model = "openai"

[[llm_backends]]
name = "openai"
provider = "openai"
model = "gpt-4o-mini"
api_key = "env:OPENAI_API_KEY"
```

**Google Gemini:**

```toml
[chat]
providers = ["gemini"]
default_model = "gemini"

[[llm_backends]]
name = "gemini"
provider = "google"
model = "gemini-2.5-flash"
api_key = "env:GEMINI_API_KEY"
```

---

## Running Benchmarks

APXM includes comprehensive benchmarks comparing AIS with LangGraph.

```bash
# List available workloads
apxm workloads list

# Check all workloads compile
apxm workloads check

# Run a specific workload
apxm workloads run 1_parallel_research

# Run with JSON output
apxm workloads run 1 --json
```

For detailed benchmark documentation, see [Benchmark Workloads](../papers/CF26/benchmarks/workloads/README.md).

---

## Troubleshooting

### "Library not loaded" errors

**macOS:**

```bash
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"
# Or use the activate command:
eval "$(./target/release/apxm activate)"
```

**Linux:**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

### "MLIR toolchain not detected"

```bash
conda activate apxm
echo $CONDA_PREFIX  # Should show .../envs/apxm
```

If the environment doesn't exist, recreate it:

```bash
./target/release/apxm install
```

### "Compiler not built"

```bash
apxm build
```

### Rebuild after C++ changes

```bash
rm -rf target/release/build/apxm-compiler-*
cargo build -p apxm-compiler --release
```

### Ollama connection errors

```bash
# Ensure Ollama is running
ollama serve

# Check the model is available
ollama list

# Pull the model if missing
ollama pull gpt-oss:20b-cloud
```

---

## Next Steps

- See [AGENTS.md](AGENTS.md) for full CLI reference and AIS DSL documentation
- Explore the `examples/` directory for more AIS programs
- Check [Benchmark Workloads](../papers/CF26/benchmarks/workloads/README.md) for real-world examples
