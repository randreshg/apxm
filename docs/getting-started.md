# Getting Started with APXM

Welcome to APXM (Agent Programming eXecution Model), a complete compiler and runtime system for building AI agent workflows. This guide will walk you through installing APXM, setting up your first LLM backend, and running your first agent program.

## What is APXM?

APXM is a full-stack toolchain for developing autonomous AI agents, similar to how LLVM is a toolchain for compiling traditional programs. It provides:

- **ApxmGraph IR**: A canonical intermediate representation for agent workflows, expressed as dataflow DAGs (directed acyclic graphs)
- **AIS DSL**: A high-level domain-specific language for writing agent programs that compiles to ApxmGraph
- **Compiler**: Lowers ApxmGraph to AIS MLIR dialect, optimizes it, and generates executable binary artifacts (`.apxmobj` files)
- **Runtime**: Executes compiled artifacts with a parallel dataflow scheduler, memory system, and LLM backend registry
- **CLI**: Command-line tools for the complete compile/run workflow

APXM enables you to write agent workflows that coordinate multiple LLMs, manage memory across tiers (short-term, long-term, episodic), invoke external tools, and communicate between agents. The compiler produces optimized artifacts that can be executed with different LLM backends, credentials, and runtime configurations without recompilation.

Think of APXM as bringing traditional compiler toolchain benefits to AI agent development: separation of concerns between program logic and runtime configuration, optimization passes, portable artifacts, and reproducible execution.

## Prerequisites

Before installing APXM, you'll need:

1. **Conda or Mamba** (Miniforge recommended)
   - Miniforge includes both `conda` and the faster `mamba` package manager
   - Download from: https://github.com/conda-forge/miniforge

2. **Git** - for cloning the repository

3. **Basic build tools** - Most Linux/macOS systems include these:
   - CMake ≥ 3.20
   - Make
   - pkg-config

**Note:** You do NOT need to install Rust beforehand - the APXM installer can set it up automatically.

### Installing Miniforge

**Linux:**
```bash
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh | bash
```

**macOS (ARM):**
```bash
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh | bash
```

**macOS (Intel):**
```bash
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh | bash
```

After installation, restart your terminal or run:
```bash
source ~/.bashrc  # or ~/.zshrc for zsh users
```

## Installation

### Automated Installation (Recommended)

The easiest way to install APXM is using the automated installer:

```bash
# 1. Clone the repository
git clone https://github.com/randreshg/apxm
cd apxm

# 2. Add the APXM launcher to your PATH
export PATH="$PATH:$(pwd)/tools"

# 3. Run the automated installer
apxm install

# 4. Activate the conda environment
conda activate apxm

# 5. Verify the installation
apxm doctor
```

The `apxm install` command automatically:
1. Detects your OS, architecture, and package manager
2. Verifies Git, CMake, and other build tools
3. Creates/updates the 'apxm' conda environment with MLIR/LLVM 21
4. Installs rustup and Rust nightly (if needed)
5. Compiles the APXM CLI with features: driver, metrics
6. Reports any issues or confirms success

#### Installation Options

```bash
apxm install --check        # Dry-run: check status without making changes
apxm install --skip-deps    # Skip dependency verification
apxm install --skip-build   # Skip the build step
apxm install --auto         # Automatic mode (no prompts)
```

### Manual Installation

If you prefer to control each step:

```bash
# 1. Clone the repository
git clone https://github.com/randreshg/apxm
cd apxm

# 2. Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
source $HOME/.cargo/env

# 3. Create the conda environment
mamba env create -f environment.yaml
# OR with conda (slower):
conda env create -f environment.yaml

# 4. Activate the environment
conda activate apxm

# 5. Set up MLIR/LLVM paths
export MLIR_DIR="$CONDA_PREFIX/lib/cmake/mlir"
export LLVM_DIR="$CONDA_PREFIX/lib/cmake/llvm"

# 6. Build the project
cargo build -p apxm-cli --features driver,metrics --release

# 7. Add the tools directory to PATH
export PATH="$PATH:$(pwd)/tools"
```

To make the PATH addition permanent:
```bash
# For bash
echo 'export PATH="$PATH:/path/to/apxm/tools"' >> ~/.bashrc

# For zsh
echo 'export PATH="$PATH:/path/to/apxm/tools"' >> ~/.zshrc
```

### Verifying Your Installation

After installation, run the doctor command:

```bash
apxm doctor
```

This checks:
- APXM directory location
- Dependencies (Rust, Cargo, Conda, CMake, Ninja, Git)
- Rust toolchain version and nightly status
- Conda environment 'apxm' activation
- MLIR version 21.x
- LLVM libraries
- Build status (compiled binary location)
- Registered LLM credentials

All checks should show a green checkmark. If any fail, see the Troubleshooting section below.

## Registering an LLM Backend

Before running APXM programs, you need to register at least one LLM backend. APXM supports multiple providers and allows you to switch between them without recompiling your programs.

### Quick Setup: OpenAI

```bash
apxm register add my-openai --provider openai --api-key sk-...your-key...
```

The system will:
- Store your credentials securely at `~/.apxm/credentials.toml` (permissions: 0600)
- Set default model to `gpt-4o-mini`
- Set base URL to `https://api.openai.com/v1`

### Quick Setup: Anthropic

```bash
apxm register add my-anthropic --provider anthropic --api-key sk-ant-...your-key...
```

Defaults:
- Model: `claude-opus-4`
- Base URL: `https://api.anthropic.com/v1`

### Quick Setup: Ollama (Local)

For running open-source models locally:

```bash
# First, start Ollama server with your model
ollama serve

# Register with APXM
apxm register add my-ollama --provider ollama
```

Defaults:
- Model: `llama2` (you can change with `--model`)
- Base URL: `http://localhost:11434`
- No API key required

### Supported Providers

| Provider | Default Model | Base URL | API Key Required |
|----------|---------------|----------|------------------|
| `openai` | `gpt-4o-mini` | `https://api.openai.com/v1` | Yes |
| `anthropic` | `claude-opus-4` | `https://api.anthropic.com/v1` | Yes |
| `google` | `gemini-flash-latest` | `https://generativelanguage.googleapis.com/v1beta` | Yes |
| `ollama` | `llama2` | `http://localhost:11434` | No |
| `openrouter` | Provider-specific | `https://openrouter.ai/api` | Yes |

### Advanced Registration: Custom Endpoints

For enterprise deployments or custom OpenAI-compatible endpoints:

```bash
apxm register add my-custom \
  --provider openai \
  --base-url https://your-gateway.example.com/v1 \
  --api-key your-key \
  --model your-model \
  --header "X-Custom-Header=value"
```

### Managing Credentials

```bash
# List all registered credentials (API keys are masked)
apxm register list

# Test a credential (makes a real API call)
apxm register test my-openai

# Remove a credential
apxm register remove my-openai

# Generate config.toml from credentials
apxm register generate-config
```

### Security Features

APXM takes credential security seriously:
- **File permissions**: Credentials file is set to 0600 (owner read/write only)
- **Git protection**: Refuses to store credentials inside Git repositories
- **Auto-gitignore**: Creates `.gitignore` in `~/.apxm/` to prevent accidental commits
- **Masked output**: CLI commands show only first 4 and last 3 characters of API keys
- **Interactive entry**: Use `apxm register add` without `--api-key` to enter keys securely

---

## Your First APXM Program

Let's create a simple "Hello World" agent that generates a friendly greeting.

### Step 1: Write the Program

APXM supports three input formats:
- **AIS DSL** (`.ais`) - High-level agent programming language (recommended)
- **ApxmGraph JSON** (`.json`) - Low-level graph IR format
- **Python API** (`.py`) - Programmatic graph construction

For beginners, we recommend the **AIS DSL**. Create a file called `my_first_agent.ais`:

```ais
// my_first_agent.ais - My first APXM agent

agent HelloWorld {
    @entry flow main() -> str {
        ask("Generate a friendly greeting for someone learning about AI agents") -> greeting
        return greeting
    }
}
```

**What this does:**
- Defines an agent named `HelloWorld`
- Creates a flow called `main` that returns a string
- The `@entry` annotation marks this as the program entry point
- Uses the `ask` operation to query an LLM
- Returns the LLM's response

### Step 2: Run the Program

The simplest way to run an APXM program is with `execute`, which compiles and runs in one step:

```bash
apxm execute my_first_agent.ais
```

**Output:**
```
Hello! Welcome to the exciting world of AI agents! You're embarking on a fascinating journey...
```

**What happened:**
1. The AIS DSL was parsed into an AST
2. The AST was lowered to ApxmGraph JSON (canonical IR)
3. The graph was compiled to MLIR
4. MLIR was optimized and lowered to the AIS dialect
5. An executable artifact was generated
6. The artifact was loaded and executed by the APXM runtime
7. The runtime scheduled the `ask` operation and called your configured LLM backend

---

## Two-Step Workflow: Compile then Run

For production workflows or when debugging, you can separate compilation from execution:

### Step 1: Compile to Artifact

```bash
apxm compile my_first_agent.ais -o my_first_agent.apxmobj
```

**Output:**
```
Wrote graph artifact to my_first_agent.apxmobj
Compiled in 45.23ms, artifact generated in 2.15ms
```

**What you get:**
- `.apxmobj` file - A self-contained binary artifact containing:
  - Compiled DAG (Directed Acyclic Graph) of operations
  - Metadata about entry points, parameters, and memory requirements
  - Optimization level information
  - No source code or intermediate representations

### Step 2: Run the Artifact

```bash
apxm run my_first_agent.apxmobj
```

This loads the pre-compiled artifact and executes it directly, skipping all compilation steps.

**Why use this approach?**
- **Deployment**: Ship `.apxmobj` files without source code
- **Performance**: Skip compilation overhead for repeated runs
- **Distribution**: Artifacts are portable across machines with APXM installed

---

## Understanding the Three Input Formats

### 1. AIS DSL (.ais) - Recommended

**Best for:** Writing agent programs by hand

The AIS (Agent Instruction Set) DSL is a high-level language designed for expressing agent workflows naturally. It supports:
- Agent and flow definitions
- LLM operations: `ask`, `think`, `reason`
- Memory declarations
- Tool/capability bindings
- Cross-agent communication
- Control flow and data dependencies

**Example:** `examples/hello.ais`
```ais
agent HelloWorld {
    @entry flow main() -> str {
        ask("Generate a friendly greeting for someone learning about AI agents") -> greeting
        return greeting
    }
}
```

### 2. ApxmGraph JSON (.json) - IR Format

**Best for:** Programmatic generation, low-level control, debugging

ApxmGraph is the canonical intermediate representation (IR). The compiler always normalizes all inputs to this format before MLIR lowering.

**Example:** `examples/hello_graph.json`
```json
{
  "name": "hello_graph",
  "nodes": [
    {
      "id": 1,
      "name": "greeting",
      "op": "ASK",
      "attributes": {
        "template_str": "Generate a friendly greeting for someone learning about AI agents"
      }
    }
  ],
  "edges": [],
  "parameters": [],
  "metadata": {
    "is_entry": true
  }
}
```

**Structure:**
- `nodes`: List of operations with IDs, names, op types, and attributes
- `edges`: Data dependencies between nodes
- `parameters`: Flow input parameters
- `metadata`: Entry point markers, memory specs

### 3. Python API (.py) - Programmatic

**Best for:** Dynamic graph generation, integration with other Python tools

The Python API allows you to construct ApxmGraph JSON programmatically and submit it to the APXM server or compiler.

**Example:** `examples/demo_apxm_agents.py` (simplified)
```python
import requests

def build_graph(question: str) -> dict:
    """Build an ApxmGraph programmatically."""
    return {
        "name": "my_graph",
        "nodes": [
            {
                "id": 1,
                "op": "ASK",
                "attributes": {"prompt": question},
                "input_tokens": [],
                "output_tokens": [10],
            }
        ],
    }

# Execute via APXM server
graph = build_graph("What is APXM?")
response = requests.post("http://localhost:18800/v1/execute", json={"graph": graph})
print(response.json())
```

**Use cases:**
- Multi-agent councils with dynamic expert selection
- Workflow generation based on runtime data
- Integration with existing Python pipelines
- Server-side execution and agent-to-agent communication

---

## Passing Arguments to Programs

Many workflows need input parameters. Here's how to pass them:

### AIS DSL with Parameters

```ais
agent Researcher {
    @entry flow main(topic: str) -> str {
        ask("Research this topic: " + topic) -> findings
        return findings
    }
}
```

**Run with arguments:**
```bash
apxm execute researcher.ais "quantum computing"
```

### ApxmGraph JSON with Parameters

```json
{
  "name": "research_graph",
  "parameters": [
    {"name": "topic", "type": "String"}
  ],
  "nodes": [
    {
      "id": 1,
      "op": "ASK",
      "attributes": {
        "template_str": "Research this topic: {0}"
      }
    }
  ],
  "metadata": {"is_entry": true}
}
```

**Run with arguments:**
```bash
apxm execute research_graph.json "quantum computing"
```

---

## Compilation Options

### Optimization Levels

```bash
# No optimization (fastest compile, slowest runtime)
apxm compile workflow.ais -o workflow.apxmobj -O0

# Standard optimization (default)
apxm compile workflow.ais -o workflow.apxmobj -O1

# Aggressive optimization
apxm compile workflow.ais -o workflow.apxmobj -O2

# Maximum optimization
apxm compile workflow.ais -o workflow.apxmobj -O3
```

**What gets optimized:**
- **O0**: No passes, direct translation
- **O1+**: normalize, build-prompt, scheduling, canonicalizer, CSE, symbol-DCE
- **O1+**: `FuseAskOps` pass merges sequential LLM calls
- **O2+**: Additional MLIR standard passes
- **O3**: Maximum optimization (may increase compile time significantly)

### Emit Diagnostics

Get detailed compilation statistics:

```bash
apxm compile workflow.ais -o workflow.apxmobj --emit-diagnostics diag.json
```

**diag.json contains:**
- Input file and mode
- Graph name
- Optimization level
- Compilation time breakdown
- DAG statistics (nodes, edges, entry/exit points)
- Applied optimization passes

---

## Runtime Options

### Tracing and Debugging

Enable runtime tracing to see what's happening:

```bash
# High-level execution flow
apxm execute workflow.ais --trace info

# Detailed operation info
apxm execute workflow.ais --trace debug

# Full verbosity (LLM calls, tokens, DAG traversal)
apxm execute workflow.ais --trace trace
```

**Trace targets:**
- `apxm::scheduler` - Task scheduling and worker threads
- `apxm::ops` - Operation execution
- `apxm::llm` - LLM backend calls
- `apxm::tokens` - Token flow and dataflow graph
- `apxm::dag` - DAG traversal and dependencies

### Emit Runtime Metrics

Collect execution statistics:

```bash
apxm execute workflow.ais --emit-metrics metrics.json
```

**metrics.json contains:**
- Input file
- Optimization level
- Execution stats: nodes executed, failed, duration
- Scheduler metrics: worker utilization, queue depth
- LLM usage: input/output tokens

---

## Understanding APXM Programs

### Core Concepts

**Agents**: Containers for related flows. Agents can call flows in other agents for multi-agent collaboration. Think of an agent as a namespace for organizing related capabilities.

**Flows**: Functions that contain operations. Flows can have parameters and return values. They represent the executable logic of your agent program.

**Operations**: The building blocks of agent behavior:
- **LLM Operations**: `ask` (Q&A), `think` (extended reasoning), `reason` (structured reasoning)
- **Control Flow**: `switch` (conditionals), `merge` (join branches)
- **Memory**: `qmem` (query memory), `umem` (update memory)
- **Coordination**: `communicate` (agent-to-agent HTTP), `wait_all` (synchronization)
- **Tools**: `inv` (invoke external capabilities)

**Dataflow Execution**: APXM uses a dataflow DAG scheduler. Operations execute as soon as their inputs are ready, enabling automatic parallelization without explicit threading.

**Memory Tiers**:
- **STM** (Short-Term Memory): Working memory for the current task
- **LTM** (Long-Term Memory): Persistent facts and knowledge
- **Episodic**: Event history and temporal sequences

**Artifacts**: Compiled `.apxmobj` files are portable binaries containing the optimized DAG, metadata, and execution requirements - but no source code or intermediate representations.

### The Execution Model

When you run an APXM program:

1. **Parsing**: Source code (AIS DSL) is parsed into an abstract syntax tree
2. **Lowering**: The AST is normalized to ApxmGraph JSON (the canonical IR)
3. **MLIR Generation**: ApxmGraph is lowered to the AIS MLIR dialect
4. **Optimization**: Standard MLIR passes run (canonicalization, CSE, DCE, etc.)
5. **Artifact Generation**: Binary artifact with serialized DAG and metadata
6. **Runtime Loading**: Artifact is deserialized and validated
7. **Dataflow Scheduling**: Operations execute when dependencies are satisfied
8. **LLM Orchestration**: Backend registry routes calls to registered providers

This separation allows you to:
- Write programs once, run with different LLM backends
- Optimize at compile-time for runtime performance
- Distribute portable artifacts without exposing source code
- Cache compiled programs to avoid repeated compilation

---

## Common Patterns

### 1. Simple Q&A

```ais
agent SimpleQA {
    @entry flow main() -> str {
        ask("What is the capital of France?") -> answer
        return answer
    }
}
```

### 2. Multi-Step Reasoning

```ais
agent Researcher {
    @entry flow main(topic: str) -> str {
        ask("What are the key concepts in " + topic + "?") -> concepts
        think("Analyze these concepts in depth: " + concepts) -> analysis
        reason("Synthesize findings: " + analysis) -> conclusion
        return conclusion
    }
}
```

### 3. Parallel Expert Council

```ais
agent Council {
    @entry flow main(question: str) -> str {
        // These three asks run in parallel automatically
        ask("Expert 1 perspective: " + question) -> expert1
        ask("Expert 2 perspective: " + question) -> expert2
        ask("Expert 3 perspective: " + question) -> expert3

        // Synthesis waits for all three to complete
        ask("Synthesize: " + expert1 + expert2 + expert3) -> synthesis
        return synthesis
    }
}
```

See `examples/apxm_council.ais` for a full parallel council implementation.

### 4. Tool Use

```ais
agent ToolAgent {
    capability search(query: str) -> str;

    tools: [search]

    @entry flow main() -> str {
        ask("What should we research?") -> topic
        search(topic) -> results
        ask("Summarize: " + results) -> summary
        return summary
    }
}
```

See `examples/tool_use.ais` for details.

### 5. Multi-Agent Communication

```ais
agent Researcher {
    flow research(topic: str) -> str {
        think("Research: " + topic) -> findings
        return findings
    }
}

agent Coordinator {
    @entry flow main() -> str {
        ask("What topic?") -> topic
        Researcher.research(topic) -> findings
        ask("Summarize: " + findings) -> summary
        return summary
    }
}
```

See `examples/multi_flow.ais` for cross-agent flow invocation.

---

## Debugging Tips

### 1. Compilation Fails

**Check syntax:**
```bash
apxm compile workflow.ais -o /tmp/test.apxmobj
```

Common errors:
- Missing semicolons
- Undefined variables
- Type mismatches
- Invalid operation names

**Get diagnostics:**
```bash
apxm compile workflow.ais -o /tmp/test.apxmobj --emit-diagnostics /tmp/diag.json
cat /tmp/diag.json
```

### 2. Runtime Fails

**Enable tracing:**
```bash
apxm execute workflow.ais --trace debug
```

Look for:
- Which operation failed
- Error messages from LLM backend
- Token flow issues
- Scheduler deadlocks

**Check credentials:**
```bash
apxm register test
```

Make sure your LLM provider is accessible and the API key is valid.

### 3. LLM Backend Issues

**Test a specific backend:**
```bash
apxm register test my-openai
```

**Common credential issues:**

- **Invalid API key**: Remove and re-add the credential
  ```bash
  apxm register remove my-openai
  apxm register add my-openai --provider openai --api-key sk-...
  ```

- **Ollama not running**: Start the Ollama server
  ```bash
  ollama serve
  # Test connection:
  curl http://localhost:11434/api/tags
  ```

- **Custom headers not working**: Ensure proper format
  ```bash
  apxm register add my-gateway \
    --provider openai \
    --base-url https://gateway.example.com/v1 \
    --header "X-API-Key=value" \
    --header "X-User=username"
  ```

- **Duplicate credentials**: Each credential name must be unique
  ```bash
  apxm register list  # Check existing names
  apxm register remove old-name
  ```

**Check config:**
```bash
cat ~/.apxm/config.toml
```

Ensure:
- Provider names match between `config.toml` and credential store
- Base URLs are correct (especially for custom gateways)
- API keys are set (or use `env:VAR_NAME` indirection for environment variables)
- Models are spelled correctly and available from your provider

### 4. Performance Issues

**Profile with metrics:**
```bash
apxm execute workflow.ais --emit-metrics metrics.json --trace info
cat metrics.json
```

Check:
- Nodes executed vs total nodes (any deadlocks?)
- Duration per node (slow LLM calls?)
- Scheduler worker utilization

**Try higher optimization:**
```bash
apxm execute -O3 workflow.ais
```

### 5. Installation and Environment Issues

**Conda environment not found:**
```bash
# Create or update manually:
mamba env create -f environment.yaml
# OR update existing:
mamba env update -f environment.yaml -n apxm
```

**Rust nightly not found:**
```bash
rustup toolchain install nightly
# Verify:
rustup show
```

**Build fails with MLIR/LLVM errors:**
```bash
# Ensure conda environment is activated:
conda activate apxm

# Verify MLIR/LLVM paths:
echo $MLIR_DIR  # Should show: /path/to/conda/envs/apxm/lib/cmake/mlir
echo $LLVM_DIR  # Should show: /path/to/conda/envs/apxm/lib/cmake/llvm

# If empty, set them manually:
export MLIR_DIR="$CONDA_PREFIX/lib/cmake/mlir"
export LLVM_DIR="$CONDA_PREFIX/lib/cmake/llvm"

# Retry build:
apxm build
```

**Python dependencies missing (typer, rich):**
```bash
conda activate apxm
pip install typer rich
```

**apxm command not found:**
```bash
# From the project root:
export PATH="$PATH:$(pwd)/tools"

# Or use absolute path:
export PATH="$PATH:/home/raherrer/projects/agents/apxm/tools"

# Make permanent (bash):
echo 'export PATH="$PATH:/path/to/apxm/tools"' >> ~/.bashrc
```

**Permission denied on credentials file:**
```bash
# Fix permissions:
chmod 600 ~/.apxm/credentials.toml
chmod 700 ~/.apxm

# Verify:
ls -la ~/.apxm/
```

### 6. Getting Help

For more assistance:
- Run `apxm doctor` for automated diagnostics
- Check existing issues: https://github.com/randreshg/apxm/issues
- Review documentation: [CLI_GUIDE.md](CLI_GUIDE.md), [CONTRACTS.md](CONTRACTS.md)
- Enable verbose tracing: `apxm execute workflow.ais --trace debug`
- Check compilation diagnostics: `apxm compile workflow.ais -o out.apxmobj --emit-diagnostics diag.json`

---

## Next Steps

Now that you understand the basics, here's your learning path:

### 1. Explore the Examples (Recommended Order)

Start simple and progress to more complex patterns:

1. **`examples/hello.ais`** - Minimal single-operation agent (start here!)
2. **`examples/tool_use.ais`** - External capability invocation with the `inv` operation
3. **`examples/multi_flow.ais`** - Cross-agent flow calls and multi-step workflows
4. **`examples/apxm_council.ais`** - Parallel expert council with `wait_all` synchronization
5. **`examples/code_review_council.ais`** - Advanced 3-specialist code review pattern
6. **`examples/multi_agent_communicate.ais`** - HTTP-based agent-to-agent communication
7. **`examples/demo_apxm_agents.py`** - Python API with A2A protocol and MCP tool discovery

Run any example:
```bash
apxm execute examples/hello.ais
apxm execute examples/apxm_council.ais --trace info
```

### 2. Read the Documentation

**For Users:**
- **[CLI_GUIDE.md](CLI_GUIDE.md)** - Complete CLI reference with all commands and options
  - Credential management deep-dive
  - Testing (375+ tests, all mocked - no API keys needed)
  - Configuration files and precedence
  - Build options and features
  - Diagnostics and troubleshooting

**For Advanced Users:**
- **[CONTRACTS.md](CONTRACTS.md)** - Internal compiler/runtime contracts
  - Wire format specification (52-byte header + bincode payload)
  - Operation index table (31 operations)
  - Phase 1 ISA extensions (UpdateGoal, Guard, Claim, Pause, Resume)
  - Not user-facing, but useful for understanding internals

**For Developers:**
- **[DEVELOPMENT.md](../DEVELOPMENT.md)** - Development setup and project structure
  - 12 crates overview
  - Build system (Makefile + Cargo)
  - Testing strategy
  - Code quality tools (fmt, lint, check)

### 3. Run the Full Demo

Experience the complete APXM ecosystem with the flagship demo:

```bash
# Terminal 1: Start the APXM server
APXM_SERVER_ADDR=127.0.0.1:18800 cargo run -p apxm-server

# Terminal 2: Run the demo (requires on-premises LLM or registered provider)
python examples/demo_apxm_agents.py
```

The demo showcases:
- Direct LLM API calls to on-premises endpoints
- A2A v0.3 protocol (AgentCard discovery, task submission)
- MCP 2025-11-05 tool discovery and invocation
- APXM council graph execution (3 parallel experts + synthesis)
- Memory operations (fact storage and search)

### 4. Build Your Own Agent

**Quick Start Workflow:**

```bash
# 1. Create a new .ais file
cat > my_agent.ais << 'EOF'
agent MyAgent {
    @entry flow main(question: str) -> str {
        ask("Answer this: " + question) -> answer
        return answer
    }
}
EOF

# 2. Iterate with execute (fast compilation)
apxm execute my_agent.ais "What is APXM?"

# 3. Add complexity (multiple steps, parallel operations)
# Edit my_agent.ais...

# 4. Test with different LLM backends
apxm register add backup-llm --provider anthropic
# Edit ~/.apxm/config.toml to switch default provider

# 5. Compile for production
apxm compile my_agent.ais -o my_agent.apxmobj -O3

# 6. Deploy the artifact
apxm run my_agent.apxmobj "What is APXM?"
```

**Best Practices:**
- Start with a single flow and one or two operations
- Use `--trace debug` during development to understand execution
- Test with different prompts and inputs
- Add error handling with `switch` and conditional logic
- Use `wait_all` for coordinating parallel operations
- Compile with `-O3` for production deployments

### 5. Join the Community

- **GitHub Issues**: Report bugs or request features at https://github.com/randreshg/apxm/issues
- **Discussions**: Ask questions and share your agent programs
- **Examples Gallery**: Submit your own examples via pull requests

---

## Quick Reference

### Essential Commands

```bash
# Installation & Setup
apxm install                          # Install/update environment
apxm install --check                  # Dry-run check only
apxm doctor                           # Verify installation
conda activate apxm                   # Activate environment

# Credential Management
apxm register add <name> --provider <type> --api-key <key>
apxm register add <name> --provider <type>  # Prompts for key
apxm register list                    # List credentials (masked)
apxm register test [name]             # Test credential(s)
apxm register remove <name>           # Delete credential
apxm register generate-config         # Export to config.toml

# Building
apxm build                            # Build full project
apxm build --compiler                 # Build compiler only
apxm build --runtime                  # Build runtime only
apxm build --debug                    # Debug build (faster compile)
apxm build --no-trace                 # Zero-overhead (tracing compiled out)
apxm build --clean                    # Clean before building

# Running Programs
apxm execute <file> [args...]         # Compile + run (one step)
apxm execute <file> --trace debug     # Run with debug tracing
apxm execute <file> --trace trace     # Full verbosity
apxm compile <file> -o <out.apxmobj>  # Compile only
apxm compile <file> -o <out> -O3      # Compile with optimization
apxm run <file.apxmobj> [args...]     # Run pre-compiled artifact

# Testing
apxm test                             # Run all tests (375+ tests)
apxm test --runtime                   # Runtime tests only
apxm test --compiler                  # Compiler tests only
apxm test --credentials               # Credential tests
apxm test --backends                  # LLM backend tests
apxm test --package <name>            # Specific crate tests
```

### File Extensions

- `.ais` - AIS DSL source (high-level agent programs)
- `.json` - ApxmGraph IR (low-level graph representation)
- `.apxmobj` - Compiled artifact (binary executable)
- `.py` - Python script (programmatic graph construction)

### LLM Operations

- `ask(prompt)` - Simple Q&A with LLM
- `think(prompt, budget: N)` - Extended thinking with token budget
- `reason(prompt, context)` - Structured reasoning with belief updates

### Control Flow Operations

- `switch(condition)` - Conditional branching
- `merge()` - Join multiple branches
- `wait_all(...)` - Synchronize parallel operations

### Memory Operations

- `qmem(query)` - Query memory (search for facts)
- `umem(fact)` - Update memory (store new facts)

### Coordination Operations

- `communicate(agent, message)` - Agent-to-agent HTTP communication
- `inv(capability, params)` - Invoke external tool/capability

### Key Directories and Files

**Project Structure:**
- `examples/` - Sample programs (*.ais, *.json, *.py)
- `docs/` - Documentation (this file, CLI_GUIDE.md, CONTRACTS.md)
- `crates/` - Rust workspace (12 crates)
- `tools/` - CLI launcher and Python wrapper
- `target/release/apxm` - Compiled Rust binary

**User Configuration:**
- `~/.apxm/credentials.toml` - LLM credentials (mode 0600)
- `~/.apxm/config.toml` - Runtime configuration
- `~/.apxm/.gitignore` - Auto-generated git protection

**Conda Environment:**
- `environment.yaml` - Dependency specification
- `$CONDA_PREFIX/` - Installed environment (MLIR, LLVM, Python)

### Supported LLM Providers

| Provider | Default Model | Requires API Key |
|----------|---------------|------------------|
| `openai` | `gpt-4o-mini` | Yes |
| `anthropic` | `claude-opus-4` | Yes |
| `google` | `gemini-flash-latest` | Yes |
| `ollama` | `llama2` | No (local) |
| `openrouter` | Provider-specific | Yes |

### Common Patterns Quick Reference

**Simple Q&A:**
```ais
agent QA {
    @entry flow main(q: str) -> str {
        ask(q) -> answer
        return answer
    }
}
```

**Parallel Council:**
```ais
agent Council {
    @entry flow main(q: str) -> str {
        ask("Expert 1: " + q) -> e1
        ask("Expert 2: " + q) -> e2
        ask("Expert 3: " + q) -> e3
        wait_all(e1, e2, e3)
        ask("Synthesize: " + e1 + e2 + e3) -> result
        return result
    }
}
```

**Multi-Agent:**
```ais
agent Researcher {
    flow research(topic: str) -> str {
        ask("Research: " + topic) -> findings
        return findings
    }
}

agent Main {
    @entry flow start(topic: str) -> str {
        Researcher.research(topic) -> data
        ask("Summarize: " + data) -> summary
        return summary
    }
}
```

---

**You're ready to build AI agent workflows with APXM!** Start with `examples/hello.ais`, explore the patterns, and consult the [CLI_GUIDE.md](CLI_GUIDE.md) for advanced features.

Happy agent building!
