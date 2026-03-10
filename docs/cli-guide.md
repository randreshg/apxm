# APXM CLI Guide

## Quick Start

```bash
# 1. Install the environment
apxm install

# 2. Activate conda environment
conda activate apxm

# 3. Build the compiler and runtime
apxm build

# 4. Register your LLM provider credentials
apxm register add my-openai --provider openai --api-key sk-...

# 5. Run your first workflow
apxm execute examples/hello.json "Hello world"

# 6. Check everything is working
apxm doctor
```

---

## Credential Management

APXM uses a credential store at `~/.apxm/credentials.toml` to manage API keys securely. Credentials are stored as plaintext with strict file permissions (0600), following the same approach as AWS CLI, GitHub CLI, and npm.

### Register a Standard Provider

```bash
apxm register add my-openai --provider openai --api-key sk-proj-abc123...

# Omit --api-key to enter interactively (hidden input)
apxm register add my-anthropic --provider anthropic
```

### Register an Enterprise Gateway

```bash
apxm register add corp-gateway \
  --provider openai \
  --api-key dummy \
  --base-url https://llm-api.company.com/OnPrem \
  --model GPT-oss-20B \
  --header "Ocp-Apim-Subscription-Key=abc123" \
  --header "user=$USER"
```

### Register Local Ollama

```bash
apxm register add local --provider ollama
```

### Test Credentials

```bash
# Test a specific credential
apxm register test my-openai

# Test all credentials
apxm register test
```

### List Credentials

```bash
apxm register list
```

Output shows masked keys for security:
```
my-openai        openai       key=sk-p...xyz
corp-gateway     openai       key=dumm...ummy  model=GPT-oss-20B  +2 headers
local            ollama       key=<none>
```

### Remove and Re-add

Credentials are immutable. To update, remove and re-add:

```bash
apxm register remove my-openai
apxm register add my-openai --provider openai --api-key sk-new-key
```

### Generate config.toml

Generate `config.toml` entries from registered credentials:

```bash
apxm register generate-config >> .apxm/config.toml
```

### Supported Providers

| Provider | Name | Notes |
|----------|------|-------|
| OpenAI | `openai` | GPT-4, GPT-4o, etc. |
| Anthropic | `anthropic` | Claude models |
| Google | `google` | Gemini models |
| Ollama | `ollama` | Local models, no API key needed |
| OpenRouter | `openrouter` | Multi-provider gateway |

### Security

- File permissions: `0600` (owner read/write only), validated on every read
- Directory permissions: `0700` (owner-only access)
- Git protection: refuses to store in a git repository, auto-creates `.gitignore`
- Keys never shown in full in CLI output

---

## Building

```bash
# Build everything (compiler + runtime)
apxm build

# Build compiler only
apxm build --compiler

# Build runtime only
apxm build --runtime

# Debug build
apxm build --debug

# Clean build
apxm build --clean

# Build without tracing (zero overhead)
apxm build --no-trace
```

### Compiler Sub-commands

```bash
# Build via compiler sub-app
apxm compiler build
apxm compiler build --clean --no-trace
```

---

## Compiling

Compile an ApxmGraph JSON file to an artifact:

```bash
# Basic compile
apxm compile workflow.json -o workflow.apxmobj

# With optimization level
apxm compile workflow.json -o workflow.apxmobj -O2

# Emit diagnostics
apxm compile workflow.json -o workflow.apxmobj --emit-diagnostics diag.json

# Auto-build with cargo (slower but convenient)
apxm compile workflow.json -o workflow.apxmobj --cargo
```

---

## Running

### Execute (compile + run in one step)

```bash
# Basic execute
apxm execute workflow.json "input text"

# With optimization and tracing
apxm execute -O2 --trace debug workflow.json "input text"

# Emit runtime metrics
apxm execute --emit-metrics metrics.json workflow.json "input"
```

### Run Pre-compiled Artifacts

```bash
# Run a .apxmobj file
apxm run workflow.apxmobj "input text"

# With tracing
apxm run --trace info workflow.apxmobj
```

---

## Testing

**No LLM API keys are needed to run tests.** The entire test suite (375 tests) uses `MockLLMBackend` and dummy keys -- no real API calls are ever made. The only distinction is compiler tests (need MLIR/LLVM 21 installed) vs everything else (always works offline).

```bash
# Run all tests except compiler (no API keys needed, no MLIR needed)
apxm test

# Run all tests including compiler (requires MLIR/LLVM 21)
apxm test --all

# Test specific components
apxm test --runtime          # Runtime tests (133 tests, all mocked)
apxm test --compiler         # Compiler tests (requires MLIR)
apxm test --credentials      # Credential store tests
apxm test --backends         # Backend mock tests (74 tests)
apxm test --package apxm-core
```

### What Each Crate Tests Without API Keys

| Crate | Tests | What's Tested |
|-------|-------|---------------|
| apxm-runtime | 133 | Scheduler, memory, executor, DAG splicing (mock DAGs) |
| apxm-backends | 74 | MockLLMBackend, request/response parsing, registry, storage |
| apxm-core | 69 | Types, values, token counting, provider specs |
| apxm-credentials | 12 | File I/O, permissions, masking, TOML roundtrip |
| apxm-driver | 6 | Config TOML parsing |
| apxm-graph | 3 | Graph JSON serialization |
| apxm-artifact | 2 | Artifact serialization |
| apxm-compiler | 9 | MLIR context, FFI (needs MLIR installed) |

---

## Installation

```bash
# Full install (interactive)
apxm install

# Dry-run to check status
apxm install --check

# Automatic mode (no prompts)
apxm install --auto

# Skip dependency checks
apxm install --skip-deps

# Skip build step
apxm install --skip-build
```

### Install Stages

1. **Platform detection** - OS, architecture, package manager
2. **Dependency checks** - Rust, CMake, Ninja, Mamba/Conda
3. **Conda environment** - Create/update from `environment.yaml`
4. **Rust toolchain** - Install rustup + nightly if needed
5. **Build** - `cargo build --release`
6. **Verify** - Run doctor checks

---

## Diagnostics

```bash
apxm doctor
```

Checks:
- APXM directory existence
- System dependencies (Rust, Cargo, CMake, Ninja, Mamba)
- Rust toolchain (nightly recommended)
- Conda environment and MLIR/LLVM installation
- MLIR version (21.x expected)
- Compiler binary build status
- Registered credentials

---

## Environment Setup

### Shell Activation

```bash
# Activate conda env
conda activate apxm

# Set up MLIR/LLVM environment variables
eval "$(apxm activate)"
```

### Configuration

APXM looks for configuration in this order:
1. `--config` flag (explicit path)
2. `.apxm/config.toml` (project-local, walking up from cwd)
3. `~/.apxm/config.toml` (global)

### config.toml Example

```toml
[chat]
providers = ["my-openai", "corp-gateway"]
default_model = "gpt-4"

# Only needed if NOT using `apxm register`:
[[llm_backends]]
name = "my-openai"
provider = "openai"
api_key = "env:OPENAI_API_KEY"
model = "gpt-4"
```

When using `apxm register`, the `[[llm_backends]]` section is unnecessary -- the credential store is the source of truth. The `[chat].providers` list references credential names.
