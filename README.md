# APXM – Agent Programming eXecution Model

APXM is a full toolchain for building autonomous agents. It combines:

- **A high-level DSL** (AIS) for declaring memory, flows, handlers, and tool invocations.
- **A compiler** that lowers AIS → MLIR → executable artifacts.
- **A runtime/linker** that wires artifacts to capabilities, an LLM registry, and execution memory.
- **A conversational UX** for translating natural language into runnable AIS, executing it, and showing diagnostics.

The project is inspired by the “Agent Programming eXecution Model” paper: agents maintain belief/goal structures, compile their plans into deterministic flows, and execute them under a scheduler with long‑term/short‑term memory.

---

## Getting Started

```bash
git clone https://github.com/miguelcsx/apxm
cd apxm
# create and start the mamba env
cargo install --path crates/apxm-cli # optional; cargo run -p apxm-cli also works
```

APXM keeps workspace state under `.apxm/` (artifacts, sessions, logs). The CLI expects a config file there (see below).

### CLI Commands

| Command | Description |
| ------- | ----------- |
| `apxm compile <input>` | Compile AIS/MLIR to an artifact (`.apxmobj`) or Rust source. |
| `apxm run <input>` | Compile + execute an AIS/MLIR file; optionally emit artifacts/Rust. |
| `apxm chat` | Launch the Codex-style chat TUI to translate English ⇢ AIS ⇢ execution. |
| `apxm version` | Print versions of the CLI and embedded compiler. |

Global flags: `-v/--verbose`, `-q/--quiet`, `-c/--config <file>`, `--no-color`.

Examples:

```bash
# Compile DSL to an artifact
apxm compile examples/simple_agent.ais -o .apxm/artifacts/simple.apxmobj

# Run a raw MLIR module
apxm run examples/pipeline.mlir --mlir

# Chat with the orchestrator using project config overrides
apxm --config .apxm/config.toml chat --model ollama
```

---

## Configuration

APXM reads configuration from (in precedence order):

1. `--config <path>` passed on the CLI
2. Project-scoped `.apxm/config.toml` (walking up ancestor directories)
3. Global `~/.apxm/config.toml`

The TOML schema is defined in `crates/apxm-config/src/lib.rs`. A minimal example:

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

At runtime the linker instantiates each backend, registers them with the LLM registry, and sets the default provider to `chat.providers[0]`. Ollama endpoints skip API-key validation; cloud providers must provide `api_key`.

---

## Chat Workflow

`apxm chat` launches a three-pane TUI:

- **Left**: session list and key bindings.
- **Center**: conversation transcript + input box.
- **Right**: tabbed “specialist” view (plan, compiler diagnostics, AAM state, etc.).

Natural language is sent through the translator:

1. Build a JSON outer-plan using the planning model.
2. Convert the plan into AIS via the DSL model.
3. Compile/run the AIS; retry up to 3× if the compiler reports a syntax error (LLM gets the diagnostics as feedback).
4. Stream the final response/diagnostics to the UI.

All tracing/logging (compiler/linker/runtime) is recorded in `~/.apxm/logs/apxm.log` so the UX stays clean.

---

## Project Layout

```
crates/
  apxm-chat       # Chat UX, translator, session store
  apxm-cli        # CLI entrypoints
  apxm-compiler   # MLIR passes + codegen
  apxm-config     # TOML schema
  apxm-linker     # Orchestrates compiler/runtime, capability loading
  apxm-models     # LLM backends/registry
  apxm-runtime    # Scheduler, memory, execution engine
  ...
examples/         # Sample AIS programs
.apxm/            # Generated artifacts, sessions, logs (gitignored)
```

Refer to `DESIGN_COMPLETE.md` and `ECOSYSTEM_DESIGN.md` for deeper architecture notes.

---

## Frequently Used Commands

```bash
# Compile + run with verbose tracing (logs end up in ~/.apxm/logs/apxm.log)
apxm -v run examples/planning_agent.ais

# Emit intermediate MLIR for debugging
apxm compile agent.ais --dump-parsed-mlir parsed.mlir --dump-optimized-mlir optimized.mlir

# Chat with a specific model override
apxm chat --model gemini

# List sessions the chat UI will show
sqlite3 ~/.apxm/sessions/sessions.db 'select id, model, message_count from sessions;'
```

---

## Resources

- `CLI.md`: exhaustive command reference
- `crates/apxm-runtime/DESIGN_SUMMARY.md`: runtime architecture
- `papers/` (or the cited APXM paper) for the theoretical background

Questions? Open an issue or ping the maintainers on the project chat.
