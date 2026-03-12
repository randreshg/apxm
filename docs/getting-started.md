# Getting Started with APXM

APXM (Agent Programming eXecution Model) is a compiler and runtime for AI agent workflows — like LLVM for agent programs. It provides:

- **AIS DSL** — A high-level language for writing agent programs
- **ApxmGraph IR** — Canonical intermediate representation (dataflow DAGs)
- **Compiler** — Lowers to MLIR, optimizes, generates `.apxmobj` artifacts
- **Runtime** — Parallel dataflow scheduler with memory tiers and LLM backends

Write agent workflows once, compile them, run with any LLM backend.

---

## Installation

### Prerequisites

- **Conda or Mamba** ([Miniforge](https://github.com/conda-forge/miniforge) recommended)
- **Git**

Rust, CMake, and MLIR/LLVM are installed automatically by the installer.

### Install

```bash
git clone --recursive https://github.com/randreshg/apxm
cd apxm
pip install -e external/sniff
python3 tools/apxm_cli.py install
source ~/.bashrc  # or ~/.zshrc — restart shell
```

Verify:

```bash
apxm doctor
```

The installer creates a conda environment with MLIR/LLVM 21, builds the project, and installs a wrapper at `~/.local/bin/apxm` that handles all environment setup automatically.

### Troubleshooting

- **"apxm: command not found"** — Run `source ~/.bashrc` or check `ls ~/.local/bin/apxm`
- **Build failures** — Check `.apxm/install.log` for details, then re-run `apxm install`
- **Start fresh** — `conda env remove -n apxm && rm -rf target/ bin/ && python3 tools/apxm_cli.py install`

---

## Register an LLM Backend

Before running programs, register at least one LLM provider:

```bash
# OpenAI
apxm register add my-openai --provider openai --api-key sk-...

# Anthropic
apxm register add my-anthropic --provider anthropic --api-key sk-ant-...

# Ollama (local, no API key)
apxm register add local --provider ollama
```

Verify: `apxm register test`

See [LLM Backends](llm-backends.md) for full provider documentation, enterprise gateways, and security details.

---

## Your First Program

### Write it

Create `hello.ais`:

```ais
agent HelloWorld {
    @entry flow main() -> str {
        ask("Generate a friendly greeting for someone learning about AI agents") -> greeting
        return greeting
    }
}
```

### Run it

```bash
apxm execute hello.ais
```

What happens under the hood:
1. AIS DSL is parsed into an AST
2. AST is lowered to ApxmGraph JSON (canonical IR)
3. Graph is compiled to MLIR and optimized
4. Executable artifact is generated and run by the dataflow scheduler
5. The `ask` operation calls your configured LLM backend

### Compile and run separately

For production, separate compilation from execution:

```bash
# Compile to artifact
apxm compile hello.ais -o hello.apxmobj

# Run the artifact (skips compilation)
apxm run hello.apxmobj
```

Artifacts are portable, contain no source code, and can be distributed independently.

### With parameters

```ais
agent Researcher {
    @entry flow main(topic: str) -> str {
        ask("Research this topic: " + topic) -> findings
        return findings
    }
}
```

```bash
apxm execute researcher.ais "quantum computing"
```

---

## Common Patterns

### Multi-step reasoning

```ais
agent Analyst {
    @entry flow main(topic: str) -> str {
        ask("Key concepts in " + topic) -> concepts
        think("Analyze in depth: " + concepts) -> analysis
        reason("Synthesize: " + analysis) -> conclusion
        return conclusion
    }
}
```

### Parallel expert council

```ais
agent Council {
    @entry flow main(question: str) -> str {
        // These three run in parallel automatically (no data dependencies)
        ask("Expert 1: " + question) -> e1
        ask("Expert 2: " + question) -> e2
        ask("Expert 3: " + question) -> e3

        // Synthesis waits for all three
        ask("Synthesize: " + e1 + e2 + e3) -> result
        return result
    }
}
```

### Multi-agent collaboration

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

### Tool use

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

---

## Input Formats

APXM supports three input formats:

| Format | Extension | Best for |
|--------|-----------|----------|
| **AIS DSL** | `.ais` | Writing programs by hand (recommended) |
| **ApxmGraph JSON** | `.json` | Programmatic generation, low-level control |
| **Python API** | `.py` | Dynamic graph construction, integration |

All formats compile to the same ApxmGraph IR before MLIR lowering.

---

## Key Concepts

- **Dataflow execution** — Operations run when inputs are ready, not in textual order. Independent operations automatically parallelize.
- **Memory tiers** — STM (working memory), LTM (persistent facts), Episodic (event history).
- **Artifacts** — Compiled `.apxmobj` files are portable binaries. Ship without source code, run with any LLM backend.

For deeper understanding, see [Concepts](concepts/overview.md).

---

## Next Steps

1. **Explore examples** — `examples/hello.ais`, `examples/apxm_council.ais`, `examples/multi_flow.ais`
2. **CLI reference** — [CLI Reference](cli-reference.md) for all commands and options
3. **LLM backends** — [LLM Backends](llm-backends.md) for provider setup
4. **Architecture** — [Architecture](concepts/architecture.md) for system design
5. **The paper** — [A-PXM Paper](paper/_CF_26__A_PXM_for_Agentic_AI.pdf) for formal design and evaluation
