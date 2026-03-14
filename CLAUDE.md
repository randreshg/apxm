# APXM -- Agent Programming eXecution Model

Compiler + dataflow runtime for agent workflows. Graphs of AIS (Agent Instruction Set) operations compile through MLIR to optimized artifacts, then execute on a parallel scheduler.

## CLI Commands

Global flag: `--json` emits machine-readable JSON for most commands.

### Discovery
```bash
apxm ops list                          # all 32 AIS ops, grouped by category
apxm ops list --category reasoning     # filter by category
apxm ops show ASK                      # detailed info + example JSON for one op
apxm template list                     # available starter graph patterns
apxm template show fan-out --json      # emit template as ready-to-use graph JSON
```

### Graph Authoring
```bash
apxm validate graph.json               # check graph against AIS contract
apxm validate graph.json --json        # machine-readable validation errors
apxm analyze graph.json                # parallelism, critical path, speedup estimate
apxm analyze graph.json --json         # full analysis as JSON
apxm explain graph.json                # human-readable summary of what a graph does
```

### Composition
```bash
apxm codelet merge a.json b.json --name combined   # merge graph fragments into one workflow
apxm codelet merge a.json b.json --name combined -o out.json
```

### Compilation & Execution
```bash
apxm compile graph.json                # compile graph to .apxmobj artifact
apxm compile graph.json -o out.apxmobj -O2          # with output path + opt level
apxm compile graph.json --emit-diagnostics diag.json # compilation statistics
apxm execute graph.json                # compile + run in one step
apxm execute graph.json -O0            # skip optimizations (e.g., FuseReasoning)
apxm execute graph.json --emit-metrics metrics.json  # runtime statistics
apxm run out.apxmobj                   # run a pre-compiled artifact
apxm run out.apxmobj --emit-metrics metrics.json
```

### Environment
```bash
apxm doctor                            # diagnose MLIR/LLVM/conda dependencies
apxm doctor --json                     # machine-readable environment report
apxm activate --shell bash             # print shell exports for MLIR/LLVM env setup
apxm install                           # install/update conda env from environment.yaml
```

### Configuration
```bash
apxm register add my-key --provider openai --api-key sk-...   # add LLM credential
apxm register add local --provider ollama --base-url http://localhost:11434
apxm register list                     # show registered credentials
apxm register test                     # validate all credentials via test API call
apxm register test my-key              # validate one credential
apxm register remove my-key            # delete a credential
apxm tools register my-tool --description "..."    # register external tool for INV nodes
apxm tools list                        # list registered tools
apxm tools remove my-tool              # remove a tool registration
```

## Graph JSON Contract

All graphs share this shape (see `apxm ops show <OP>` for per-op attributes):

```json
{"name": "...", "nodes": [{"id": 1, "name": "...", "op": "ASK", "attributes": {...}}], "edges": [{"from": 1, "to": 2, "dependency": "Data"}], "parameters": [], "metadata": {}}
```

Valid dependency types: `Data`, `Control`, `Effect`. Valid parameter types: `str`, `int`, `float`, `bool`, `json`.

## Build

Requires conda env with MLIR 21+. `cargo build -p apxm-cli --features driver` for full CLI.
