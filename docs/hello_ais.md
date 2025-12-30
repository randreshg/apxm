# Hello AIS

A minimal end-to-end example that compiles and runs a tiny AIS program.

## 1) Install Prerequisites

```bash
# Create the conda environment (installs MLIR/LLVM toolchain)
cargo run -p apxm-cli -- install

# Install Python CLI dependencies
pip install typer rich
```

## 2) Build the Compiler

```bash
python tools/apxm_cli.py compiler build
```

## 3) Run the Example

```bash
python tools/apxm_cli.py compiler run examples/hello_world.ais
```

If your program uses LLM-backed operations (`rsn`, `plan`, `reflect`, `verify`, `talk`), configure an LLM provider/model via `.apxm/config.toml` (see `docs/getting_started.md`).

Example output (will vary):

```
Executed 3 nodes in 42 ms
token 1 => "hello"
```

## 4) Compile and Inspect the Artifact

```bash
# Compile only
python tools/apxm_cli.py compiler compile examples/hello_world.ais -o examples/hello_world.apxmobj

# Inspect the artifact
cargo run -p apxm-artifact --example inspect -- examples/hello_world.apxmobj
```

Example output:

```
Artifact: examples/hello_world.apxmobj
Module: hello_world
Compiler version: 0.0.1
Created at (ms): 1735220000000
Nodes: 3
Edges: 2
Entry nodes: 1
Exit nodes: 1
```

The compile step writes an `.apxmobj` artifact to the specified output path.

## Troubleshooting

Check your environment status:

```bash
python tools/apxm_cli.py doctor
```

This shows whether conda, MLIR, LLVM, and the compiler are properly set up.
