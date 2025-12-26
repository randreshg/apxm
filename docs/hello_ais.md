# Hello AIS

A minimal end‑to‑end example that compiles and runs a tiny AIS program.

## 1) Install and Activate

```bash
cargo run -p apxm-cli -- install
conda activate apxm
eval "$(cargo run -p apxm-cli -- activate)"
```

## 2) Run the Example

```bash
cargo run -p apxm-cli --features driver -- run examples/hello_world.ais
```

Example output (will vary):

```
Executed 3 nodes in 42 ms
token 1 => "hello"
```

## 3) Compile and Inspect the Artifact

```bash
cargo run -p apxm-cli --features driver -- compile examples/hello_world.ais
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

The compile step writes an `.apxmobj` artifact next to the source file.
