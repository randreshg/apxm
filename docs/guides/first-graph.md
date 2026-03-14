# How to Build Your First Graph

This tutorial walks you through building APXM graphs from scratch using JSON and the CLI. Each step is copy-pasteable.

## 1. Your first graph: a single ASK node

Create a file called `hello.json`:

```json
{
  "name": "hello",
  "nodes": [
    {"id": 1, "name": "greet", "op": "ASK", "attributes": {"template_str": "Explain quantum computing in one sentence"}}
  ],
  "edges": [],
  "parameters": [],
  "metadata": {}
}
```

Every graph needs `name`, `nodes`, `edges`, `parameters`, and `metadata`. A single ASK node with no edges is the smallest valid graph.

Validate it:

```bash
apxm validate hello.json
```

Then inspect what the compiler sees:

```bash
apxm explain hello.json
```

## 2. Adding a second step: pipeline pattern

Create `pipeline.json`. The second node references the first node's output with `{{node_1}}`, and a `Data` edge declares the dependency:

```json
{
  "name": "pipeline",
  "nodes": [
    {"id": 1, "name": "draft", "op": "ASK", "attributes": {"template_str": "Write a haiku about Rust"}},
    {"id": 2, "name": "critique", "op": "ASK", "attributes": {"template_str": "Critique this haiku and suggest improvements: {{node_1}}"}}
  ],
  "edges": [
    {"from": 1, "to": 2, "dependency": "Data"}
  ],
  "parameters": [],
  "metadata": {}
}
```

The `Data` edge tells the scheduler that node 2 cannot start until node 1 finishes. The `{{node_1}}` placeholder is replaced with node 1's output at runtime.

Analyze the execution plan:

```bash
apxm analyze pipeline.json
```

This shows two sequential phases: node 1 runs first, then node 2.

## 3. Going parallel: fan-out pattern

Create `fanout.json`. Two independent ASK nodes run in parallel, then a WAIT_ALL node synchronizes them:

```json
{
  "name": "fan-out",
  "nodes": [
    {"id": 1, "name": "pros", "op": "ASK", "attributes": {"template_str": "List 3 pros of static typing"}},
    {"id": 2, "name": "cons", "op": "ASK", "attributes": {"template_str": "List 3 cons of static typing"}},
    {"id": 3, "name": "sync", "op": "WAIT_ALL", "attributes": {"tokens": ["{{node_1}}", "{{node_2}}"]}}
  ],
  "edges": [
    {"from": 1, "to": 3, "dependency": "Data"},
    {"from": 2, "to": 3, "dependency": "Data"}
  ],
  "parameters": [],
  "metadata": {}
}
```

Nodes 1 and 2 have no edges between them, so the scheduler runs them concurrently. Node 3 waits for both to finish.

Run the analysis to confirm parallelism:

```bash
apxm analyze fanout.json
```

The output will show nodes 1 and 2 in the same execution phase and report a speedup estimate.

## 4. Using parameters

Parameters let you pass values into a graph at runtime. Add a `parameters` array with objects containing `name` and `type_name`:

```json
{
  "name": "parameterized",
  "nodes": [
    {"id": 1, "name": "research", "op": "ASK", "attributes": {"template_str": "Summarize the key ideas of {{topic}}"}},
    {"id": 2, "name": "quiz", "op": "ASK", "attributes": {"template_str": "Write 3 quiz questions about {{topic}} based on: {{node_1}}"}}
  ],
  "edges": [
    {"from": 1, "to": 2, "dependency": "Data"}
  ],
  "parameters": [
    {"name": "topic", "type_name": "str"}
  ],
  "metadata": {}
}
```

Valid `type_name` values are: `str`, `int`, `float`, `bool`, `json`.

The `{{topic}}` placeholder is resolved from the parameter value provided at execution time:

```bash
apxm validate parameterized.json
apxm execute parameterized.json "quantum computing"
```

## 5. Composing graphs

Save two small graphs to separate files, then merge them with `apxm codelet merge`.

`step-a.json`:

```json
{
  "name": "step-a",
  "nodes": [
    {"id": 1, "name": "research", "op": "ASK", "attributes": {"template_str": "Research the history of LLVM"}}
  ],
  "edges": [],
  "parameters": [],
  "metadata": {}
}
```

`step-b.json`:

```json
{
  "name": "step-b",
  "nodes": [
    {"id": 1, "name": "research", "op": "ASK", "attributes": {"template_str": "Research the history of GCC"}}
  ],
  "edges": [],
  "parameters": [],
  "metadata": {}
}
```

Merge them into a single graph:

```bash
apxm codelet merge step-a.json step-b.json --name combined -o combined.json
```

This re-numbers node IDs to avoid collisions and adds a WAIT_ALL sync node. Validate the result:

```bash
apxm validate combined.json
apxm analyze combined.json
```

## 6. Compiling and running

The `compile` and `execute` commands require the `driver` feature and a conda environment with MLIR 21+.

Compile a graph to an optimized artifact:

```bash
apxm compile fanout.json -o fanout.apxmobj
```

Compile and execute in one step:

```bash
apxm execute fanout.json
```

Set the optimization level with `-O`:

```bash
apxm execute fanout.json -O2
```

## Next steps

- Run `apxm ops list` to browse all 32 AIS operations.
- Run `apxm template list` to see built-in graph patterns (ask, pipeline, fan-out, map-reduce, verify, conditional).
- Run `apxm template show <name> --json` to get a ready-to-use graph and pipe it to validate: `apxm template show fan-out --json | apxm validate /dev/stdin`.
