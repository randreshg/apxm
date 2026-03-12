---
title: "APXM Graph JSON Contract"
description: "Stable JSON contract for external graph generators using ApxmGraph::from_json()."
---

# APXM Graph JSON Contract

`ApxmGraph::from_json()` is a stable public contract for external graph generators.

## Required Fields

- `name: string`
- `nodes: GraphNode[]`
- `edges: GraphEdge[]`

## Canonical Value Formats

- `nodes[].op` uses `SCREAMING_SNAKE_CASE` AIS operation names (for example `ASK`, `WAIT_ALL`, `CONST_STR`).
- `edges[].dependency` uses `Data | Control | Effect`.
- Keys are case-sensitive and must match the canonical field names exactly (`name`, `nodes`, `edges`, `parameters`, `metadata`, `attributes`).

## Optional Fields (defaulted)

- `parameters: Parameter[]` defaults to `[]`
- `metadata: object` defaults to `{}`
- `GraphNode.attributes` defaults to `{}`

## Validation Guarantees

`from_json()` parses then validates. Invalid graphs are rejected early, including:

- duplicate node IDs
- invalid edges
- empty graph or node names
- duplicate/invalid parameters
- cyclic/invalid dependency structures (must be a DAG)

## Compatibility Guidance

- Treat operation names and attribute keys as case-sensitive.
- Prefer explicit `parameters` and `metadata` fields for forward compatibility.
- Keep unknown data in graph-level `metadata` instead of overloading node attributes.
