# APXM Graph JSON Contract

`ApxmGraph::from_json()` is a stable public contract for external graph generators.

## Required Fields

- `name: string`
- `nodes: GraphNode[]`
- `edges: GraphEdge[]`

## Optional Fields (defaulted)

- `parameters: Parameter[]` defaults to `[]`
- `metadata: object` defaults to `{}`
- `GraphNode.attributes` defaults to `{}`

## Validation Guarantees

`from_json()` parses then validates. Invalid graphs are rejected early, including:

- duplicate node IDs
- invalid edges
- missing required operation attributes
- cyclic/invalid dependency structures

## Compatibility Guidance

- Treat operation names and attribute keys as case-sensitive.
- Prefer explicit `parameters` and `metadata` fields for forward compatibility.
- Keep unknown data in graph-level `metadata` instead of overloading node attributes.
