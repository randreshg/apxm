//! APXM Graph — canonical intermediate representation between AIS DSL and MLIR.
//!
//! `ApxmGraph` is the single exchange format that the compiler front-end
//! produces and the runtime executor consumes.  It sits between the human-
//! authored AIS DSL and the MLIR dialect:
//!
//! ```text
//! AIS DSL  ──→  ApxmGraph (JSON)  ──→  AIS MLIR dialect  ──→  .apxmobj
//! ```
//!
//! # Key types
//!
//! - [`ApxmGraph`] — a named directed graph of [`GraphNode`]s connected by [`GraphEdge`]s.
//! - [`GraphNode`] — one AIS operation (op + typed attribute map).
//! - [`GraphEdge`] — a typed dependency between two nodes ([`DependencyType`]).
//!
//! # Usage
//!
//! ```rust
//! use apxm_graph::ApxmGraph;
//!
//! let json = r#"{"name":"hello","nodes":[],"edges":[],"parameters":[],"metadata":{}}"#;
//! let graph: ApxmGraph = serde_json::from_str(json).expect("parse");
//! assert_eq!(graph.name, "hello");
//! ```

use apxm_core::types::{AISOperationType, DependencyType, Value, execution::ExecutionDag};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

mod lower_dag;
mod lower_mlir;
mod validate;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("validation error: {0}")]
    Validation(String),
    #[error("lowering error: {0}")]
    Lowering(String),
}

impl From<serde_json::Error> for GraphError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ApxmGraph {
    pub name: String,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    #[serde(default)]
    pub parameters: Vec<Parameter>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GraphNode {
    pub id: u64,
    pub name: String,
    pub op: AISOperationType,
    #[serde(default)]
    pub attributes: HashMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphEdge {
    pub from: u64,
    pub to: u64,
    pub dependency: DependencyType,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Parameter {
    pub name: String,
    pub type_name: String,
}

impl ApxmGraph {
    pub fn from_json(input: &str) -> Result<Self, GraphError> {
        let graph = serde_json::from_str::<Self>(input)?;
        graph.validate()?;
        Ok(graph)
    }

    pub fn from_bytes(input: &[u8]) -> Result<Self, GraphError> {
        let graph = serde_json::from_slice::<Self>(input)?;
        graph.validate()?;
        Ok(graph)
    }

    pub fn to_json(&self) -> Result<String, GraphError> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, GraphError> {
        serde_json::to_vec(self).map_err(Into::into)
    }

    pub fn validate(&self) -> Result<(), GraphError> {
        validate::validate_graph(self)
    }

    pub fn to_execution_dag(&self) -> Result<ExecutionDag, GraphError> {
        lower_dag::lower_to_execution_dag(self)
    }

    pub fn to_mlir(&self) -> Result<String, GraphError> {
        lower_mlir::lower_to_mlir(self)
    }

    /// Merge multiple sub-graphs into a single graph.
    ///
    /// Node IDs are remapped to avoid collisions: graph\[0\] keeps original IDs,
    /// graph\[1\] IDs are offset by the max ID from graph\[0\], and so on.
    /// A `WAIT_ALL` synchronization node is appended that depends on all
    /// "exit" nodes (nodes with no outgoing edges) via `Control` edges.
    pub fn merge(name: &str, graphs: &[ApxmGraph]) -> ApxmGraph {
        if graphs.is_empty() {
            return ApxmGraph {
                name: name.to_string(),
                nodes: Vec::new(),
                edges: Vec::new(),
                parameters: Vec::new(),
                metadata: HashMap::new(),
            };
        }

        let mut merged_nodes: Vec<GraphNode> = Vec::new();
        let mut merged_edges: Vec<GraphEdge> = Vec::new();
        let mut merged_params: Vec<Parameter> = Vec::new();
        let mut merged_metadata: HashMap<String, Value> = HashMap::new();
        let mut seen_param_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        let mut id_offset: u64 = 0;

        for graph in graphs {
            for node in &graph.nodes {
                merged_nodes.push(GraphNode {
                    id: node.id + id_offset,
                    name: node.name.clone(),
                    op: node.op,
                    attributes: node.attributes.clone(),
                });
            }

            for edge in &graph.edges {
                merged_edges.push(GraphEdge {
                    from: edge.from + id_offset,
                    to: edge.to + id_offset,
                    dependency: edge.dependency.clone(),
                });
            }

            for param in &graph.parameters {
                if seen_param_names.insert(param.name.clone()) {
                    merged_params.push(param.clone());
                }
            }

            for (k, v) in &graph.metadata {
                merged_metadata.entry(k.clone()).or_insert_with(|| v.clone());
            }

            if let Some(max_id) = graph.nodes.iter().map(|n| n.id).max() {
                id_offset += max_id;
            }
        }

        // Find exit nodes: nodes with no outgoing edges
        let sources: std::collections::HashSet<u64> =
            merged_edges.iter().map(|e| e.from).collect();
        let exit_ids: Vec<u64> = merged_nodes
            .iter()
            .filter(|n| !sources.contains(&n.id))
            .map(|n| n.id)
            .collect();

        // Create WAIT_ALL sync node with a unique ID
        let wait_all_id = merged_nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        merged_nodes.push(GraphNode {
            id: wait_all_id,
            name: format!("{}_sync", name),
            op: AISOperationType::WaitAll,
            attributes: HashMap::new(),
        });

        // Add Control edges from each exit node to the sync node
        for exit_id in &exit_ids {
            merged_edges.push(GraphEdge {
                from: *exit_id,
                to: wait_all_id,
                dependency: DependencyType::Control,
            });
        }

        ApxmGraph {
            name: name.to_string(),
            nodes: merged_nodes,
            edges: merged_edges,
            parameters: merged_params,
            metadata: merged_metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxm_core::constants::graph::attrs as graph_attrs;

    #[test]
    fn graph_roundtrip_json_and_bytes() {
        let graph = ApxmGraph {
            name: "test_graph".to_string(),
            nodes: vec![
                GraphNode {
                    id: 1,
                    name: "start".to_string(),
                    op: AISOperationType::ConstStr,
                    attributes: HashMap::from([(
                        "value".to_string(),
                        Value::String("hello".to_string()),
                    )]),
                },
                GraphNode {
                    id: 2,
                    name: "ask".to_string(),
                    op: AISOperationType::Ask,
                    attributes: HashMap::from([(
                        graph_attrs::TEMPLATE_STR.to_string(),
                        Value::String("Summarize {0}".to_string()),
                    )]),
                },
            ],
            edges: vec![GraphEdge {
                from: 1,
                to: 2,
                dependency: DependencyType::Data,
            }],
            parameters: vec![Parameter {
                name: "topic".to_string(),
                type_name: "str".to_string(),
            }],
            metadata: HashMap::new(),
        };

        let json = graph.to_json().expect("serialize json");
        let parsed_json = ApxmGraph::from_json(&json).expect("parse json");
        assert_eq!(parsed_json, graph);

        let bytes = graph.to_bytes().expect("serialize bytes");
        let parsed_bytes = ApxmGraph::from_bytes(&bytes).expect("parse bytes");
        assert_eq!(parsed_bytes, graph);
    }

    #[test]
    fn from_json_contract_allows_missing_optional_fields() {
        let json = r#"{
          "name": "contract_graph",
          "nodes": [
            { "id": 1, "name": "ask", "op": "ASK", "attributes": { "template_str": "{0}" } }
          ],
          "edges": []
        }"#;
        let graph = ApxmGraph::from_json(json).expect("graph should parse");
        assert_eq!(graph.name, "contract_graph");
        assert!(graph.parameters.is_empty());
        assert!(graph.metadata.is_empty());
    }

    #[test]
    fn from_json_contract_rejects_duplicate_node_ids() {
        let json = r#"{
          "name": "invalid_graph",
          "nodes": [
            { "id": 1, "name": "a", "op": "CONST_STR", "attributes": { "value": "x" } },
            { "id": 1, "name": "b", "op": "ASK", "attributes": { "template_str": "{0}" } }
          ],
          "edges": [],
          "parameters": [],
          "metadata": {}
        }"#;
        let err = ApxmGraph::from_json(json).expect_err("duplicate ids should fail");
        assert!(err.to_string().contains("validation"));
    }

    fn make_simple_graph(name: &str, id_start: u64) -> ApxmGraph {
        ApxmGraph {
            name: name.to_string(),
            nodes: vec![
                GraphNode {
                    id: id_start,
                    name: format!("{}_const", name),
                    op: AISOperationType::ConstStr,
                    attributes: HashMap::from([("value".into(), Value::String("hi".into()))]),
                },
                GraphNode {
                    id: id_start + 1,
                    name: format!("{}_ask", name),
                    op: AISOperationType::Ask,
                    attributes: HashMap::from([(
                        graph_attrs::TEMPLATE_STR.into(),
                        Value::String("{0}".into()),
                    )]),
                },
            ],
            edges: vec![GraphEdge {
                from: id_start,
                to: id_start + 1,
                dependency: DependencyType::Data,
            }],
            parameters: vec![Parameter {
                name: "input".to_string(),
                type_name: "str".to_string(),
            }],
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn merge_empty_graphs_list() {
        let merged = ApxmGraph::merge("empty", &[]);
        assert_eq!(merged.name, "empty");
        assert!(merged.nodes.is_empty());
        assert!(merged.edges.is_empty());
    }

    #[test]
    fn merge_single_graph() {
        let g = make_simple_graph("solo", 1);
        let merged = ApxmGraph::merge("merged", &[g]);
        // 2 original nodes + 1 WAIT_ALL sync
        assert_eq!(merged.nodes.len(), 3);
        let sync = merged.nodes.last().unwrap();
        assert_eq!(sync.op, AISOperationType::WaitAll);
        assert_eq!(sync.name, "merged_sync");
        // The sync node should have a Control edge from the exit node (id=2)
        let sync_edges: Vec<_> = merged.edges.iter().filter(|e| e.to == sync.id).collect();
        assert_eq!(sync_edges.len(), 1);
        assert_eq!(sync_edges[0].dependency, DependencyType::Control);
    }

    #[test]
    fn merge_two_graphs_remaps_ids() {
        let g1 = make_simple_graph("a", 1);
        let g2 = make_simple_graph("b", 1); // same starting IDs
        let merged = ApxmGraph::merge("combo", &[g1, g2]);

        // 2 + 2 + 1 WAIT_ALL = 5 nodes
        assert_eq!(merged.nodes.len(), 5);

        // All node IDs should be unique
        let ids: Vec<u64> = merged.nodes.iter().map(|n| n.id).collect();
        let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "node IDs must be unique after merge");

        // Edges should only reference valid node IDs
        for edge in &merged.edges {
            assert!(unique.contains(&edge.from), "edge.from {} not in node IDs", edge.from);
            assert!(unique.contains(&edge.to), "edge.to {} not in node IDs", edge.to);
        }

        // The WAIT_ALL node should have 2 Control edges (one per sub-graph exit)
        let sync = merged.nodes.last().unwrap();
        assert_eq!(sync.op, AISOperationType::WaitAll);
        let sync_edges: Vec<_> = merged.edges.iter().filter(|e| e.to == sync.id).collect();
        assert_eq!(sync_edges.len(), 2);
    }

    #[test]
    fn merge_deduplicates_parameters() {
        let mut g1 = make_simple_graph("a", 1);
        g1.parameters = vec![
            Parameter { name: "shared".into(), type_name: "str".into() },
            Parameter { name: "only_a".into(), type_name: "int".into() },
        ];
        let mut g2 = make_simple_graph("b", 1);
        g2.parameters = vec![
            Parameter { name: "shared".into(), type_name: "float".into() }, // dupe, should be skipped
            Parameter { name: "only_b".into(), type_name: "bool".into() },
        ];

        let merged = ApxmGraph::merge("combo", &[g1, g2]);
        assert_eq!(merged.parameters.len(), 3);
        let names: Vec<&str> = merged.parameters.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["shared", "only_a", "only_b"]);
        // First-wins: "shared" keeps type_name from g1
        assert_eq!(merged.parameters[0].type_name, "str");
    }

    #[test]
    fn merge_result_validates() {
        let g1 = make_simple_graph("a", 1);
        let g2 = make_simple_graph("b", 1);
        let merged = ApxmGraph::merge("valid", &[g1, g2]);
        merged.validate().expect("merged graph should pass validation");
    }
}
