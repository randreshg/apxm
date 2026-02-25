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
}
