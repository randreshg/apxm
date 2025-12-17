use std::collections::HashMap;

use apxm_core::types::execution::{DagMetadata, ExecutionDag, Node, NodeMetadata};
use apxm_core::types::values::{Number, Value};
use apxm_core::types::{AISOperationType, DependencyType, Edge};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireDag {
    pub nodes: Vec<WireNode>,
    pub edges: Vec<WireEdge>,
    pub entry_nodes: Vec<u64>,
    pub exit_nodes: Vec<u64>,
    pub metadata_name: Option<String>,
}

impl WireDag {
    pub fn from_execution_dag(dag: &ExecutionDag) -> Self {
        Self {
            nodes: dag.nodes.iter().map(WireNode::from).collect(),
            edges: dag.edges.iter().map(WireEdge::from).collect(),
            entry_nodes: dag.entry_nodes.clone(),
            exit_nodes: dag.exit_nodes.clone(),
            metadata_name: dag.metadata.name.clone(),
        }
    }

    pub fn into_execution_dag(self) -> ExecutionDag {
        let nodes = self
            .nodes
            .into_iter()
            .map(|node| node.into_node())
            .collect();
        let edges = self
            .edges
            .into_iter()
            .map(|edge| Edge {
                from: edge.from,
                to: edge.to,
                token_id: edge.token_id,
                dependency_type: edge.dependency,
            })
            .collect();
        ExecutionDag {
            nodes,
            edges,
            entry_nodes: self.entry_nodes,
            exit_nodes: self.exit_nodes,
            metadata: DagMetadata {
                name: self.metadata_name,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireNode {
    pub id: u64,
    pub op_type: AISOperationType,
    pub attributes: Vec<WireAttribute>,
    pub input_tokens: Vec<u64>,
    pub output_tokens: Vec<u64>,
    pub metadata: WireNodeMetadata,
}

impl From<&Node> for WireNode {
    fn from(node: &Node) -> Self {
        let mut attributes: Vec<_> = node
            .attributes
            .iter()
            .map(|(key, value)| WireAttribute {
                key: key.clone(),
                value: WireValue::from(value),
            })
            .collect();
        attributes.sort_by(|a, b| a.key.cmp(&b.key));

        Self {
            id: node.id,
            op_type: node.op_type.clone(),
            attributes,
            input_tokens: node.input_tokens.clone(),
            output_tokens: node.output_tokens.clone(),
            metadata: WireNodeMetadata {
                priority: node.metadata.priority,
                estimated_latency: node.metadata.estimated_latency,
            },
        }
    }
}

impl WireNode {
    fn into_node(self) -> Node {
        let mut attributes = HashMap::with_capacity(self.attributes.len());
        for attr in self.attributes {
            attributes.insert(attr.key, attr.value.into_value());
        }

        Node {
            id: self.id,
            op_type: self.op_type,
            attributes,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            metadata: NodeMetadata {
                priority: self.metadata.priority,
                estimated_latency: self.metadata.estimated_latency,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireEdge {
    pub from: u64,
    pub to: u64,
    pub token_id: u64,
    pub dependency: DependencyType,
}

impl From<&Edge> for WireEdge {
    fn from(edge: &Edge) -> Self {
        Self {
            from: edge.from,
            to: edge.to,
            token_id: edge.token_id,
            dependency: edge.dependency_type.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireNodeMetadata {
    pub priority: u32,
    pub estimated_latency: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireAttribute {
    pub key: String,
    pub value: WireValue,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WireValue {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Array(Vec<WireValue>),
    Object(Vec<WireObjectEntry>),
    Token(u64),
}

impl WireValue {
    fn into_value(self) -> Value {
        match self {
            WireValue::Null => Value::Null,
            WireValue::Bool(v) => Value::Bool(v),
            WireValue::Integer(v) => Value::Number(Number::Integer(v)),
            WireValue::Float(v) => Value::Number(Number::Float(v)),
            WireValue::String(v) => Value::String(v),
            WireValue::Array(values) => {
                Value::Array(values.into_iter().map(|v| v.into_value()).collect())
            }
            WireValue::Object(entries) => {
                let mut map = HashMap::with_capacity(entries.len());
                for entry in entries {
                    map.insert(entry.key, entry.value.into_value());
                }
                Value::Object(map)
            }
            WireValue::Token(id) => Value::Token(id),
        }
    }
}

impl From<&Value> for WireValue {
    fn from(value: &Value) -> Self {
        match value {
            Value::Null => WireValue::Null,
            Value::Bool(v) => WireValue::Bool(*v),
            Value::Number(Number::Integer(v)) => WireValue::Integer(*v),
            Value::Number(Number::Float(v)) => WireValue::Float(*v),
            Value::String(v) => WireValue::String(v.clone()),
            Value::Array(values) => WireValue::Array(values.iter().map(WireValue::from).collect()),
            Value::Object(map) => {
                let mut entries: Vec<_> = map
                    .iter()
                    .map(|(k, v)| WireObjectEntry {
                        key: k.clone(),
                        value: WireValue::from(v),
                    })
                    .collect();
                entries.sort_by(|a, b| a.key.cmp(&b.key));
                WireValue::Object(entries)
            }
            Value::Token(id) => WireValue::Token(*id),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireObjectEntry {
    pub key: String,
    pub value: WireValue,
}
