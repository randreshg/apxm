//! Execution DAG node representation.
//!
//! Nodes represent operations in the execution DAG, with their inputs, outputs,
//! and metadata for scheduling.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::codelet::CodeletId;
use crate::types::{AISOperationType, TokenId, Value, validate_operation};

/// Type alias for node identifiers.
pub type NodeId = u64;

/// Metadata associated with a node for scheduling and optimization.
///
/// Fields with default values are omitted during serialization to keep the
/// JSON compact. When deserializing, missing fields fall back to their
/// defaults, so older payloads remain compatible.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct NodeMetadata {
    /// Priority for execution (higher = more important).
    /// Omitted from serialization when zero (the default).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub priority: u32,
    /// Estimated execution latency in nanoseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_latency: Option<u64>,
    /// Source codelet ID used to preserve codelet boundaries through lowering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codelet_source_id: Option<CodeletId>,
}

/// Returns `true` when a `u32` is zero (used by `skip_serializing_if`).
fn is_zero(v: &u32) -> bool {
    *v == 0
}

/// Returns `true` when a `HashMap` is empty (used by `skip_serializing_if`).
fn is_empty_map<K, V>(m: &HashMap<K, V>) -> bool {
    m.is_empty()
}

/// Returns `true` when a `Vec` is empty (used by `skip_serializing_if`).
fn is_empty_vec<T>(v: &[T]) -> bool {
    v.is_empty()
}

/// Returns `true` when `NodeMetadata` equals its `Default`.
fn is_default_metadata(m: &NodeMetadata) -> bool {
    *m == NodeMetadata::default()
}

/// Represents a node in the execution DAG.
///
/// A node represents a single operation with its inputs, outputs, attributes,
/// and scheduling metadata.
///
/// Serialization is tuned for compact, readable JSON:
/// - Empty collections (`attributes`, `input_tokens`, `output_tokens`) are
///   omitted rather than emitted as `{}` / `[]`.
/// - Default `metadata` is omitted entirely.
/// - All fields use `#[serde(default)]` so older payloads without new fields
///   deserialize without error.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Node {
    /// Unique identifier for the node.
    pub id: NodeId,
    /// The type of operation this node represents.
    pub op_type: AISOperationType,
    /// Attributes/parameters for this operation.
    /// Omitted from serialization when empty.
    #[serde(default, skip_serializing_if = "is_empty_map")]
    pub attributes: HashMap<String, Value>,
    /// Token IDs that serve as inputs to this node.
    /// Omitted from serialization when empty.
    #[serde(default, skip_serializing_if = "is_empty_vec")]
    pub input_tokens: Vec<TokenId>,
    /// Token IDs that serve as outputs from this node.
    /// Omitted from serialization when empty.
    #[serde(default, skip_serializing_if = "is_empty_vec")]
    pub output_tokens: Vec<TokenId>,
    /// Metadata for scheduling and optimization.
    /// Omitted from serialization when all fields are at their defaults.
    #[serde(default, skip_serializing_if = "is_default_metadata")]
    pub metadata: NodeMetadata,
}

impl Node {
    /// Creates a new node with default metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use apxm_core::types::{Node, NodeIdType, AISOperationType};
    ///
    /// let node = Node::new(1, AISOperationType::Inv);
    /// assert_eq!(node.id, 1);
    /// ```
    pub fn new(id: NodeId, op_type: AISOperationType) -> Self {
        Node {
            id,
            op_type,
            attributes: HashMap::new(),
            input_tokens: Vec::new(),
            output_tokens: Vec::new(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Adds an inptut token to this node.
    pub fn add_input_token(&mut self, token_id: TokenId) {
        self.input_tokens.push(token_id);
    }

    /// Adds an output token to this node.
    pub fn add_output_token(&mut self, token_id: TokenId) {
        self.output_tokens.push(token_id);
    }

    /// Sets an attribute for this node.
    pub fn set_attribute(&mut self, key: String, value: Value) {
        self.attributes.insert(key, value);
    }

    /// Gets and attribute value by key.
    pub fn get_attribute(&self, key: &str) -> Option<&Value> {
        self.attributes.get(key)
    }

    /// Validates the node's operation against its metadata.
    pub fn validate(&self) -> Result<(), crate::types::ValidationError> {
        validate_operation(self.op_type, &self.attributes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialization_with_missing_fields() {
        // Minimal JSON with only required fields should deserialize correctly,
        // filling in defaults for omitted optional fields.
        let json = r#"{"id":10,"op_type":"INV"}"#;
        let node: Node = serde_json::from_str(json).expect("deserialize minimal node");

        assert_eq!(node.id, 10);
        assert_eq!(node.op_type, AISOperationType::Inv);
        assert!(node.attributes.is_empty());
        assert!(node.input_tokens.is_empty());
        assert!(node.output_tokens.is_empty());
        assert_eq!(node.metadata, NodeMetadata::default());
    }

    #[test]
    fn test_roundtrip_with_all_fields() {
        let mut node = Node::new(5, AISOperationType::QMem);
        node.set_attribute("query".to_string(), Value::String("test".to_string()));
        node.add_input_token(100);
        node.add_output_token(200);
        node.metadata.priority = 10;
        node.metadata.estimated_latency = Some(5000);
        node.metadata.codelet_source_id = Some(7);

        let json = serde_json::to_string(&node).expect("serialize");
        let restored: Node = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(node, restored);
    }

    #[test]
    fn test_validation() {
        let mut node = Node::new(1, AISOperationType::QMem);
        node.set_attribute("query".to_string(), Value::String("test".to_string()));
        assert!(node.validate().is_ok());

        let node_invalid = Node::new(2, AISOperationType::QMem);
        assert!(node_invalid.validate().is_err());
    }
}
