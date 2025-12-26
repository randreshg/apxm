//! Execution DAG node representation.
//!
//! Nodes represent operations in the execution DAG, with their inputs, outputs,
//! and metadata for scheduling.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{AISOperationType, TokenId, Value, validate_operation};

/// Type alias for node identifiers.
pub type NodeId = u64;

/// Metadata associated with a node for scheduling and optimization.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct NodeMetadata {
    /// Priority for execution (higher = more important).
    pub priority: u32,
    /// Estimated execution latency
    /// TODO: Implement more sophisticated serialization format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_latency: Option<u64>, // nanoseconds
}

/// Represents a node in the execution DAG.
///
/// A node represents a single operation with its inputs, outputs, attributes,
/// and scheduling metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Node {
    /// Unique identifier for the node.
    pub id: NodeId,
    /// The type of operation this node represents.
    pub op_type: AISOperationType,
    /// Attributes/parameters for this operation.
    pub attributes: HashMap<String, Value>,
    /// Token IDs that serve as inputs to this node.
    pub input_tokens: Vec<TokenId>,
    /// Token IDs that serve as outputs from this node.
    pub output_tokens: Vec<TokenId>,
    /// Metadata for scheduling and optimization.
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
    fn test_new_node() {
        let node = Node::new(1, AISOperationType::Inv);
        assert_eq!(node.id, 1);
        assert_eq!(node.op_type, AISOperationType::Inv);
        assert!(node.attributes.is_empty());
        assert!(node.input_tokens.is_empty());
        assert!(node.output_tokens.is_empty());
    }

    #[test]
    fn test_add_input_token() {
        let mut node = Node::new(1, AISOperationType::Inv);
        node.add_input_token(10);
        node.add_input_token(20);
        assert_eq!(node.input_tokens.len(), 2);
        assert_eq!(node.input_tokens[0], 10);
        assert_eq!(node.input_tokens[1], 20);
    }

    #[test]
    fn test_add_output_token() {
        let mut node = Node::new(1, AISOperationType::Inv);
        node.add_output_token(30);
        assert_eq!(node.output_tokens.len(), 1);
        assert_eq!(node.output_tokens[0], 30);
    }

    #[test]
    fn test_set_get_attribute() {
        let mut node = Node::new(1, AISOperationType::Inv);
        node.set_attribute("key".to_string(), Value::String("value".to_string()));
        assert_eq!(
            node.get_attribute("key"),
            Some(&Value::String("value".to_string()))
        );
        assert_eq!(node.get_attribute("nonexistent"), None);
    }

    #[test]
    fn test_serialization() {
        let mut node = Node::new(1, AISOperationType::Inv);
        node.set_attribute("test".to_string(), Value::Bool(true));
        let json = serde_json::to_string(&node).expect("serialize node");
        assert!(json.contains("1"));
        assert!(json.contains("INV"));
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
