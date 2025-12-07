//! AIS (Agent Instruction Set) operation types.
//!
//! This module defines all the operation types that can be executed in the APXM system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::types::Value;

/// Represents all possible AIS operation types.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AISOperationType {
    /// Invoke a capability (tool/function call).
    Inv,
    /// Reasoning operation (LLM call for reasoning).
    Rsn,
    /// Query memory (read from memory system).
    QMem,
    /// Update memory (write to memory system).
    UMem,
    /// Planning operation (generate a plan using LLM).
    Plan,
    /// Wait for all input tokens to be ready.
    WaitAll,
    /// Merge multiple tokens into one.
    Merge,
    /// Memory fence (synchronization barrier).
    Fence,
    /// Exception handling.
    Exc,
}

impl fmt::Display for AISOperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AISOperationType::Inv => write!(f, "INV"),
            AISOperationType::Rsn => write!(f, "RSN"),
            AISOperationType::QMem => write!(f, "QMEM"),
            AISOperationType::UMem => write!(f, "UMEM"),
            AISOperationType::Plan => write!(f, "PLAN"),
            AISOperationType::WaitAll => write!(f, "WAIT_ALL"),
            AISOperationType::Merge => write!(f, "MERGE"),
            AISOperationType::Fence => write!(f, "FENCE"),
            AISOperationType::Exc => write!(f, "EXC"),
        }
    }
}

/// Representation of an AIS operation instance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AISOperation {
    /// Unique operation identifier.
    pub id: u64,
    /// Operation type.
    pub op_type: AISOperationType,
    /// Optional attributes associated with the operation.
    pub attributes: HashMap<String, Value>,
}

impl AISOperation {
    /// Creates a new operation with the given id and type.
    pub fn new(id: u64, op_type: AISOperationType) -> Self {
        AISOperation {
            id,
            op_type,
            attributes: HashMap::new(),
        }
    }

    /// Adds or replaces an attribute value.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: Value) {
        self.attributes.insert(key.into(), value);
    }

    /// Retrieves an attribute by key.
    pub fn get_attribute(&self, key: &str) -> Option<&Value> {
        self.attributes.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        assert_eq!(AISOperationType::Inv.to_string(), "INV");
        assert_eq!(AISOperationType::Rsn.to_string(), "RSN");
        assert_eq!(AISOperationType::QMem.to_string(), "QMEM");
        assert_eq!(AISOperationType::UMem.to_string(), "UMEM");
        assert_eq!(AISOperationType::Plan.to_string(), "PLAN");
        assert_eq!(AISOperationType::WaitAll.to_string(), "WAIT_ALL");
        assert_eq!(AISOperationType::Merge.to_string(), "MERGE");
        assert_eq!(AISOperationType::Fence.to_string(), "FENCE");
        assert_eq!(AISOperationType::Exc.to_string(), "EXC");
    }

    #[test]
    fn test_serialization() {
        let op = AISOperationType::Inv;
        let json = serde_json::to_string(&op).expect("serialize AISOperationType");
        assert_eq!(json, "\"INV\"");
    }

    #[test]
    fn test_deserialization() {
        let json = "\"RSN\"";
        let op: AISOperationType =
            serde_json::from_str(json).expect("deserialize AISOperationType");
        assert_eq!(op, AISOperationType::Rsn);
    }

    #[test]
    fn test_equality() {
        assert_eq!(AISOperationType::Inv, AISOperationType::Inv);
        assert_ne!(AISOperationType::Inv, AISOperationType::Rsn);
    }

    #[test]
    fn test_clone() {
        let op1 = AISOperationType::Inv;
        let op2 = op1.clone();
        assert_eq!(op1, op2);
    }

    #[test]
    fn test_operation_attributes() {
        let mut op = AISOperation::new(1, AISOperationType::Plan);
        op.set_attribute("key", Value::String("v".into()));
        assert_eq!(
            op.get_attribute("key"),
            Some(&Value::String("v".to_string()))
        );
    }

    #[test]
    fn test_operation_serialization() {
        let mut op = AISOperation::new(7, AISOperationType::Inv);
        op.set_attribute("flag", Value::Bool(true));
        let json = serde_json::to_string(&op).unwrap();
        let restored: AISOperation = serde_json::from_str(&json).unwrap();
        assert_eq!(op.id, restored.id);
        assert_eq!(restored.get_attribute("flag"), Some(&Value::Bool(true)));
    }
}
