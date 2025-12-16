//! AIS operation definition.
//!
//! Contains the AISOperation struct that represents an operation instance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::Value;

use super::AISOperationType;

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

    /// Validates the operation against its metadata.
    pub fn validate(&self) -> Result<(), String> {
        super::validate_operation(&self.op_type, &self.attributes)
    }
}
