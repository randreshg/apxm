//! Operation validation functions.
//!
//! Contains functions for validating operations against their metadata.

use std::collections::HashMap;

use crate::types::{AISOperationType, Value};

use super::registry::find_operation;

/// Validates an operation against its metadata.
pub fn validate_operation(
    op_type: &AISOperationType,
    attributes: &HashMap<String, Value>,
) -> Result<(), String> {
    let meta_name = op_type.metadata_name();
    if let Some(meta) = find_operation(meta_name) {
        for field in meta.required_fields() {
            if !attributes.contains_key(field.name) {
                return Err(format!(
                    "Missing required field '{}' for operation {}",
                    field.name, meta_name
                ));
            }
        }
    }
    Ok(())
}
