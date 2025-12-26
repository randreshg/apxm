//! Operation Validation
//!
//! Provides shared validation logic used by both compiler and runtime to ensure
//! operations have all required fields and correct types.

use crate::operations::{get_operation_spec, AISOperationType};
use crate::types::Value;
use std::collections::HashMap;
use thiserror::Error;

/// Validation error types.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// A required field is missing.
    #[error("Missing required field '{field}' for operation {operation}")]
    MissingField {
        operation: String,
        field: &'static str,
    },

    /// A field has an invalid type.
    #[error("Invalid type for field '{field}' in operation {operation}: expected {expected}, got {actual}")]
    InvalidFieldType {
        operation: String,
        field: String,
        expected: String,
        actual: String,
    },

    /// An unknown field was provided.
    #[error("Unknown field '{field}' for operation {operation}")]
    UnknownField { operation: String, field: String },

    /// Operation-specific validation failed.
    #[error("Validation failed for operation {operation}: {message}")]
    OperationSpecific { operation: String, message: String },
}

/// Validates an operation against its specification.
///
/// This function checks that all required fields are present. Unlike the
/// previous implementation, this function **fails loudly** if validation
/// cannot be performed (e.g., if the operation spec is missing).
///
/// # Arguments
///
/// * `op_type` - The operation type to validate
/// * `attributes` - The attributes/fields provided for the operation
///
/// # Returns
///
/// * `Ok(())` if validation passes
/// * `Err(ValidationError)` if validation fails
pub fn validate_operation(
    op_type: AISOperationType,
    attributes: &HashMap<String, Value>,
) -> Result<(), ValidationError> {
    let spec = get_operation_spec(op_type);

    // Check all required fields are present
    for field in spec.required_fields() {
        if !attributes.contains_key(field.name) {
            return Err(ValidationError::MissingField {
                operation: spec.name.to_string(),
                field: field.name,
            });
        }
    }

    Ok(())
}

/// Validates an operation with strict unknown field checking.
///
/// This is a stricter version that also rejects unknown fields.
pub fn validate_operation_strict(
    op_type: AISOperationType,
    attributes: &HashMap<String, Value>,
) -> Result<(), ValidationError> {
    let spec = get_operation_spec(op_type);

    // Check all required fields are present
    for field in spec.required_fields() {
        if !attributes.contains_key(field.name) {
            return Err(ValidationError::MissingField {
                operation: spec.name.to_string(),
                field: field.name,
            });
        }
    }

    // Check for unknown fields
    for key in attributes.keys() {
        if spec.get_field(key).is_none() {
            return Err(ValidationError::UnknownField {
                operation: spec.name.to_string(),
                field: key.clone(),
            });
        }
    }

    Ok(())
}

/// Check if an operation has all required fields.
///
/// Returns true if validation would pass, false otherwise.
/// Does not provide error details - use `validate_operation` for that.
pub fn has_required_fields(op_type: AISOperationType, attributes: &HashMap<String, Value>) -> bool {
    let spec = get_operation_spec(op_type);
    spec.required_fields().all(|f| attributes.contains_key(f.name))
}

/// Get the list of missing required fields for an operation.
pub fn missing_required_fields(
    op_type: AISOperationType,
    attributes: &HashMap<String, Value>,
) -> Vec<&'static str> {
    let spec = get_operation_spec(op_type);
    spec.required_fields()
        .filter(|f| !attributes.contains_key(f.name))
        .map(|f| f.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_rsn_success() {
        let mut attrs = HashMap::new();
        attrs.insert("prompt".to_string(), Value::String("test".to_string()));

        let result = validate_operation(AISOperationType::Rsn, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_rsn_missing_prompt() {
        let attrs = HashMap::new();

        let result = validate_operation(AISOperationType::Rsn, &attrs);
        assert!(matches!(result, Err(ValidationError::MissingField { .. })));
    }

    #[test]
    fn test_validate_qmem_success() {
        let mut attrs = HashMap::new();
        attrs.insert("query".to_string(), Value::String("test query".to_string()));

        let result = validate_operation(AISOperationType::QMem, &attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_umem_missing_value() {
        let mut attrs = HashMap::new();
        attrs.insert("key".to_string(), Value::String("mykey".to_string()));
        // Missing "value" field

        let result = validate_operation(AISOperationType::UMem, &attrs);
        assert!(matches!(result, Err(ValidationError::MissingField { field: "value", .. })));
    }

    #[test]
    fn test_validate_control_flow_ops() {
        // These were previously missing metadata - ensure they validate
        let mut attrs = HashMap::new();
        attrs.insert("label".to_string(), Value::String("target".to_string()));

        let result = validate_operation(AISOperationType::Jump, &attrs);
        assert!(result.is_ok(), "Jump should validate with label field");
    }

    #[test]
    fn test_validate_loop_end_no_fields() {
        // LoopEnd has no required fields
        let attrs = HashMap::new();
        let result = validate_operation(AISOperationType::LoopEnd, &attrs);
        assert!(result.is_ok(), "LoopEnd should validate with empty attributes");
    }

    #[test]
    fn test_has_required_fields() {
        let mut attrs = HashMap::new();
        attrs.insert("prompt".to_string(), Value::String("test".to_string()));

        assert!(has_required_fields(AISOperationType::Rsn, &attrs));
        assert!(!has_required_fields(AISOperationType::UMem, &attrs));
    }

    #[test]
    fn test_missing_required_fields() {
        let attrs = HashMap::new();
        let missing = missing_required_fields(AISOperationType::UMem, &attrs);
        assert!(missing.contains(&"key"));
        assert!(missing.contains(&"value"));
    }

    #[test]
    fn test_strict_validation_unknown_field() {
        let mut attrs = HashMap::new();
        attrs.insert("prompt".to_string(), Value::String("test".to_string()));
        attrs.insert("unknown_field".to_string(), Value::String("value".to_string()));

        // Normal validation should pass
        assert!(validate_operation(AISOperationType::Rsn, &attrs).is_ok());

        // Strict validation should fail
        let result = validate_operation_strict(AISOperationType::Rsn, &attrs);
        assert!(matches!(result, Err(ValidationError::UnknownField { .. })));
    }
}
