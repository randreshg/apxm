//! Capability metadata for AAM.
//!
//! Capabilities represent tools and functions that the agent can invoke,
//! with metadata describing their parameters, schemas, and characteristics.

use serde::{Deserialize, Serialize};

/// Represents a parameter of a capability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    /// Name of the parameter.
    pub name: String,
    /// Type of the parameter (e.g., "string", "number", "object").
    pub param_type: String,
    /// Whether this parameter is required.
    pub required: bool,
    /// JSON Schema for this parameter.
    pub schema: serde_json::Value,
}

impl Parameter {
    /// Creates a new parameter.
    pub fn new(name: String, param_type: String, required: bool) -> Self {
        Parameter {
            name,
            param_type,
            required,
            schema: serde_json::json!({}),
        }
    }

    /// Creates a new parameter with a JSON Schema.
    pub fn with_schema(
        name: String,
        param_type: String,
        required: bool,
        schema: serde_json::Value,
    ) -> Self {
        Parameter {
            name,
            param_type,
            required,
            schema,
        }
    }
}

/// Metadata describing a capability that the agent can invoke.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilitySchemas {
    /// JSON Schema for the parameter schema.
    pub parameter: serde_json::Value,
    /// JSON Schema for the return value.
    pub result: serde_json::Value,
}

impl CapabilitySchemas {
    /// Creates paired parameter/return schemas.
    pub fn new(parameter: serde_json::Value, result: serde_json::Value) -> Self {
        CapabilitySchemas { parameter, result }
    }
}

/// Execution characteristics of a capability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilityExecutionProfile {
    /// Whether this capability requires sandboxing.
    pub requires_sandbox: bool,
    /// Estimated cost per invocation (in arbitrary units).
    pub estimated_cost: f64,
    /// Estimated latency in milliseconds.
    pub estimated_latency_ms: u64,
}

impl CapabilityExecutionProfile {
    /// Creates a new execution profile.
    pub fn new(requires_sandbox: bool, estimated_cost: f64, estimated_latency_ms: u64) -> Self {
        CapabilityExecutionProfile {
            requires_sandbox,
            estimated_cost,
            estimated_latency_ms,
        }
    }
}

/// Metadata describing a capability that the agent can invoke.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilityMetadata {
    /// Name of the capability.
    pub name: String,
    /// Human-readable description of what the capability does.
    pub description: String,
    /// Version of the capability.
    pub version: String,
    /// JSON Schema for the parameter schema.
    pub parameter_schema: serde_json::Value,
    /// JSON Schema for the return value.
    pub return_schema: serde_json::Value,
    /// Whether this capability requires sandboxing.
    pub requires_sandbox: bool,
    /// Estimated cost per invocation (in arbitrary units).
    pub estimated_cost: f64,
    /// Estimated latency in milliseconds.
    pub estimated_latency_ms: u64,
    /// Tags for categorizing the capability.
    pub tags: Vec<String>,
}

impl CapabilityMetadata {
    /// Creates a new capability metadata.
    pub fn new(
        name: String,
        description: String,
        version: String,
        parameter_schema: serde_json::Value,
        return_schema: serde_json::Value,
    ) -> Self {
        CapabilityMetadata {
            name,
            description,
            version,
            parameter_schema,
            return_schema,
            requires_sandbox: false,
            estimated_cost: 0.0,
            estimated_latency_ms: 0,
            tags: Vec::new(),
        }
    }

    /// Creates a new capability metadata with all fields.
    pub fn with_details(
        name: String,
        description: String,
        version: String,
        schemas: CapabilitySchemas,
        execution: CapabilityExecutionProfile,
        tags: Vec<String>,
    ) -> Self {
        CapabilityMetadata {
            name,
            description,
            version,
            parameter_schema: schemas.parameter,
            return_schema: schemas.result,
            requires_sandbox: execution.requires_sandbox,
            estimated_cost: execution.estimated_cost,
            estimated_latency_ms: execution.estimated_latency_ms,
            tags,
        }
    }

    /// Adds a tag to the capability.
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Checks if the capability has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(&tag.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_new() {
        let param = Parameter::new("test_param".to_string(), "string".to_string(), true);
        assert_eq!(param.name, "test_param");
        assert_eq!(param.param_type, "string");
        assert!(param.required);
    }

    #[test]
    fn test_parameter_with_schema() {
        let schema = serde_json::json!({"type": "string", "minLength": 1});
        let param = Parameter::with_schema(
            "name".to_string(),
            "string".to_string(),
            true,
            schema.clone(),
        );
        assert_eq!(param.schema, schema);
    }

    #[test]
    fn test_capability_metadata_new() {
        let schema = serde_json::json!({});
        let capability = CapabilityMetadata::new(
            "test_cap".to_string(),
            "Test capability".to_string(),
            "1.0.0".to_string(),
            schema.clone(),
            schema.clone(),
        );
        assert_eq!(capability.name, "test_cap");
        assert_eq!(capability.description, "Test capability");
        assert!(!capability.requires_sandbox);
        assert_eq!(capability.estimated_cost, 0.0);
    }

    #[test]
    fn test_capability_metadata_with_details() {
        let schema = serde_json::json!({});
        let schemas = CapabilitySchemas::new(schema.clone(), schema.clone());
        let execution = CapabilityExecutionProfile::new(true, 10.5, 100);
        let capability = CapabilityMetadata::with_details(
            "cap".to_string(),
            "Desc".to_string(),
            "1.0.0".to_string(),
            schemas,
            execution,
            vec!["tag1".to_string()],
        );
        assert!(capability.requires_sandbox);
        assert_eq!(capability.estimated_cost, 10.5);
        assert_eq!(capability.estimated_latency_ms, 100);
        assert_eq!(capability.tags.len(), 1);
    }

    #[test]
    fn test_capability_metadata_tags() {
        let schema = serde_json::json!({});
        let mut capability = CapabilityMetadata::new(
            "cap".to_string(),
            "Desc".to_string(),
            "1.0.0".to_string(),
            schema.clone(),
            schema.clone(),
        );
        capability.add_tag("tag1".to_string());
        capability.add_tag("tag2".to_string());
        capability.add_tag("tag1".to_string()); // Duplicate should not be added
        assert_eq!(capability.tags.len(), 2);
        assert!(capability.has_tag("tag1"));
        assert!(capability.has_tag("tag2"));
        assert!(!capability.has_tag("tag3"));
    }

    #[test]
    fn test_capability_metadata_serialization() {
        let schema = serde_json::json!({});
        let capability = CapabilityMetadata::new(
            "cap".to_string(),
            "Desc".to_string(),
            "1.0.0".to_string(),
            schema.clone(),
            schema.clone(),
        );
        let json = serde_json::to_string(&capability).expect("serialize capability");
        let deserialized: CapabilityMetadata =
            serde_json::from_str(&json).expect("deserialize capability");
        assert_eq!(capability.name, deserialized.name);
        assert_eq!(capability.description, deserialized.description);
    }
}
