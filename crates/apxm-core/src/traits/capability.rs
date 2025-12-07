//! Capability trait definition.
//!
//! Capabilities are tools/functions that can be invoked by the agent.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::aam::Parameter;
use crate::types::Value;

/// Errors that can occur during capability invocation.
#[derive(Debug, Error, Clone, PartialEq, Serialize, Deserialize)]
pub enum CapabilityError {
    /// Parameters failed validation or were missing.
    #[error("invalid parameters: {0}")]
    InvalidParameters(String),
    /// Execution failed with a descriptive message.
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
}

/// Contract implemented by all capabilities.
pub trait Capability: Send + Sync {
    /// Returns the unique name of the capability.
    fn name(&self) -> &str;
    /// Human readable description of the capability.
    fn description(&self) -> &str;
    /// Parameter definitions for this capability.
    fn parameters(&self) -> Vec<Parameter>;
    /// Invoke the capability with the provided parameters.
    fn invoke(&self, params: Value) -> Result<Value, CapabilityError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoCapability;

    impl Capability for EchoCapability {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echo back the provided value"
        }

        fn parameters(&self) -> Vec<Parameter> {
            vec![Parameter::new("input".into(), "any".into(), true)]
        }

        fn invoke(&self, params: Value) -> Result<Value, CapabilityError> {
            Ok(params)
        }
    }

    #[test]
    fn test_capability_invoke() {
        let cap = EchoCapability;
        let input = Value::String("hi".into());
        let out = cap.invoke(input.clone()).expect("invoke ok");
        assert_eq!(out, input);
    }

    #[test]
    fn test_capability_metadata() {
        let cap = EchoCapability;
        assert_eq!(cap.name(), "echo");
        assert_eq!(cap.description(), "Echo back the provided value");
        assert_eq!(cap.parameters().len(), 1);
    }

    #[test]
    fn test_error_types() {
        let err = CapabilityError::InvalidParameters("missing field".into());
        let serialized = serde_json::to_string(&err).expect("serialize");
        let restored: CapabilityError = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(err, restored);
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn trait_is_send_sync() {
        assert_send_sync::<Box<dyn Capability>>();
    }
}
