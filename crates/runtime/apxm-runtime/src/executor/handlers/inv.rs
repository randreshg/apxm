//! INV operation - Capability invocation with validation and timeout
//!
//! Invokes registered capabilities/tools through the capability system.
//! Provides automatic input validation, timeout enforcement, and error handling.

use super::{
    ExecutionContext, Node, Result, Value, get_optional_u64_attribute, get_string_attribute,
};
use std::collections::HashMap;

/// Execute INV operation - Invoke a registered capability
///
/// # Attributes
///
/// - `capability` (required): Name of the capability to invoke
/// - `timeout_ms` (optional): Custom timeout in milliseconds (default: 30000)
///
/// # Inputs
///
/// Input values are passed as capability arguments. The number and types
/// of inputs depend on the capability's schema.
///
/// # Returns
///
/// Result value from capability execution
///
/// # Errors
///
/// Returns error if:
/// - Capability not found
/// - Input validation fails against capability schema
/// - Execution times out
/// - Capability execution fails
///
/// # Example
///
/// ```text
/// INV(capability="echo", timeout_ms=5000) -> result
/// ```
pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    let capability_name = get_string_attribute(node, "capability")?;
    let timeout_ms = get_optional_u64_attribute(node, "timeout_ms")?.unwrap_or(30000);

    tracing::debug!(
        capability = %capability_name,
        inputs = inputs.len(),
        timeout_ms = timeout_ms,
        "Executing INV operation"
    );

    // Convert inputs to HashMap<String, Value>
    // Priority: 1) params_json attribute, 2) arg_* attributes, 3) positional inputs
    let mut args = HashMap::new();

    // First, check for params_json attribute (from InvOp MLIR)
    if let Some(params_json) = node.attributes.get("params_json").and_then(|v| v.as_string()) {
        // Parse JSON and extract key-value pairs
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(params_json)
            && let Some(obj) = parsed.as_object() {
                for (k, v) in obj {
                    let value = match v {
                        serde_json::Value::String(s) => Value::String(s.clone()),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                Value::Number(apxm_core::types::values::Number::Integer(i))
                            } else if let Some(f) = n.as_f64() {
                                Value::Number(apxm_core::types::values::Number::Float(f))
                            } else {
                                continue;
                            }
                        }
                        serde_json::Value::Bool(b) => Value::Bool(*b),
                        _ => continue, // Skip complex nested values
                    };
                    args.insert(k.clone(), value);
                }
            }
    }

    // If no args from params_json, check for arg_* attributes (named arguments)
    if args.is_empty() {
        let args_from_attrs = node
            .attributes
            .iter()
            .filter(|(k, _)| k.starts_with("arg_"))
            .map(|(k, v)| (k.trim_start_matches("arg_").to_string(), v.clone()))
            .collect::<HashMap<String, Value>>();

        if !args_from_attrs.is_empty() {
            args = args_from_attrs;
        } else {
            // Fall back to positional arguments from inputs
            for (i, input_value) in inputs.iter().enumerate() {
                args.insert(format!("arg{}", i), input_value.clone());
            }
        }
    }

    // Invoke capability with timeout
    let timeout = std::time::Duration::from_millis(timeout_ms);
    let result = ctx
        .capability_system
        .invoke_with_timeout(&capability_name, args, timeout)
        .await
        .map_err(|e| {
            tracing::error!(
                capability = %capability_name,
                error = %e,
                "Capability invocation failed"
            );
            e
        })?;

    tracing::info!(
        capability = %capability_name,
        "Capability invocation successful"
    );

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        capability::{CapabilitySystem, executor::EchoCapability},
        memory::{MemoryConfig, MemorySystem},
    };
    use apxm_backends::LLMRegistry;
    use apxm_core::types::{execution::NodeMetadata, operations::AISOperationType};
    use std::sync::Arc;

    async fn create_test_context_with_capability() -> ExecutionContext {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());

        // Register echo capability
        capability_system
            .register(Arc::new(EchoCapability::new()))
            .unwrap();

        ExecutionContext::new(
            memory,
            llm_registry,
            capability_system,
            crate::aam::Aam::new(),
        )
    }

    #[tokio::test]
    async fn test_inv_with_named_args() {
        let ctx = create_test_context_with_capability().await;

        let mut node = Node {
            id: 1,
            op_type: AISOperationType::Inv,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };

        node.attributes
            .insert("capability".to_string(), Value::String("echo".to_string()));
        node.attributes.insert(
            "arg_message".to_string(),
            Value::String("Hello World".to_string()),
        );

        let result = execute(&ctx, &node, vec![]).await.unwrap();
        assert_eq!(
            result.as_string().map(|s| s.as_str()),
            Some("Echo: Hello World")
        );
    }

    #[tokio::test]
    async fn test_inv_capability_not_found() {
        let ctx = create_test_context_with_capability().await;

        let mut node = Node {
            id: 1,
            op_type: AISOperationType::Inv,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };

        node.attributes.insert(
            "capability".to_string(),
            Value::String("nonexistent".to_string()),
        );

        let result = execute(&ctx, &node, vec![]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_inv_with_custom_timeout() {
        let ctx = create_test_context_with_capability().await;

        let mut node = Node {
            id: 1,
            op_type: AISOperationType::Inv,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };

        node.attributes
            .insert("capability".to_string(), Value::String("echo".to_string()));
        node.attributes.insert(
            "timeout_ms".to_string(),
            Value::Number(apxm_core::types::values::Number::Integer(5000)),
        );
        node.attributes
            .insert("arg_message".to_string(), Value::String("Test".to_string()));

        let result = execute(&ctx, &node, vec![]).await.unwrap();
        assert_eq!(result.as_string().map(|s| s.as_str()), Some("Echo: Test"));
    }
}
