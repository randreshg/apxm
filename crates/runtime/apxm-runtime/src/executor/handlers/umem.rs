//! UMEM operation - Update memory

use super::{
    ExecutionContext, Node, Result, Value, get_input, get_optional_string_attribute,
    get_string_attribute,
};
use crate::{aam::TransitionLabel, memory::MemorySpace};

/// Execute UMEM operation - Store value in specified memory tier
pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    let key = get_string_attribute(node, "key")?;
    let memory_tier = get_optional_string_attribute(node, "memory_tier")?;

    // Get value from first input or attribute
    let value = if !inputs.is_empty() {
        get_input(node, &inputs, 0)?
    } else {
        node.attributes.get("value").cloned().ok_or_else(|| {
            apxm_core::error::RuntimeError::Operation {
                op_type: node.op_type.clone(),
                message: "UMEM requires either input or value attribute".to_string(),
            }
        })?
    };

    // Determine memory space
    let space = match memory_tier.as_deref() {
        Some("stm") | Some("STM") => MemorySpace::Stm,
        Some("ltm") | Some("LTM") => MemorySpace::Ltm,
        None => MemorySpace::Stm, // Default to STM
        Some(other) => {
            return Err(apxm_core::error::RuntimeError::Memory {
                message: format!("Unknown memory tier: {}", other),
                space: Some(other.to_string()),
            });
        }
    };

    // Store in memory
    ctx.memory.write(space, key.clone(), value.clone()).await?;

    ctx.aam.set_belief(
        key,
        value.clone(),
        TransitionLabel::operation(node.id, format!("{:?}", node.op_type)),
    );

    // Return the stored value
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilitySystem;
    use crate::memory::{MemoryConfig, MemorySystem};
    use apxm_core::types::operations::AISOperationType;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_umem_stm() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(apxm_backends::LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());
        let ctx = ExecutionContext::new(
            memory.clone(),
            llm_registry,
            capability_system,
            crate::aam::Aam::new(),
        );

        // Create UMEM node
        let mut node = Node {
            id: 1,
            op_type: AISOperationType::UMem,
            attributes: std::collections::HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![],
            metadata: apxm_core::types::execution::NodeMetadata::default(),
        };
        node.attributes
            .insert("key".to_string(), Value::String("test_key".to_string()));
        node.attributes
            .insert("memory_tier".to_string(), Value::String("stm".to_string()));

        let input_value = Value::String("test_value".to_string());
        let result = execute(&ctx, &node, vec![input_value.clone()])
            .await
            .unwrap();

        assert_eq!(result, input_value);

        // Verify it was stored
        let stored = memory.read(MemorySpace::Stm, "test_key").await.unwrap();
        assert_eq!(stored, Some(input_value));
    }
}
