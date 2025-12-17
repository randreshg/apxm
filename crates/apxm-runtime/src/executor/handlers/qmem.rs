//! QMEM operation - Query memory

use super::{
    ExecutionContext, Node, Result, Value, get_optional_string_attribute, get_string_attribute,
};
use crate::{
    aam::{STAGED_BELIEF_PREFIX, TransitionLabel},
    memory::MemorySpace,
};

/// Execute QMEM operation - Query memory from specified tier
pub async fn execute(ctx: &ExecutionContext, node: &Node, _inputs: Vec<Value>) -> Result<Value> {
    let query = get_string_attribute(node, "query")?;
    let memory_tier = get_optional_string_attribute(node, "memory_tier")?;
    let limit = node
        .attributes
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;

    // Determine memory space
    let space = match memory_tier.as_deref() {
        Some("stm") | Some("STM") => MemorySpace::Stm,
        Some("ltm") | Some("LTM") => MemorySpace::Ltm,
        Some("episodic") | Some("EPISODIC") => MemorySpace::Episodic,
        None => MemorySpace::Stm, // Default to STM
        Some(other) => {
            return Err(apxm_core::error::RuntimeError::Memory {
                message: format!("Unknown memory tier: {}", other),
                space: Some(other.to_string()),
            });
        }
    };

    // Search memory
    let results = ctx.memory.search(space, &query, limit).await?;

    // Convert search results to Value::Array
    let values: Vec<Value> = results
        .into_iter()
        .map(|r| {
            // Create object with key and value
            let mut obj = std::collections::HashMap::new();
            obj.insert("key".to_string(), Value::String(r.key));
            obj.insert("value".to_string(), r.value);
            obj.insert(
                "score".to_string(),
                Value::Number(apxm_core::types::values::Number::Float(r.score)),
            );
            Value::Object(obj)
        })
        .collect();

    let array_value = Value::Array(values.clone());

    // Stage results as a belief so downstream ops can refer to them.
    let staging_id = node
        .attributes
        .get("staging_id")
        .and_then(|v| v.as_string().map(|s| s.to_string()))
        .unwrap_or_else(|| format!("{}:{}", ctx.execution_id, node.id));
    let stage_key = format!("{}{}", STAGED_BELIEF_PREFIX, staging_id);
    ctx.aam.set_belief(
        stage_key,
        array_value.clone(),
        TransitionLabel::operation(node.id, format!("{:?}", node.op_type)),
    );

    Ok(array_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilitySystem;
    use crate::memory::{MemoryConfig, MemorySystem};
    use apxm_core::types::operations::AISOperationType;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_qmem_stm() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(apxm_models::registry::LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());
        let ctx = ExecutionContext::new(
            memory.clone(),
            llm_registry,
            capability_system,
            crate::aam::Aam::new(),
        );

        // Populate STM
        memory
            .write(
                MemorySpace::Stm,
                "user:1".to_string(),
                Value::String("alice".to_string()),
            )
            .await
            .unwrap();
        memory
            .write(
                MemorySpace::Stm,
                "user:2".to_string(),
                Value::String("bob".to_string()),
            )
            .await
            .unwrap();

        // Create QMEM node
        let mut node = Node {
            id: 1,
            op_type: AISOperationType::QMem,
            attributes: std::collections::HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![],
            metadata: apxm_core::types::execution::NodeMetadata::default(),
        };
        node.attributes
            .insert("query".to_string(), Value::String("user".to_string()));
        node.attributes
            .insert("memory_tier".to_string(), Value::String("stm".to_string()));

        let result = execute(&ctx, &node, vec![]).await.unwrap();

        // Should return array of results
        if let Value::Array(arr) = result {
            assert_eq!(arr.len(), 2);
        } else {
            panic!("Expected array result");
        }
    }
}
