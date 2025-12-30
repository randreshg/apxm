//! FLOW_CALL operation - Call a flow on another agent

use std::future::Future;
use std::pin::Pin;

use super::{ExecutionContext, Node, Result, Value, get_string_attribute};
use crate::aam::TransitionLabel;
use crate::executor::ExecutorEngine;
use apxm_core::error::RuntimeError;

/// Maximum recursion depth for flow calls to prevent stack overflow
const MAX_FLOW_CALL_DEPTH: usize = 100;

/// Execute a flow call operation
///
/// Invokes a flow on another agent, passing arguments and receiving a result.
/// This enables cross-agent communication and coordination.
///
/// The flow call:
/// 1. Looks up the target flow in the FlowRegistry
/// 2. Creates a child execution context for the sub-flow
/// 3. Executes the sub-flow DAG
/// 4. Returns the result from the sub-flow's exit node
///
/// If the target flow is not registered, returns an error.
///
/// Note: This function returns a boxed future to break the async recursion chain
/// that occurs when a flow calls another flow (flow_call -> execute_dag -> dispatch -> flow_call).
pub fn execute<'a>(
    ctx: &'a ExecutionContext,
    node: &'a Node,
    inputs: Vec<Value>,
) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
    Box::pin(execute_impl(ctx, node, inputs))
}

async fn execute_impl(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // Get agent name
    let agent_name = get_string_attribute(node, "agent_name")?;

    // Get flow name
    let flow_name = get_string_attribute(node, "flow_name")?;

    tracing::info!(
        agent = %agent_name,
        flow = %flow_name,
        num_args = inputs.len(),
        "Executing flow call"
    );

    // Check recursion depth via metadata
    let current_depth: usize = ctx
        .metadata
        .get("flow_call_depth")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if current_depth >= MAX_FLOW_CALL_DEPTH {
        return Err(RuntimeError::Capability {
            capability: format!("flow_call:{}:{}", agent_name, flow_name),
            message: format!(
                "Maximum flow call depth ({}) exceeded. Possible infinite recursion.",
                MAX_FLOW_CALL_DEPTH
            ),
        });
    }

    // Record the flow call in AAM beliefs for tracking
    let call_request = Value::Object(
        vec![
            ("type".to_string(), Value::String("flow_call".to_string())),
            ("agent".to_string(), Value::String(agent_name.clone())),
            ("flow".to_string(), Value::String(flow_name.clone())),
            ("args".to_string(), Value::Array(inputs.clone())),
            (
                "execution_id".to_string(),
                Value::String(ctx.execution_id.clone()),
            ),
        ]
        .into_iter()
        .collect(),
    );

    let label = TransitionLabel::Custom(format!("flow_call:{}:{}", agent_name, flow_name));
    ctx.aam.set_belief(
        format!("_pending_flow_call:{}:{}", agent_name, flow_name),
        call_request.clone(),
        label,
    );

    // Record in episodic memory for tracing
    ctx.memory
        .record_episode(
            format!("flow_call:{}:{}", agent_name, flow_name),
            call_request.clone(),
            ctx.execution_id.clone(),
        )
        .await
        .ok();

    // Look up the target flow in the FlowRegistry
    let sub_dag = match ctx.flow_registry.get_flow(&agent_name, &flow_name) {
        Some(dag) => dag,
        None => {
            // Flow not registered - return error with available flows
            let available = ctx.flow_registry.list_flows();
            let available_str = if available.is_empty() {
                "No flows registered".to_string()
            } else {
                available
                    .iter()
                    .map(|(a, f)| format!("{}.{}", a, f))
                    .collect::<Vec<_>>()
                    .join(", ")
            };

            return Err(RuntimeError::Capability {
                capability: format!("flow_call:{}:{}", agent_name, flow_name),
                message: format!(
                    "Flow '{}.{}' not found in registry. Available flows: {}",
                    agent_name, flow_name, available_str
                ),
            });
        }
    };

    tracing::debug!(
        agent = %agent_name,
        flow = %flow_name,
        nodes = sub_dag.nodes.len(),
        "Found flow in registry, executing sub-DAG"
    );

    // Create a child context for the sub-flow execution
    let child_ctx = ctx
        .child()
        .with_metadata(
            "parent_execution_id".to_string(),
            ctx.execution_id.clone(),
        )
        .with_metadata(
            "flow_call_depth".to_string(),
            (current_depth + 1).to_string(),
        )
        .with_metadata("target_agent".to_string(), agent_name.clone())
        .with_metadata("target_flow".to_string(), flow_name.clone());

    // Inject input arguments into the sub-DAG entry nodes
    // The inputs are provided to the first operations in the sub-DAG
    // For now, we store them in STM so the sub-flow can access them
    for (i, input) in inputs.iter().enumerate() {
        let _ = child_ctx
            .memory
            .write(
                crate::memory::MemorySpace::Stm,
                format!("_flow_arg_{}", i),
                input.clone(),
            )
            .await;
    }

    // Execute the sub-flow DAG
    let engine = ExecutorEngine::new(child_ctx);
    let dag_to_execute = (*sub_dag).clone();

    let result = engine.execute_dag(dag_to_execute).await.map_err(|e| {
        tracing::error!(
            agent = %agent_name,
            flow = %flow_name,
            error = %e,
            "Sub-flow execution failed"
        );
        RuntimeError::Capability {
            capability: format!("flow_call:{}:{}", agent_name, flow_name),
            message: format!("Sub-flow execution failed: {}", e),
        }
    })?;

    // Clear the pending flow call belief
    ctx.aam.set_belief(
        format!("_pending_flow_call:{}:{}", agent_name, flow_name),
        Value::Null,
        TransitionLabel::Custom(format!("flow_call_completed:{}:{}", agent_name, flow_name)),
    );

    // Extract the result from the sub-flow's exit nodes
    // If there are multiple exit nodes, we return the first non-null value
    let return_value = result
        .results
        .values()
        .find(|v| !matches!(v, Value::Null))
        .cloned()
        .unwrap_or(Value::Null);

    tracing::info!(
        agent = %agent_name,
        flow = %flow_name,
        duration_ms = result.stats.duration_ms,
        executed_nodes = result.stats.executed_nodes,
        "Flow call completed successfully"
    );

    Ok(return_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilitySystem;
    use crate::capability::flow_registry::FlowRegistry;
    use crate::memory::{MemoryConfig, MemorySystem};
    use apxm_core::types::{
        execution::{ExecutionDag, NodeMetadata},
        operations::AISOperationType,
    };
    use apxm_backends::LLMRegistry;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_simple_flow_dag() -> ExecutionDag {
        // Create a simple DAG that returns a constant string
        let mut const_node = apxm_core::types::execution::Node {
            id: 1,
            op_type: AISOperationType::ConstStr,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };
        const_node.attributes.insert(
            "value".to_string(),
            Value::String("hello from sub-flow".to_string()),
        );

        ExecutionDag {
            nodes: vec![const_node],
            edges: vec![],
            entry_nodes: vec![1],
            exit_nodes: vec![1],
            metadata: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_flow_call_with_registered_flow() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());
        let flow_registry = Arc::new(FlowRegistry::new());

        // Register a test flow
        flow_registry.register_flow("TestAgent", "greet", create_simple_flow_dag());

        let ctx = ExecutionContext::new(memory, llm_registry, capability_system, crate::aam::Aam::new());
        // Need to use with_inner_plan_support to set flow_registry, or modify the context
        let ctx = ExecutionContext {
            flow_registry,
            ..ctx
        };

        // Create a FLOW_CALL node
        let mut node = apxm_core::types::execution::Node {
            id: 1,
            op_type: AISOperationType::FlowCall,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };
        node.attributes.insert(
            "agent_name".to_string(),
            Value::String("TestAgent".to_string()),
        );
        node.attributes.insert(
            "flow_name".to_string(),
            Value::String("greet".to_string()),
        );

        let result = execute(&ctx, &node, vec![]).await.unwrap();
        assert_eq!(result, Value::String("hello from sub-flow".to_string()));
    }

    #[tokio::test]
    async fn test_flow_call_not_found() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());

        let ctx = ExecutionContext::new(memory, llm_registry, capability_system, crate::aam::Aam::new());

        // Create a FLOW_CALL node for a non-existent flow
        let mut node = apxm_core::types::execution::Node {
            id: 1,
            op_type: AISOperationType::FlowCall,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };
        node.attributes.insert(
            "agent_name".to_string(),
            Value::String("NonExistent".to_string()),
        );
        node.attributes.insert(
            "flow_name".to_string(),
            Value::String("missing".to_string()),
        );

        let result = execute(&ctx, &node, vec![]).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not found in registry"));
    }

    #[tokio::test]
    async fn test_flow_call_recursion_limit() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());
        let flow_registry = Arc::new(FlowRegistry::new());

        // Register a simple flow
        flow_registry.register_flow("TestAgent", "test", create_simple_flow_dag());

        let ctx = ExecutionContext::new(memory, llm_registry, capability_system, crate::aam::Aam::new());
        let ctx = ExecutionContext {
            flow_registry,
            ..ctx
        };
        // Simulate being at max depth
        let ctx = ctx.with_metadata("flow_call_depth".to_string(), MAX_FLOW_CALL_DEPTH.to_string());

        let mut node = apxm_core::types::execution::Node {
            id: 1,
            op_type: AISOperationType::FlowCall,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };
        node.attributes.insert(
            "agent_name".to_string(),
            Value::String("TestAgent".to_string()),
        );
        node.attributes.insert(
            "flow_name".to_string(),
            Value::String("test".to_string()),
        );

        let result = execute(&ctx, &node, vec![]).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Maximum flow call depth"));
    }
}
