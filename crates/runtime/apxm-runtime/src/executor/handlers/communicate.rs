//! COMMUNICATE operation - Inter-agent communication
//!
//! Looks up the target agent by name in the FlowRegistry and executes its
//! default "communicate" (or "main") flow, passing the message as input.
//! If no flow is registered for the target agent, returns a clear error.

use super::{ExecutionContext, Node, Result, Value, get_string_attribute};
use crate::aam::TransitionLabel;
use crate::executor::ExecutorEngine;
use apxm_core::error::RuntimeError;

/// Well-known flow names tried in order when looking up a recipient agent.
const COMMUNICATE_FLOW_NAMES: &[&str] = &["communicate", "main"];

pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // Accept both the historical "target" attribute and the current "recipient"
    // emitted by the compiler.
    let recipient = get_string_attribute(node, "recipient")
        .or_else(|_| get_string_attribute(node, "target"))?;
    let message = inputs.first().cloned().unwrap_or(Value::Null);

    tracing::info!(
        execution_id = %ctx.execution_id,
        recipient = %recipient,
        "Executing COMMUNICATE operation"
    );

    // Record the outgoing message in AAM beliefs for observability
    let label = TransitionLabel::Custom(format!("communicate:{}", recipient));
    ctx.aam.set_belief(
        format!("_pending_communicate:{}", recipient),
        Value::Object(
            vec![
                ("recipient".to_string(), Value::String(recipient.clone())),
                ("message".to_string(), message.clone()),
            ]
            .into_iter()
            .collect(),
        ),
        label,
    );

    // Look up the target agent's flow in the FlowRegistry.
    // Try well-known flow names in order: "communicate", then "main".
    let sub_dag = {
        let mut found = None;
        for flow_name in COMMUNICATE_FLOW_NAMES {
            if let Some(dag) = ctx.flow_registry.get_flow(&recipient, flow_name) {
                found = Some(dag);
                break;
            }
        }
        match found {
            Some(dag) => dag,
            None => {
                let available = ctx.flow_registry.flows_for_agent(&recipient);
                let hint = if available.is_empty() {
                    let all_flows = ctx.flow_registry.list_flows();
                    if all_flows.is_empty() {
                        "No agents are registered in the flow registry.".to_string()
                    } else {
                        format!(
                            "Agent '{}' not found. Registered agents: {}",
                            recipient,
                            all_flows
                                .iter()
                                .map(|(a, f)| format!("{}.{}", a, f))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                } else {
                    format!(
                        "Agent '{}' has no 'communicate' or 'main' flow. Available flows: {}",
                        recipient,
                        available.join(", ")
                    )
                };

                return Err(RuntimeError::Operation {
                    op_type: node.op_type,
                    message: hint,
                });
            }
        }
    };

    tracing::debug!(
        recipient = %recipient,
        nodes = sub_dag.nodes.len(),
        "Found flow for recipient agent, executing sub-DAG"
    );

    // Create a child context for the sub-flow execution
    let child_ctx = ctx
        .child()
        .with_metadata("parent_execution_id".to_string(), ctx.execution_id.clone())
        .with_metadata("communicate_sender".to_string(), ctx.execution_id.clone())
        .with_metadata("communicate_recipient".to_string(), recipient.clone());

    // Inject the message into STM so the sub-flow can access it
    let _ = child_ctx
        .memory
        .write(
            crate::memory::MemorySpace::Stm,
            "_communicate_message".to_string(),
            message.clone(),
        )
        .await;

    // Execute the sub-flow DAG
    let engine = ExecutorEngine::new(child_ctx);
    let dag_to_execute = (*sub_dag).clone();

    let result = engine.execute_dag(dag_to_execute).await.map_err(|e| {
        tracing::error!(
            recipient = %recipient,
            error = %e,
            "Inter-agent communication failed"
        );
        RuntimeError::Operation {
            op_type: node.op_type,
            message: format!(
                "Communication with agent '{}' failed: {}",
                recipient, e
            ),
        }
    })?;

    // Clear the pending belief
    ctx.aam.set_belief(
        format!("_pending_communicate:{}", recipient),
        Value::Null,
        TransitionLabel::Custom(format!("communicate_completed:{}", recipient)),
    );

    // Extract the response from the sub-flow's exit nodes
    let response = result
        .results
        .values()
        .find(|v| !matches!(v, Value::Null))
        .cloned()
        .unwrap_or(Value::Null);

    tracing::info!(
        recipient = %recipient,
        duration_ms = result.stats.duration_ms,
        executed_nodes = result.stats.executed_nodes,
        "COMMUNICATE completed successfully"
    );

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilitySystem;
    use crate::capability::flow_registry::FlowRegistry;
    use crate::memory::{MemoryConfig, MemorySystem};
    use apxm_backends::LLMRegistry;
    use apxm_core::types::{
        execution::{ExecutionDag, NodeMetadata},
        operations::AISOperationType,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_echo_dag() -> ExecutionDag {
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
            Value::String("ack from agent".to_string()),
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
    async fn test_communicate_with_registered_agent() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());
        let flow_registry = Arc::new(FlowRegistry::new());

        // Register a "communicate" flow for "PeerAgent"
        flow_registry.register_flow("PeerAgent", "communicate", create_echo_dag());

        let ctx = ExecutionContext::new(
            memory,
            llm_registry,
            capability_system,
            crate::aam::Aam::new(),
        );
        let ctx = ExecutionContext {
            flow_registry,
            ..ctx
        };

        let mut node = apxm_core::types::execution::Node {
            id: 1,
            op_type: AISOperationType::Communicate,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };
        node.attributes.insert(
            "recipient".to_string(),
            Value::String("PeerAgent".to_string()),
        );

        let result = execute(&ctx, &node, vec![Value::String("hello".to_string())])
            .await
            .unwrap();
        assert_eq!(result, Value::String("ack from agent".to_string()));
    }

    #[tokio::test]
    async fn test_communicate_agent_not_found() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());

        let ctx = ExecutionContext::new(
            memory,
            llm_registry,
            capability_system,
            crate::aam::Aam::new(),
        );

        let mut node = apxm_core::types::execution::Node {
            id: 1,
            op_type: AISOperationType::Communicate,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };
        node.attributes.insert(
            "recipient".to_string(),
            Value::String("NonExistent".to_string()),
        );

        let result = execute(&ctx, &node, vec![]).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No agents"));
    }
}
