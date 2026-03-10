//! COMMUNICATE operation - Inter-agent communication
//!
//! Supports three dispatch modes:
//!   - `local` (default): in-process sub-flow execution via FlowRegistry
//!   - `http`: POST to an external APXM agent's `/v1/receive` endpoint
//!   - `broadcast`: fan-out to ALL registered agents in FlowRegistry in parallel;
//!     returns an Array of all responses (non-fatal errors included as strings)
//!
//! The protocol is selected via the `protocol` node attribute.
//! For HTTP, `recipient` may be a full URL (`http://...`) or an agent name
//! looked up via `APXM_SERVER_URL/v1/agents/{name}`.
//!
//! For BROADCAST, `recipient` is ignored. The message is sent to every agent
//! currently registered in the FlowRegistry; results are collected in parallel.

use super::{ExecutionContext, Node, Result, Value, get_string_attribute};
use crate::aam::TransitionLabel;
use crate::executor::ExecutorEngine;
use apxm_core::constants::graph::attrs as graph_attrs;
use apxm_core::error::RuntimeError;

/// Well-known flow names tried in order when looking up a recipient agent.
const COMMUNICATE_FLOW_NAMES: &[&str] = &["communicate", "main"];

pub async fn execute(ctx: &ExecutionContext, node: &Node, inputs: Vec<Value>) -> Result<Value> {
    // Accept both the historical "target" attribute and the current "recipient"
    // emitted by the compiler.
    let recipient = get_string_attribute(node, graph_attrs::RECIPIENT)
        .or_else(|_| get_string_attribute(node, graph_attrs::TARGET))
        .unwrap_or_default();
    let protocol =
        get_string_attribute(node, graph_attrs::PROTOCOL).unwrap_or_else(|_| "local".to_string());
    let message = inputs.first().cloned().unwrap_or(Value::Null);

    match protocol.as_str() {
        "http" | "https" => return execute_http(ctx, node, &recipient, message).await,
        "broadcast" => return execute_broadcast(ctx, node, message).await,
        _ => {
            // local — require recipient
            if recipient.is_empty() {
                return Err(RuntimeError::Operation {
                    op_type: node.op_type,
                    message: "COMMUNICATE requires 'recipient' attribute for local protocol"
                        .to_string(),
                });
            }
        }
    }

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
            message: format!("Communication with agent '{}' failed: {}", recipient, e),
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

// ─── Broadcast dispatch ────────────────────────────────────────────────────

/// Fan-out COMMUNICATE to ALL agents registered in the FlowRegistry in parallel.
///
/// Dispatches to every unique agent name that has a known "communicate" or "main"
/// flow.  Each agent is called concurrently via `tokio::spawn`.  Non-fatal errors
/// are captured as `Value::String("<agent>: <error>")` so the caller receives a
/// complete picture rather than a partial failure.
///
/// Returns `Value::Array` of all responses (one per agent, in arbitrary order).
async fn execute_broadcast(ctx: &ExecutionContext, _node: &Node, message: Value) -> Result<Value> {
    // Collect unique agent names that have a usable flow
    let all_flows = ctx.flow_registry.list_flows();
    let agent_names: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        let mut agents = Vec::new();
        for (agent_name, flow_name) in &all_flows {
            if COMMUNICATE_FLOW_NAMES.contains(&flow_name.as_str())
                && seen.insert(agent_name.clone())
            {
                agents.push(agent_name.clone());
            }
        }
        agents
    };

    if agent_names.is_empty() {
        tracing::warn!(
            execution_id = %ctx.execution_id,
            "BROADCAST: no agents with communicate/main flows registered"
        );
        return Ok(Value::Array(vec![]));
    }

    tracing::info!(
        execution_id = %ctx.execution_id,
        agents = agent_names.len(),
        "COMMUNICATE BROADCAST fan-out starting"
    );

    // Build a fake single-recipient node for re-use in the local path
    let mut handles = Vec::with_capacity(agent_names.len());
    for agent_name in &agent_names {
        // Find the sub-DAG
        let sub_dag = {
            let mut found = None;
            for flow_name in COMMUNICATE_FLOW_NAMES {
                if let Some(dag) = ctx.flow_registry.get_flow(agent_name, flow_name) {
                    found = Some(dag);
                    break;
                }
            }
            match found {
                Some(dag) => dag,
                None => continue,
            }
        };

        // Build a child context per recipient
        let child_ctx = ctx
            .child()
            .with_metadata("parent_execution_id".to_string(), ctx.execution_id.clone())
            .with_metadata("communicate_sender".to_string(), ctx.execution_id.clone())
            .with_metadata("communicate_recipient".to_string(), agent_name.clone())
            .with_metadata("communicate_mode".to_string(), "broadcast".to_string());

        let msg = message.clone();
        let agent = agent_name.clone();
        let dag_clone = (*sub_dag).clone();

        handles.push(tokio::spawn(async move {
            // Inject message into child STM
            let _ = child_ctx
                .memory
                .write(
                    crate::memory::MemorySpace::Stm,
                    "_communicate_message".to_string(),
                    msg,
                )
                .await;

            let engine = ExecutorEngine::new(child_ctx);
            match engine.execute_dag(dag_clone).await {
                Ok(result) => {
                    let response = result
                        .results
                        .values()
                        .find(|v| !matches!(v, Value::Null))
                        .cloned()
                        .unwrap_or(Value::Null);
                    (agent, Ok(response))
                }
                Err(e) => (agent, Err(e)),
            }
        }));
    }

    // Collect results — non-fatal errors become string values
    let mut responses = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok((agent, Ok(val))) => {
                tracing::debug!(agent = %agent, "BROADCAST response received");
                responses.push(val);
            }
            Ok((agent, Err(e))) => {
                tracing::warn!(agent = %agent, error = %e, "BROADCAST agent returned error");
                responses.push(Value::String(format!("{}: {}", agent, e)));
            }
            Err(join_err) => {
                tracing::error!(error = %join_err, "BROADCAST task panicked");
                responses.push(Value::String(format!("panic: {}", join_err)));
            }
        }
    }

    tracing::info!(
        execution_id = %ctx.execution_id,
        responses = responses.len(),
        "COMMUNICATE BROADCAST completed"
    );

    Ok(Value::Array(responses))
}

// ─── HTTP dispatch ─────────────────────────────────────────────────────────

/// Dispatch COMMUNICATE over HTTP to an external APXM agent.
///
/// `recipient` may be:
///   - A full URL: `http://host:port` — used directly as base URL.
///   - An agent name: looked up via `APXM_SERVER_URL/v1/agents/{name}`.
async fn execute_http(
    ctx: &ExecutionContext,
    node: &Node,
    recipient: &str,
    message: Value,
) -> Result<Value> {
    let op_err = |msg: String| RuntimeError::Operation {
        op_type: node.op_type,
        message: msg,
    };

    // Resolve base URL
    let base_url = if recipient.starts_with("http://") || recipient.starts_with("https://") {
        recipient.to_string()
    } else {
        // Name-based lookup via APXM server agent registry
        let server_url = std::env::var("APXM_SERVER_URL")
            .unwrap_or_else(|_| "http://localhost:18800".to_string());
        let lookup_url = format!("{}/v1/agents/{}", server_url, recipient);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| op_err(format!("Failed to build HTTP client: {e}")))?;
        let resp = client
            .get(&lookup_url)
            .send()
            .await
            .map_err(|e| op_err(format!("Agent registry lookup failed: {e}")))?;
        if !resp.status().is_success() {
            return Err(op_err(format!(
                "Agent '{}' not found in registry ({})",
                recipient,
                resp.status()
            )));
        }
        let agent_info: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| op_err(format!("Failed to parse agent info: {e}")))?;
        agent_info["url"]
            .as_str()
            .ok_or_else(|| op_err(format!("Agent '{}' has no 'url' field", recipient)))?
            .to_string()
    };

    // Serialize message
    let msg_json = message
        .to_json()
        .map_err(|e| op_err(format!("Failed to serialize message: {e}")))?;

    // POST to /v1/receive
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| op_err(format!("Failed to build HTTP client: {e}")))?;

    let receive_url = format!("{}/v1/receive", base_url.trim_end_matches('/'));
    tracing::info!(
        execution_id = %ctx.execution_id,
        recipient = %recipient,
        url = %receive_url,
        "COMMUNICATE HTTP dispatch"
    );

    let resp = client
        .post(&receive_url)
        .json(&serde_json::json!({
            "from": ctx.execution_id,
            "message": msg_json
        }))
        .send()
        .await
        .map_err(|e| op_err(format!("HTTP COMMUNICATE to '{}' failed: {e}", recipient)))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(op_err(format!(
            "HTTP COMMUNICATE '{}' returned {}: {}",
            recipient, status, text
        )));
    }

    let result: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| op_err(format!("Failed to parse COMMUNICATE response: {e}")))?;

    Value::try_from(result).map_err(|e| op_err(format!("Failed to convert response: {e}")))
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
