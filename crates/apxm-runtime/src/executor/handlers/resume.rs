//! RESUME operation — restore execution from a PAUSE checkpoint.
//!
//! Polls the APXM server for the named checkpoint's resumed state. If the
//! checkpoint has been resumed by a human (via `POST /v1/checkpoints/{id}/resume`),
//! this handler restores any STM snapshot that was stored during PAUSE, and
//! returns the `human_input` value for downstream nodes to use.
//!
//! ## AIS usage
//! ```ais
//! pause(checkpoint: "review_plan", message: "Please review: " + plan) -> _
//! resume(checkpoint: "review_plan") -> human_input
//! ask("Implement approved plan. Human notes: " + human_input) -> code
//! ```
//!
//! ## Polling behaviour
//! RESUME polls the checkpoint endpoint up to `poll_max_attempts` times
//! (default 60) with `poll_interval_ms` (default 5000 ms = 5 s) between
//! attempts. Total default wait: 5 min. Adjust via node attributes if needed.

use super::{ExecutionContext, Node, Result, Value};
use apxm_core::constants::defaults;
use apxm_core::constants::graph::attrs as graph_attrs;
use apxm_core::error::RuntimeError;
use std::time::Duration;

/// Default number of polling attempts (60 × 5 s = 5 min total).
const DEFAULT_POLL_ATTEMPTS: u64 = 60;
/// Default interval between polling attempts (milliseconds).
const DEFAULT_POLL_INTERVAL_MS: u64 = 5_000;
/// Timeout for each individual HTTP request (seconds).
const HTTP_TIMEOUT_SECS: u64 = 10;

pub async fn execute(ctx: &ExecutionContext, node: &Node, _inputs: Vec<Value>) -> Result<Value> {
    let checkpoint_id = node
        .attributes
        .get(graph_attrs::CHECKPOINT)
        .and_then(|v| v.as_string().map(|s| s.to_string()))
        .ok_or_else(|| RuntimeError::Operation {
            op_type: node.op_type,
            message: "RESUME requires a `checkpoint` attribute".to_string(),
        })?;

    let server_url = node
        .attributes
        .get(graph_attrs::SERVER_URL)
        .and_then(|v| v.as_string().map(|s| s.to_string()))
        .or_else(|| ctx.metadata.get("apxm_server_url").cloned())
        .unwrap_or_else(|| {
            std::env::var("APXM_SERVER_URL")
                .unwrap_or_else(|_| defaults::DEFAULT_SERVER_URL.to_string())
        });

    let poll_attempts = node
        .attributes
        .get(graph_attrs::POLL_MAX_ATTEMPTS)
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_POLL_ATTEMPTS);

    let poll_interval_ms = node
        .attributes
        .get(graph_attrs::POLL_INTERVAL_MS)
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_POLL_INTERVAL_MS);

    let url = format!(
        "{}/v1/checkpoints/{}",
        server_url.trim_end_matches('/'),
        checkpoint_id
    );

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
        .build()
        .map_err(|e| RuntimeError::Operation {
            op_type: node.op_type,
            message: format!("RESUME: failed to build HTTP client: {}", e),
        })?;

    tracing::info!(
        execution_id = %ctx.execution_id,
        checkpoint_id = %checkpoint_id,
        poll_attempts,
        poll_interval_ms,
        "RESUME: polling for checkpoint completion"
    );

    for attempt in 0..poll_attempts {
        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| RuntimeError::Operation {
                op_type: node.op_type,
                message: format!(
                    "RESUME: HTTP GET {} failed (attempt {}): {}",
                    url,
                    attempt + 1,
                    e
                ),
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(RuntimeError::Operation {
                op_type: node.op_type,
                message: format!(
                    "RESUME: server returned {} for checkpoint '{}': {}",
                    status, checkpoint_id, text
                ),
            });
        }

        let body: serde_json::Value = resp.json().await.map_err(|e| RuntimeError::Operation {
            op_type: node.op_type,
            message: format!("RESUME: failed to parse checkpoint response: {}", e),
        })?;

        let status = body
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("pending");

        match status {
            "resumed" => {
                tracing::info!(
                    execution_id = %ctx.execution_id,
                    checkpoint_id = %checkpoint_id,
                    attempt,
                    "RESUME: checkpoint has been resumed by human"
                );

                // Restore STM snapshot if one was stored by PAUSE.
                restore_stm_snapshot(ctx, &checkpoint_id).await;

                // Extract and return human_input.
                let human_input = body
                    .get("human_input")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                return Value::try_from(human_input).map_err(|e| RuntimeError::Operation {
                    op_type: node.op_type,
                    message: format!("RESUME: failed to convert human_input to Value: {}", e),
                });
            }
            "expired" => {
                return Err(RuntimeError::Operation {
                    op_type: node.op_type,
                    message: format!("RESUME: checkpoint '{}' has expired", checkpoint_id),
                });
            }
            _ => {
                // Still pending — wait and retry
                tracing::debug!(
                    execution_id = %ctx.execution_id,
                    checkpoint_id = %checkpoint_id,
                    attempt,
                    "RESUME: checkpoint still pending, waiting {}ms",
                    poll_interval_ms
                );
                tokio::time::sleep(Duration::from_millis(poll_interval_ms)).await;
            }
        }
    }

    Err(RuntimeError::Operation {
        op_type: node.op_type,
        message: format!(
            "RESUME: checkpoint '{}' was not resumed within {} attempts ({} ms each)",
            checkpoint_id, poll_attempts, poll_interval_ms
        ),
    })
}

/// Restore STM state from a snapshot stored by PAUSE.
async fn restore_stm_snapshot(ctx: &ExecutionContext, checkpoint_id: &str) {
    let snapshot_key = format!("_checkpoint_snapshot:{}", checkpoint_id);
    if let Ok(Some(snapshot)) = ctx
        .memory
        .read(crate::memory::MemorySpace::Stm, &snapshot_key)
        .await
    {
        if let Value::Object(entries) = snapshot {
            for (key, value) in entries {
                // Restore all snapshot entries except the execution_id (keep current).
                if key != "_execution_id" {
                    let _ = ctx
                        .memory
                        .write(crate::memory::MemorySpace::Stm, key, value)
                        .await;
                }
            }
        }
        // Clean up the snapshot key
        let _ = ctx
            .memory
            .delete(crate::memory::MemorySpace::Stm, &snapshot_key)
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aam::Aam;
    use crate::capability::CapabilitySystem;
    use crate::executor::context::ExecutionContext;
    use crate::memory::{MemoryConfig, MemorySystem};
    use apxm_backends::LLMRegistry;
    use apxm_core::types::{execution::NodeMetadata, operations::AISOperationType};
    use std::collections::HashMap;
    use std::sync::Arc;

    async fn make_ctx() -> ExecutionContext {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());
        ExecutionContext::new(memory, llm_registry, capability_system, Aam::new())
    }

    #[tokio::test]
    async fn test_resume_missing_checkpoint_attribute() {
        let ctx = make_ctx().await;
        let node = apxm_core::types::execution::Node {
            id: 1,
            op_type: AISOperationType::Resume,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };
        let result = execute(&ctx, &node, vec![]).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("checkpoint"));
    }

    #[tokio::test]
    async fn test_resume_server_unavailable() {
        let ctx = make_ctx().await;
        let mut node = apxm_core::types::execution::Node {
            id: 1,
            op_type: AISOperationType::Resume,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };
        node.attributes.insert(
            "checkpoint".to_string(),
            Value::String("test-cp".to_string()),
        );
        node.attributes.insert(
            "server_url".to_string(),
            Value::String("http://localhost:19999".to_string()),
        );
        // Use very small poll count so test doesn't hang
        node.attributes.insert(
            "poll_max_attempts".to_string(),
            Value::String("1".to_string()),
        );
        node.attributes.insert(
            "poll_interval_ms".to_string(),
            Value::String("1".to_string()),
        );
        let result = execute(&ctx, &node, vec![]).await;
        assert!(result.is_err());
    }
}
