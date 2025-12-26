//! Executor engine - Main orchestrator for DAG execution

use super::{ExecutionContext, Result, dispatcher::OperationDispatcher};
use apxm_core::types::{
    execution::{ExecutionDag, ExecutionStats, Node, NodeStatus, OpStatus},
    values::Value,
};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;

/// Executor engine orchestrates DAG execution
pub struct ExecutorEngine {
    context: ExecutionContext,
}

impl ExecutorEngine {
    /// Create a new executor engine with the given context
    pub fn new(context: ExecutionContext) -> Self {
        Self { context }
    }

    /// Execute an operation using the provided context
    pub async fn execute_with_context(
        &self,
        node: &Node,
        inputs: Vec<Value>,
        ctx: &ExecutionContext,
    ) -> Result<OperationOutcome> {
        let value = OperationDispatcher::dispatch(ctx, node, inputs).await?;
        Ok(OperationOutcome { value })
    }

    /// Execute a complete DAG
    ///
    /// This is a simplified synchronous executor for now.
    /// A production implementation would use the dataflow scheduler.
    pub async fn execute_dag(&self, dag: ExecutionDag) -> Result<ExecutionResult> {
        tracing::info!(
            execution_id = %self.context.execution_id,
            nodes = dag.nodes.len(),
            "Starting DAG execution"
        );

        let start_time = std::time::Instant::now();
        let node_statuses = Arc::new(RwLock::new(HashMap::<u64, NodeStatus>::new()));
        let node_results = Arc::new(RwLock::new(HashMap::<u64, Value>::new()));

        // Simple topological execution for now
        // TODO: Integrate with dataflow scheduler for parallel execution
        for node in &dag.nodes {
            let node_start = std::time::Instant::now();

            // Mark as running
            {
                let mut statuses = node_statuses.write().await;
                statuses.insert(
                    node.id,
                    NodeStatus {
                        node_id: node.id,
                        status: OpStatus::Running,
                        retries: 0,
                        last_error: None,
                        started_at_ms: Some(node_start.elapsed().as_millis()),
                        finished_at_ms: None,
                        duration_ms: None,
                    },
                );
            }

            // Gather inputs from dependencies
            let mut inputs = Vec::new();
            for input_token_id in &node.input_tokens {
                let results = node_results.read().await;
                if let Some(value) = results.get(input_token_id) {
                    inputs.push(value.clone());
                }
            }

            // Execute operation
            let result = OperationDispatcher::dispatch(&self.context, node, inputs).await;

            let node_end = node_start.elapsed();

            // Store result and update status
            match result {
                Ok(value) => {
                    // Store result for output tokens
                    {
                        let mut results = node_results.write().await;
                        for output_token_id in &node.output_tokens {
                            results.insert(*output_token_id, value.clone());
                        }
                    }

                    // Mark as completed
                    {
                        let mut statuses = node_statuses.write().await;
                        statuses.insert(
                            node.id,
                            NodeStatus {
                                node_id: node.id,
                                status: OpStatus::Completed,
                                retries: 0,
                                last_error: None,
                                started_at_ms: Some(node_start.elapsed().as_millis()),
                                finished_at_ms: Some(node_end.as_millis()),
                                duration_ms: Some(node_end.as_millis()),
                            },
                        );
                    }
                }
                Err(e) => {
                    // Mark as failed
                    {
                        let mut statuses = node_statuses.write().await;
                        statuses.insert(
                            node.id,
                            NodeStatus {
                                node_id: node.id,
                                status: OpStatus::Failed,
                                retries: 0,
                                last_error: Some(e.to_string()),
                                started_at_ms: Some(node_start.elapsed().as_millis()),
                                finished_at_ms: Some(node_end.as_millis()),
                                duration_ms: Some(node_end.as_millis()),
                            },
                        );
                    }

                    tracing::error!(
                        execution_id = %self.context.execution_id,
                        node_id = node.id,
                        error = %e,
                        "Node execution failed"
                    );

                    return Err(e);
                }
            }
        }

        // Collect final results from exit nodes
        let mut final_results = HashMap::new();
        {
            let results = node_results.read().await;
            for exit_node_id in &dag.exit_nodes {
                // Find the output token for this exit node
                if let Some(node) = dag.nodes.iter().find(|n| n.id == *exit_node_id) {
                    for output_token_id in &node.output_tokens {
                        if let Some(value) = results.get(output_token_id) {
                            final_results.insert(*output_token_id, value.clone());
                        }
                    }
                }
            }
        }

        // Build execution stats
        let statuses = node_statuses.read().await;
        let stats = ExecutionStats {
            executed_nodes: statuses.len(),
            failed_nodes: statuses
                .values()
                .filter(|s| matches!(s.status, OpStatus::Failed))
                .count(),
            duration_ms: start_time.elapsed().as_millis(),
            node_statuses: statuses.values().cloned().collect(),
        };

        tracing::info!(
            execution_id = %self.context.execution_id,
            duration_ms = stats.duration_ms,
            executed = stats.executed_nodes,
            failed = stats.failed_nodes,
            "DAG execution completed"
        );

        Ok(ExecutionResult {
            results: final_results,
            stats,
        })
    }

    /// Execute a single node (for testing/debugging)
    pub async fn execute_node(
        &self,
        node: &apxm_core::types::execution::Node,
        inputs: Vec<Value>,
    ) -> Result<Value> {
        let outcome = self
            .execute_with_context(node, inputs, &self.context)
            .await?;
        Ok(outcome.value)
    }
}

/// Outcome of a single operation execution
#[derive(Debug, Clone)]
pub struct OperationOutcome {
    pub value: Value,
}

/// Result of DAG execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Final results mapped by token ID
    pub results: HashMap<u64, Value>,
    /// Execution statistics
    pub stats: ExecutionStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilitySystem;
    use crate::memory::{MemoryConfig, MemorySystem};
    use apxm_core::types::{
        execution::Node, execution::NodeMetadata, operations::AISOperationType,
    };
    use std::sync::Arc;

    #[tokio::test]
    async fn test_executor_single_node() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(apxm_backends::LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());
        let ctx = ExecutionContext::new(
            memory,
            llm_registry,
            capability_system,
            crate::aam::Aam::new(),
        );
        let engine = ExecutorEngine::new(ctx);

        // Create a CONST_STR node
        let mut node = Node {
            id: 1,
            op_type: AISOperationType::ConstStr,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![],
            metadata: NodeMetadata::default(),
        };
        node.attributes.insert(
            "value".to_string(),
            Value::String("Hello, World!".to_string()),
        );

        let result = engine.execute_node(&node, vec![]).await.unwrap();
        assert_eq!(result, Value::String("Hello, World!".to_string()));
    }

    #[tokio::test]
    async fn test_executor_simple_dag() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(apxm_backends::LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());
        let ctx = ExecutionContext::new(
            memory,
            llm_registry,
            capability_system,
            crate::aam::Aam::new(),
        );
        let engine = ExecutorEngine::new(ctx);

        // Create a simple DAG: CONST_STR -> UMEM
        let mut const_node = Node {
            id: 1,
            op_type: AISOperationType::ConstStr,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };
        const_node
            .attributes
            .insert("value".to_string(), Value::String("test_value".to_string()));

        let mut umem_node = Node {
            id: 2,
            op_type: AISOperationType::UMem,
            attributes: HashMap::new(),
            input_tokens: vec![100],
            output_tokens: vec![101],
            metadata: NodeMetadata::default(),
        };
        umem_node
            .attributes
            .insert("key".to_string(), Value::String("test_key".to_string()));

        let dag = ExecutionDag {
            nodes: vec![const_node, umem_node],
            edges: vec![],
            entry_nodes: vec![1],
            exit_nodes: vec![2],
            metadata: Default::default(),
        };

        let result = engine.execute_dag(dag).await.unwrap();
        assert_eq!(result.stats.executed_nodes, 2);
        assert_eq!(result.stats.failed_nodes, 0);
    }
}
