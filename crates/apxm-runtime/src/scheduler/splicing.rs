//! Dynamic DAG splicing for inner/outer plan unification
//!
//! This module implements the core capability for merging inner plan DAGs
//! into the live execution of outer plan DAGs, as described in the A-PXM paper.
//!
//! Key concepts:
//! - The outer plan is already executing
//! - An inner plan DAG is generated during execution (e.g., from PLAN/RSN)
//! - The inner DAG is spliced into the live execution
//! - The result is ONE unified DAG with merged dependencies

use apxm_core::{
    error::RuntimeError,
    log_debug, log_info, log_trace,
    types::{TokenId, Value, execution::ExecutionDag},
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use super::internal_state::{OpState, TokenState};
use super::queue::Priority;
use super::state::SchedulerState;
use crate::aam::effects::operation_effects;
use crate::executor::dag_splicer::{DagSplicer, SpliceResult};

/// Configuration for splicing an inner DAG into a running execution
pub struct SpliceConfig {
    /// The inner DAG to splice in
    pub inner_dag: ExecutionDag,

    /// Mapping from inner DAG entry node inputs to outer DAG output tokens
    ///
    /// This connects the outer plan to the inner plan.
    /// Key: Inner DAG token ID that needs a value
    /// Value: Outer DAG token ID that provides the value
    pub token_connections: HashMap<TokenId, TokenId>,

    /// Offset to add to all inner DAG node IDs to avoid conflicts
    ///
    /// If None, will be auto-calculated as max(outer_dag_node_ids) + 1
    pub node_id_offset: Option<u64>,

    /// Offset to add to all inner DAG token IDs to avoid conflicts
    ///
    /// If None, will be auto-calculated as max(outer_dag_token_ids) + 1
    pub token_id_offset: Option<u64>,
}

impl SchedulerState {
    /// Splice an inner DAG into the live execution
    ///
    /// This is the core operation for inner/outer plan unification.
    /// The inner DAG nodes are added to the scheduler state and
    /// integrated with the currently executing outer DAG.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying the inner DAG and how to connect it
    ///
    /// # Returns
    ///
    /// Mapping from original inner DAG token IDs to remapped token IDs
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Token connections reference non-existent tokens
    /// - Node/token ID conflicts occur
    /// - Inner DAG is malformed
    pub fn splice_dag(
        &self,
        config: SpliceConfig,
    ) -> Result<HashMap<TokenId, TokenId>, RuntimeError> {
        log_info!(
            "scheduler::splice",
            inner_nodes = config.inner_dag.nodes.len(),
            inner_edges = config.inner_dag.edges.len(),
            connections = config.token_connections.len(),
            "Splicing inner DAG into live execution"
        );

        // Calculate offsets to avoid ID conflicts
        let node_offset = config.node_id_offset.unwrap_or_else(|| {
            self.nodes
                .iter()
                .map(|entry| *entry.key())
                .max()
                .unwrap_or(0)
                + 1
        });

        let token_offset = config.token_id_offset.unwrap_or_else(|| {
            self.tokens
                .iter()
                .map(|entry| *entry.key())
                .max()
                .unwrap_or(0)
                + 1
        });

        log_debug!(
            "scheduler::splice",
            node_offset = node_offset,
            token_offset = token_offset,
            "Calculated ID offsets for splicing"
        );

        // Build token remapping
        let mut token_remap = HashMap::new();
        for (inner_token_id, outer_token_id) in &config.token_connections {
            token_remap.insert(*inner_token_id, *outer_token_id);
        }

        let mut remap_token = |token_id: TokenId| -> TokenId {
            *token_remap
                .entry(token_id)
                .or_insert_with(|| token_id + token_offset)
        };

        // Remap all inner DAG token IDs
        for node in &config.inner_dag.nodes {
            for &token_id in node.output_tokens.iter() {
                remap_token(token_id);
            }
            for &token_id in node.input_tokens.iter() {
                remap_token(token_id);
            }
        }

        // Remap nodes and add to scheduler state
        let mut ready_nodes = Vec::new();

        for node in &config.inner_dag.nodes {
            let original_inputs = node.input_tokens.clone();
            let original_outputs = node.output_tokens.clone();

            let mut node = node.clone();
            // Remap node ID
            let old_node_id = node.id;
            node.id = node.id + node_offset;

            // Remap input tokens
            node.input_tokens = node
                .input_tokens
                .iter()
                .map(|&tid| *token_remap.get(&tid).unwrap_or(&tid))
                .collect();

            // Remap output tokens
            node.output_tokens = node
                .output_tokens
                .iter()
                .map(|&tid| *token_remap.get(&tid).unwrap_or(&tid))
                .collect();

            // Add node to state
            let node_arc = Arc::new(node.clone());
            self.nodes.insert(node.id, Arc::clone(&node_arc));

            // Initialize operation state
            self.op_states.insert(
                node.id,
                OpState::new_with_effects(operation_effects(&node.op_type)),
            );

            // Set priority
            let priority = Priority::from_u8(node.metadata.priority as u8);
            self.priorities.insert(node.id, priority);

            // Initialize output tokens
            for (&original, &token_id) in original_outputs.iter().zip(node.output_tokens.iter()) {
                if config.token_connections.contains_key(&original) {
                    continue;
                }
                if !self.tokens.contains_key(&token_id) {
                    self.tokens.insert(token_id, TokenState::new());
                }
            }

            // Register as consumer for input tokens and check readiness
            let mut all_inputs_ready = true;
            for (original_token, &token_id) in original_inputs.iter().zip(node.input_tokens.iter())
            {
                if config.token_connections.contains_key(original_token) {
                    let Some(mut token_state) = self.tokens.get_mut(&token_id) else {
                        return Err(RuntimeError::Scheduler {
                            message: format!(
                                "Token connection references non-existent token: {}",
                                token_id
                            ),
                        });
                    };
                    token_state.consumers.push(node.id);
                    if !token_state.ready {
                        all_inputs_ready = false;
                    }
                } else {
                    let mut entry = self.tokens.entry(token_id).or_insert_with(|| {
                        let mut ts = TokenState::new();
                        ts.ready = true;
                        ts.value = Some(Value::Null);
                        ts
                    });
                    entry.consumers.push(node.id);
                    if !entry.ready {
                        all_inputs_ready = false;
                    }
                }
            }

            // If all inputs are ready, mark for scheduling
            if all_inputs_ready {
                ready_nodes.push((node.id, priority));
            }

            log_trace!(
                "scheduler::splice",
                old_node_id = old_node_id,
                new_node_id = node.id,
                op_type = ?node.op_type,
                ready = all_inputs_ready,
                "Spliced node into DAG"
            );
        }

        // Increment remaining counter for new nodes
        let new_node_count = config.inner_dag.nodes.len();
        self.remaining
            .fetch_add(new_node_count, std::sync::atomic::Ordering::Relaxed);

        // Enqueue ready nodes
        for (node_id, priority) in &ready_nodes {
            self.queue.push(*node_id, *priority);
            tracing::debug!(node_id = node_id, "Enqueued ready inner DAG node");
        }

        // Record progress to prevent deadlock detection
        self.record_progress();

        tracing::info!(
            spliced_nodes = new_node_count,
            ready_count = ready_nodes.len(),
            "Inner DAG spliced successfully"
        );

        Ok(token_remap)
    }
}

/// Concrete implementation of the DagSplicer trait backed by a SchedulerState.
pub struct SchedulerDagSplicer {
    state: Arc<SchedulerState>,
}

impl SchedulerDagSplicer {
    pub fn new(state: Arc<SchedulerState>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl DagSplicer for SchedulerDagSplicer {
    async fn splice_dag(
        &self,
        inner_dag: ExecutionDag,
        token_connections: HashMap<TokenId, TokenId>,
    ) -> SpliceResult {
        let config = SpliceConfig {
            inner_dag,
            token_connections,
            node_id_offset: None,
            token_id_offset: None,
        };

        self.state.splice_dag(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::MetricsCollector;
    use crate::scheduler::config::SchedulerConfig;
    use crate::scheduler::state::SchedulerState;
    use apxm_core::types::{Node, Value, execution::DagMetadata, operations::AISOperationType};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    #[test]
    fn test_splice_config_creation() {
        let inner_dag = ExecutionDag {
            nodes: vec![],
            edges: vec![],
            entry_nodes: vec![],
            exit_nodes: vec![],
            metadata: DagMetadata::default(),
        };

        let config = SpliceConfig {
            inner_dag,
            token_connections: HashMap::new(),
            node_id_offset: Some(100),
            token_id_offset: Some(200),
        };

        assert_eq!(config.node_id_offset, Some(100));
        assert_eq!(config.token_id_offset, Some(200));
    }

    #[test]
    fn test_splice_dag_connects_outer_token() {
        let mut outer_dag = ExecutionDag::new();

        let mut producer = Node::new(1, AISOperationType::ConstStr);
        producer.output_tokens = vec![1];
        producer
            .attributes
            .insert("value".into(), Value::String("seed".into()));

        let mut consumer = Node::new(2, AISOperationType::Return);
        consumer.input_tokens = vec![1];

        outer_dag.nodes.push(producer);
        outer_dag.nodes.push(consumer);
        outer_dag.entry_nodes = vec![1];
        outer_dag.exit_nodes = vec![2];

        let cfg = SchedulerConfig::default();
        let metrics = Arc::new(MetricsCollector::default());
        let (state, _) = SchedulerState::new(outer_dag, cfg, metrics, Instant::now()).unwrap();
        let state = Arc::new(state);

        let mut inner_dag = ExecutionDag::new();
        let mut inner = Node::new(5, AISOperationType::Rsn);
        inner.input_tokens = vec![10];
        inner.output_tokens = vec![11];
        inner_dag.nodes.push(inner);
        inner_dag.entry_nodes = vec![5];
        inner_dag.exit_nodes = vec![5];

        let config = SpliceConfig {
            inner_dag,
            token_connections: HashMap::from([(10u64, 1u64)]),
            node_id_offset: None,
            token_id_offset: None,
        };

        let remap = state.splice_dag(config).expect("splice succeeds");
        assert_eq!(remap.get(&10), Some(&1));

        let inserted_id = state
            .nodes
            .iter()
            .filter_map(|entry| {
                let id = *entry.key();
                let node = entry.value();
                (id > 2 && matches!(node.op_type, AISOperationType::Rsn)).then_some(id)
            })
            .next()
            .expect("inserted node present");

        let token_state = state.tokens.get(&1).expect("outer token exists");
        assert!(token_state.consumers.iter().any(|&id| id == inserted_id));
    }
}
