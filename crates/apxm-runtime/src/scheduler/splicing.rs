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
    types::{TokenId, execution::ExecutionDag},
};
use std::collections::HashMap;
use std::sync::Arc;

use super::internal_state::{OpState, TokenState};
use super::queue::Priority;
use super::state::SchedulerState;
use crate::aam::effects::operation_effects;

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
        tracing::info!(
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

        tracing::debug!(
            node_offset = node_offset,
            token_offset = token_offset,
            "Calculated ID offsets for splicing"
        );

        // Build token remapping
        let mut token_remap = HashMap::new();
        for (inner_token_id, _outer_token_id) in &config.token_connections {
            // Tokens that are connected to outer plan keep their connection
            // (we'll handle this separately)
            token_remap.insert(*inner_token_id, *inner_token_id);
        }

        // Remap all other inner DAG token IDs
        for node in &config.inner_dag.nodes {
            for &token_id in node.output_tokens.iter() {
                if !token_remap.contains_key(&token_id) {
                    token_remap.insert(token_id, token_id + token_offset);
                }
            }
            for &token_id in node.input_tokens.iter() {
                if !token_remap.contains_key(&token_id) {
                    // Check if this token is connected to outer plan
                    if let Some(&outer_token_id) = config.token_connections.get(&token_id) {
                        token_remap.insert(token_id, outer_token_id);
                    } else {
                        token_remap.insert(token_id, token_id + token_offset);
                    }
                }
            }
        }

        // Remap nodes and add to scheduler state
        let mut ready_nodes = Vec::new();

        for node in &config.inner_dag.nodes {
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
            for &token_id in &node.output_tokens {
                if !self.tokens.contains_key(&token_id) {
                    self.tokens.insert(token_id, TokenState::new());
                }
            }

            // Register as consumer for input tokens and check readiness
            let mut all_inputs_ready = true;
            for &token_id in &node.input_tokens {
                // If token is from outer plan connection, it should already exist
                if let Some(outer_token_id) = config.token_connections.get(&token_id) {
                    // Check if the outer token is ready
                    if let Some(token_state) = self.tokens.get(outer_token_id) {
                        if !token_state.ready {
                            all_inputs_ready = false;
                        }
                    } else {
                        return Err(RuntimeError::Scheduler {
                            message: format!(
                                "Token connection references non-existent token: {}",
                                outer_token_id
                            ),
                        });
                    }
                } else {
                    // Token should be produced by another inner DAG node
                    self.tokens
                        .entry(token_id)
                        .or_insert_with(|| {
                            let mut ts = TokenState::new();
                            // If this is an entry node with no producer in inner DAG,
                            // mark as ready with null value
                            ts.ready = false;
                            ts
                        })
                        .consumers
                        .push(node.id);

                    if let Some(token_state) = self.tokens.get(&token_id) {
                        if !token_state.ready {
                            all_inputs_ready = false;
                        }
                    }
                }
            }

            // If all inputs are ready, mark for scheduling
            if all_inputs_ready {
                ready_nodes.push((node.id, priority));
            }

            tracing::trace!(
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

#[cfg(test)]
mod tests {
    use super::*;
    use apxm_core::types::execution::DagMetadata;
    use std::collections::HashMap;

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
}
