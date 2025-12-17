//! O(1) readiness tracking for dataflow operations.
//!
//! This module provides efficient tracking of operation readiness using pending input counts.
//! When an operation's pending count reaches 0, it becomes ready to execute.

use std::sync::Arc;

use apxm_core::types::{Node, NodeId, OpStatus, TokenId};
use dashmap::DashMap;

use crate::scheduler::internal_state::{OpState, TokenState};
use crate::scheduler::queue::{Priority, PriorityQueue};
use apxm_core::error::RuntimeError;

/// Tracks operation readiness using pending input counts.
///
/// Each operation starts with a count of how many inputs it's waiting for.
/// As inputs become ready, the count decrements. When it reaches 0, the
/// operation is marked ready and enqueued.
type RuntimeResult<T> = std::result::Result<T, RuntimeError>;

pub(crate) struct ReadySet {
    /// Number of pending inputs per operation.
    ///
    /// Operations with count = 0 are ready to execute.
    /// Missing entries are treated as ready (0 pending).
    pending_inputs: Arc<DashMap<NodeId, usize>>,
}

impl ReadySet {
    /// Create a new empty ready set.
    pub fn new() -> Self {
        Self {
            pending_inputs: Arc::new(DashMap::new()),
        }
    }

    /// Initialize readiness tracking for all nodes in the graph.
    ///
    /// Returns the set of immediately ready nodes (those with no pending inputs).
    pub(crate) fn initialize(
        &self,
        nodes: &[Node],
        tokens: &DashMap<TokenId, TokenState>,
        priorities: &DashMap<NodeId, Priority>,
        op_states: &DashMap<NodeId, OpState>,
        queue: &PriorityQueue,
    ) -> RuntimeResult<Vec<NodeId>> {
        let mut ready_nodes = Vec::new();

        for node in nodes {
            let needed = self.count_missing_inputs(node, tokens)?;

            if needed == 0 {
                // Node is immediately ready
                self.mark_ready(node.id, priorities, op_states, queue);
                ready_nodes.push(node.id);
            } else {
                // Track pending inputs
                self.pending_inputs.insert(node.id, needed);
            }
        }

        Ok(ready_nodes)
    }

    /// Count how many inputs this node is waiting for.
    ///
    /// Returns the number of input tokens that are not yet ready.
    fn count_missing_inputs(
        &self,
        node: &Node,
        tokens: &DashMap<TokenId, TokenState>,
    ) -> RuntimeResult<usize> {
        let mut needed = 0;

        for &token_id in &node.input_tokens {
            let Some(state) = tokens.get(&token_id) else {
                return Err(RuntimeError::SchedulerMissingToken {
                    node_id: node.id,
                    token_id,
                });
            };

            if !state.ready {
                needed += 1;
            }
        }

        Ok(needed)
    }

    /// Mark a node as ready and enqueue it at the appropriate priority.
    fn mark_ready(
        &self,
        node_id: NodeId,
        priorities: &DashMap<NodeId, Priority>,
        op_states: &DashMap<NodeId, OpState>,
        queue: &PriorityQueue,
    ) {
        // Update operation status
        if let Some(mut state) = op_states.get_mut(&node_id) {
            state.status = OpStatus::Ready;
        }

        // Enqueue at appropriate priority level
        let priority = priorities
            .get(&node_id)
            .map(|entry| *entry.value())
            .unwrap_or(Priority::Normal);
        queue.push(node_id, priority);
    }

    /// Handle a token becoming ready.
    ///
    /// Decrements pending counts for all consumers. If any consumer reaches 0,
    /// it's marked ready and enqueued.
    ///
    /// Returns the list of newly ready node IDs.
    pub(crate) fn on_token_ready(
        &self,
        token_id: TokenId,
        tokens: &DashMap<TokenId, TokenState>,
        priorities: &DashMap<NodeId, Priority>,
        op_states: &DashMap<NodeId, OpState>,
        queue: &PriorityQueue,
    ) -> RuntimeResult<Vec<NodeId>> {
        let Some(token_state) = tokens.get(&token_id) else {
            // Token doesn't exist - this shouldn't happen but handle gracefully
            return Ok(Vec::new());
        };

        let mut newly_ready = Vec::new();

        // Process each consumer
        for &consumer_id in &token_state.consumers {
            // Decrement pending count
            if let Some(mut entry) = self.pending_inputs.get_mut(&consumer_id) {
                *entry.value_mut() = entry.value().saturating_sub(1);
                let new_count = *entry.value();

                if new_count == 0 {
                    // Consumer is now ready
                    drop(entry); // Release lock before marking ready
                    self.pending_inputs.remove(&consumer_id);
                    self.mark_ready(consumer_id, priorities, op_states, queue);
                    newly_ready.push(consumer_id);
                }
            }
        }

        Ok(newly_ready)
    }

    /// Check if a specific node is ready.
    ///
    /// Returns true if the node has no pending inputs.
    #[cfg(test)]
    pub fn is_ready(&self, node_id: NodeId) -> bool {
        !self.pending_inputs.contains_key(&node_id)
    }

    /// Get the pending input count for a node.
    ///
    /// Returns 0 if the node is ready or not tracked.
    #[cfg(test)]
    pub fn pending_count(&self, node_id: NodeId) -> usize {
        self.pending_inputs
            .get(&node_id)
            .map(|v| *v.value())
            .unwrap_or(0)
    }

    /// Get the total number of nodes with pending inputs.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.pending_inputs.len()
    }

    /// Check if all nodes are ready (no pending inputs).
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.pending_inputs.is_empty()
    }

    /// Get a snapshot of all pending counts for diagnostics.
    #[cfg(test)]
    pub fn snapshot(&self) -> Vec<(NodeId, usize)> {
        self.pending_inputs
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect()
    }
}

impl Default for ReadySet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::internal_state::{OpState, TokenState};
    use crate::scheduler::queue::Priority;
    use apxm_core::types::{
        TokenId,
        execution::{Node, NodeId, NodeMetadata},
        operations::AISOperationType,
    };
    use dashmap::DashMap;
    use std::collections::HashMap;

    fn create_test_node(id: NodeId, input_tokens: Vec<TokenId>) -> Node {
        Node {
            id,
            op_type: AISOperationType::Inv,
            attributes: HashMap::new(),
            input_tokens,
            output_tokens: vec![id as TokenId],
            metadata: NodeMetadata::default(),
        }
    }

    #[test]
    fn test_ready_set_creation() {
        let ready_set = ReadySet::new();
        assert!(ready_set.is_empty());
        assert_eq!(ready_set.len(), 0);
    }

    #[test]
    fn test_count_missing_inputs() {
        let ready_set = ReadySet::new();
        let tokens = DashMap::new();

        // Setup: token 1 is ready, token 2 is not ready
        let mut token1 = TokenState::new();
        token1.ready = true;
        tokens.insert(1, token1);

        let token2 = TokenState::new(); // ready = false by default
        tokens.insert(2, token2);

        let node = create_test_node(100, vec![1, 2]);

        let missing = ready_set.count_missing_inputs(&node, &tokens).unwrap();
        assert_eq!(missing, 1); // Only token 2 is not ready
    }

    #[test]
    fn test_count_missing_inputs_error() {
        let ready_set = ReadySet::new();
        let tokens = DashMap::new();

        // Token 1 exists, but token 2 doesn't
        tokens.insert(1, TokenState::new());

        let node = create_test_node(100, vec![1, 2]);

        let result = ready_set.count_missing_inputs(&node, &tokens);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RuntimeError::SchedulerMissingToken { .. }
        ));
    }

    #[test]
    fn test_initialize_with_ready_nodes() {
        let ready_set = ReadySet::new();
        let tokens = DashMap::new();
        let priorities = DashMap::new();
        let op_states = DashMap::new();
        let queue = PriorityQueue::new();

        // Setup: Create two nodes, one with no inputs (ready), one with pending inputs
        let node1 = create_test_node(1, vec![]);
        let node2 = create_test_node(2, vec![10]);

        // Token 10 is not ready
        tokens.insert(10, TokenState::new());

        // Initialize state
        priorities.insert(1, Priority::Normal);
        priorities.insert(2, Priority::Normal);
        op_states.insert(1, OpState::new());
        op_states.insert(2, OpState::new());

        let ready_nodes = ready_set
            .initialize(&[node1, node2], &tokens, &priorities, &op_states, &queue)
            .unwrap();

        // Node 1 should be ready, node 2 should have pending inputs
        assert_eq!(ready_nodes.len(), 1);
        assert_eq!(ready_nodes[0], 1);
        assert!(ready_set.is_ready(1));
        assert!(!ready_set.is_ready(2));
        assert_eq!(ready_set.pending_count(2), 1);
    }

    #[test]
    fn test_on_token_ready_propagation() {
        let ready_set = ReadySet::new();
        let tokens = DashMap::new();
        let priorities = DashMap::new();
        let op_states = DashMap::new();
        let queue = PriorityQueue::new();

        // Setup: Token 10 has two consumers (nodes 1 and 2)
        let mut token = TokenState::new();
        token.consumers = vec![1, 2];
        tokens.insert(10, token);

        // Both nodes are waiting for token 10
        ready_set.pending_inputs.insert(1, 1);
        ready_set.pending_inputs.insert(2, 1);

        priorities.insert(1, Priority::Normal);
        priorities.insert(2, Priority::Normal);
        op_states.insert(1, OpState::new());
        op_states.insert(2, OpState::new());

        // Mark token 10 as ready
        let newly_ready = ready_set
            .on_token_ready(10, &tokens, &priorities, &op_states, &queue)
            .unwrap();

        // Both consumers should now be ready
        assert_eq!(newly_ready.len(), 2);
        assert!(newly_ready.contains(&1));
        assert!(newly_ready.contains(&2));
        assert!(ready_set.is_ready(1));
        assert!(ready_set.is_ready(2));
        assert!(ready_set.is_empty());
    }

    #[test]
    fn test_on_token_ready_partial() {
        let ready_set = ReadySet::new();
        let tokens = DashMap::new();
        let priorities = DashMap::new();
        let op_states = DashMap::new();
        let queue = PriorityQueue::new();

        // Setup: Token 10 has one consumer (node 1) that needs 2 inputs
        let mut token = TokenState::new();
        token.consumers = vec![1];
        tokens.insert(10, token);

        // Node 1 is waiting for 2 tokens
        ready_set.pending_inputs.insert(1, 2);

        priorities.insert(1, Priority::Normal);
        op_states.insert(1, OpState::new());

        // Mark token 10 as ready
        let newly_ready = ready_set
            .on_token_ready(10, &tokens, &priorities, &op_states, &queue)
            .unwrap();

        // Node 1 should still have 1 pending input
        assert_eq!(newly_ready.len(), 0);
        assert!(!ready_set.is_ready(1));
        assert_eq!(ready_set.pending_count(1), 1);
    }

    #[test]
    fn test_snapshot() {
        let ready_set = ReadySet::new();

        ready_set.pending_inputs.insert(1, 2);
        ready_set.pending_inputs.insert(2, 1);
        ready_set.pending_inputs.insert(3, 3);

        let snapshot = ready_set.snapshot();
        assert_eq!(snapshot.len(), 3);

        // Verify all entries are present (order may vary)
        assert!(snapshot.contains(&(1, 2)));
        assert!(snapshot.contains(&(2, 1)));
        assert!(snapshot.contains(&(3, 3)));
    }
}
