//! Execution status types.
//!
//! Types for tracking the status of operations during DAG execution.

use crate::types::NodeId;

/// Operation execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum OpStatus {
    /// Operation is waiting for dependencies.
    Pending,
    /// Operation is ready to execute (all dependencies satisfied).
    Ready,
    /// Operation is currently executing.
    Running,
    /// Operation completed successfully.
    Completed,
    /// Operation failed after retries.
    Failed,
}

/// Status information for a single node.
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct NodeStatus {
    /// Node identifier.
    pub node_id: NodeId,
    /// Current execution status.
    pub status: OpStatus,
    /// Number of retry attempts.
    pub retries: u32,
    /// Last error message (if failed).
    pub last_error: Option<String>,
    /// Time when execution started (milliseconds since execution start).
    pub started_at_ms: Option<u128>,
    /// Time when execution finished (milliseconds since execution start).
    pub finished_at_ms: Option<u128>,
    /// Total execution duration in milliseconds.
    pub duration_ms: Option<u128>,
}

/// Execution statistics for a completed DAG.
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ExecutionStats {
    /// Total number of successfully executed nodes.
    pub executed_nodes: usize,
    /// Total number of failed nodes.
    pub failed_nodes: usize,
    /// Total execution duration in milliseconds.
    pub duration_ms: u128,
    /// Per-node status information.
    pub node_statuses: Vec<NodeStatus>,
}

impl ExecutionStats {
    /// Get the total number of nodes.
    pub fn total_nodes(&self) -> usize {
        self.executed_nodes + self.failed_nodes
    }

    /// Get the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let total = self.total_nodes();
        if total == 0 {
            return 100.0;
        }
        (self.executed_nodes as f64 / total as f64) * 100.0
    }

    /// Check if execution was fully successful.
    pub fn is_success(&self) -> bool {
        self.failed_nodes == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_stats_total_nodes() {
        let stats = ExecutionStats {
            executed_nodes: 10,
            failed_nodes: 2,
            duration_ms: 1000,
            node_statuses: vec![],
        };
        assert_eq!(stats.total_nodes(), 12);
    }

    #[test]
    fn test_execution_stats_success_rate() {
        let stats = ExecutionStats {
            executed_nodes: 8,
            failed_nodes: 2,
            duration_ms: 1000,
            node_statuses: vec![],
        };
        assert_eq!(stats.success_rate(), 80.0);
    }

    #[test]
    fn test_execution_stats_is_success() {
        let success = ExecutionStats {
            executed_nodes: 10,
            failed_nodes: 0,
            duration_ms: 1000,
            node_statuses: vec![],
        };
        assert!(success.is_success());

        let failure = ExecutionStats {
            executed_nodes: 8,
            failed_nodes: 2,
            duration_ms: 1000,
            node_statuses: vec![],
        };
        assert!(!failure.is_success());
    }

    #[test]
    fn test_empty_stats_success_rate() {
        let stats = ExecutionStats {
            executed_nodes: 0,
            failed_nodes: 0,
            duration_ms: 0,
            node_statuses: vec![],
        };
        assert_eq!(stats.success_rate(), 100.0);
        assert!(stats.is_success());
    }
}
