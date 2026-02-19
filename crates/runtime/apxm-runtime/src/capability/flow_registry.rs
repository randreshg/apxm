//! Flow Registry for multi-agent flow management.
//!
//! The FlowRegistry stores compiled flow DAGs for all agents, enabling
//! cross-agent flow calls and multi-agent execution.

use std::sync::Arc;

use apxm_core::types::ExecutionDag;
use dashmap::DashMap;

/// Registry for storing and retrieving compiled flow DAGs.
///
/// This enables cross-agent flow calls by providing a way to look up
/// and execute flows from other agents.
#[derive(Debug, Default)]
pub struct FlowRegistry {
    /// Map of (agent_name, flow_name) -> compiled DAG
    flows: DashMap<(String, String), Arc<ExecutionDag>>,
}

impl FlowRegistry {
    /// Create a new empty flow registry.
    pub fn new() -> Self {
        Self {
            flows: DashMap::new(),
        }
    }

    /// Register a flow in the registry.
    ///
    /// # Arguments
    /// * `agent` - The agent name that owns this flow
    /// * `flow` - The flow name
    /// * `dag` - The compiled execution DAG for this flow
    pub fn register_flow(&self, agent: &str, flow: &str, dag: ExecutionDag) {
        self.flows
            .insert((agent.to_string(), flow.to_string()), Arc::new(dag));
    }

    /// Get a flow from the registry.
    ///
    /// Returns a clone of the DAG if found, `None` otherwise.
    /// The DAG is wrapped in Arc for efficient sharing.
    pub fn get_flow(&self, agent: &str, flow: &str) -> Option<Arc<ExecutionDag>> {
        self.flows
            .get(&(agent.to_string(), flow.to_string()))
            .map(|dag| Arc::clone(&dag))
    }

    /// Check if a flow exists in the registry.
    pub fn has_flow(&self, agent: &str, flow: &str) -> bool {
        self.flows
            .contains_key(&(agent.to_string(), flow.to_string()))
    }

    /// List all registered flows.
    pub fn list_flows(&self) -> Vec<(String, String)> {
        self.flows.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get all flows for a specific agent.
    pub fn flows_for_agent(&self, agent: &str) -> Vec<String> {
        self.flows
            .iter()
            .filter(|entry| entry.key().0 == agent)
            .map(|entry| entry.key().1.clone())
            .collect()
    }

    /// Remove a flow from the registry.
    pub fn remove_flow(&self, agent: &str, flow: &str) -> Option<Arc<ExecutionDag>> {
        self.flows
            .remove(&(agent.to_string(), flow.to_string()))
            .map(|(_, dag)| dag)
    }

    /// Clear all registered flows.
    pub fn clear(&self) {
        self.flows.clear();
    }

    /// Get the number of registered flows.
    pub fn len(&self) -> usize {
        self.flows.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.flows.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get_flow() {
        let registry = FlowRegistry::new();

        let dag = ExecutionDag {
            nodes: vec![],
            edges: vec![],
            entry_nodes: vec![],
            exit_nodes: vec![],
            metadata: Default::default(),
        };

        registry.register_flow("TestAgent", "main", dag.clone());

        assert!(registry.has_flow("TestAgent", "main"));
        assert!(!registry.has_flow("TestAgent", "other"));
        assert!(!registry.has_flow("OtherAgent", "main"));

        let retrieved = registry.get_flow("TestAgent", "main");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_list_flows() {
        let registry = FlowRegistry::new();

        let dag = ExecutionDag {
            nodes: vec![],
            edges: vec![],
            entry_nodes: vec![],
            exit_nodes: vec![],
            metadata: Default::default(),
        };

        registry.register_flow("Agent1", "flow1", dag.clone());
        registry.register_flow("Agent1", "flow2", dag.clone());
        registry.register_flow("Agent2", "main", dag.clone());

        assert_eq!(registry.len(), 3);

        let agent1_flows = registry.flows_for_agent("Agent1");
        assert_eq!(agent1_flows.len(), 2);
        assert!(agent1_flows.contains(&"flow1".to_string()));
        assert!(agent1_flows.contains(&"flow2".to_string()));
    }
}
