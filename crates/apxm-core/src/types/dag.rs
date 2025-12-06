//! Execution DAG representation.
//!
//! The ExecutionDag represents the complete dataflow graph of operations,
//! with validation, cycle detection, and serialization support.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::types::{Edge, Node, NodeId, TokenId};

/// Metadata associated with the execution DAG.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DagMetadata {
    /// Optional name for the DAG.
    pub name: Option<String>,
}

/// Represents a complete execution DAG (Directed Acyclic Graph).
///
/// The DAG contains nodes (operations) and edges (dependencies),
/// with support for validation, cycle detection, and serialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionDag {
    /// All nodes in the DAG.
    pub nodes: Vec<Node>,
    /// All edges in the DAG.
    pub edges: Vec<Edge>,
    /// Entry nodes (nodes with no incoming edges).
    pub entry_nodes: Vec<NodeId>,
    /// Exit nodes (nodes with no outgoing edges).
    pub exit_nodes: Vec<NodeId>,
    /// Metadata for the DAG.
    pub metadata: DagMetadata,
}

impl ExecutionDag {
    /// Creates a new empty DAG.
    pub fn new() -> Self {
        ExecutionDag {
            nodes: Vec::new(),
            edges: Vec::new(),
            entry_nodes: Vec::new(),
            exit_nodes: Vec::new(),
            metadata: DagMetadata::default(),
        }
    }

    /// Adds a node to the DAG.
    ///
    /// # Errors
    ///
    /// Returns an error if a node with the same ID already exists.
    pub fn add_node(&mut self, node: Node) -> Result<(), String> {
        if self.nodes.iter().any(|n| n.id == node.id) {
            return Err(format!("Node with ID {} already exists", node.id));
        }
        self.nodes.push(node);
        Ok(())
    }

    /// Adds an edge to the DAG.
    ///
    /// # Errors
    ///
    /// Returns an error if the source or target node doesn't exist.
    pub fn add_edge(&mut self, edge: Edge) -> Result<(), String> {
        if !self.nodes.iter().any(|n| n.id == edge.from) {
            return Err(format!("Source node {} does not exist", edge.from));
        }
        if !self.nodes.iter().any(|n| n.id == edge.to) {
            return Err(format!("Target node {} does not exist", edge.to));
        }
        self.edges.push(edge);
        Ok(())
    }

    /// Gets a reference to a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Gets a mutable reference to a node by ID.
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Gets all edges that originate from a given node.
    pub fn get_edges_from(&self, node_id: NodeId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.from == node_id).collect()
    }

    /// Gets all edges that point to a given node.
    pub fn get_edges_to(&self, node_id: NodeId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.to == node_id).collect()
    }

    /// Finds all entry nodes (nodes with no incoming edges).
    pub fn find_entry_nodes(&self) -> Vec<NodeId> {
        let mut has_incoming: HashSet<NodeId> = HashSet::new();
        for edge in &self.edges {
            has_incoming.insert(edge.to);
        }
        self.nodes
            .iter()
            .filter_map(|n| {
                if !has_incoming.contains(&n.id) {
                    Some(n.id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Finds all exit nodes (nodes with no outgoing edges).
    pub fn find_exit_nodes(&self) -> Vec<NodeId> {
        let mut has_outgoing: HashSet<NodeId> = HashSet::new();
        for edge in &self.edges {
            has_outgoing.insert(edge.from);
        }
        self.nodes
            .iter()
            .filter_map(|n| {
                if !has_outgoing.contains(&n.id) {
                    Some(n.id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Checks if the DAG contains cycles.
    ///
    /// Uses Kahn's algorithm (topological sort) to detect cycles.
    pub fn has_cycles(&self) -> bool {
        // Build adjacency list and in-degree count
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        // Initialize in-degree for all nodes
        for node in &self.nodes {
            in_degree.insert(node.id, 0);
            adjacency.insert(node.id, Vec::new());
        }

        // Count in-degrees and build adjacency list
        for edge in &self.edges {
            *in_degree.entry(edge.to).or_insert(0) += 1;
            adjacency
                .entry(edge.from)
                .or_insert_with(Vec::new)
                .push(edge.to);
        }

        // Find nodes with no incoming edges
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        for (node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(*node_id);
            }
        }

        // Process nodes
        let mut processed = 0;
        while let Some(node_id) = queue.pop_front() {
            processed += 1;
            if let Some(neighbors) = adjacency.get(&node_id) {
                for &neighbor in neighbors {
                    let degree = in_degree.get_mut(&neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // If we didn't process all nodes, there's a cycle
        processed != self.nodes.len()
    }

    /// Validates the DAG structure.
    ///
    /// Checks for:
    /// - Invalid node references in edges
    /// - Invalid token references
    /// - Cycles
    /// - Consistency of entry/exit nodes
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn validate(&self) -> Result<(), String> {
        // Check for cycles
        if self.has_cycles() {
            return Err("DAG contains cycles".to_string());
        }

        // Check that all edge references are valid
        let node_ids: HashSet<NodeId> = self.nodes.iter().map(|n| n.id).collect();
        for edge in &self.edges {
            if !node_ids.contains(&edge.from) {
                return Err(format!(
                    "Edge references non-existent source node {}",
                    edge.from
                ));
            }
            if !node_ids.contains(&edge.to) {
                return Err(format!(
                    "Edge references non-existent target node {}",
                    edge.to
                ));
            }
        }

        // Check token references in nodes
        let mut token_ids: HashSet<TokenId> = HashSet::new();
        for node in &self.nodes {
            for &token_id in &node.input_tokens {
                token_ids.insert(token_id);
            }
            for &token_id in &node.output_tokens {
                token_ids.insert(token_id);
            }
        }

        // Check that edges reference valid tokens
        for edge in &self.edges {
            if !token_ids.contains(&edge.token_id) {
                return Err(format!(
                    "Edge references non-existent token {}",
                    edge.token_id
                ));
            }
        }

        // Validate entry/exit nodes
        let computed_entry = self.find_entry_nodes();
        let computed_exit = self.find_exit_nodes();

        // Check if stored entry/exit nodes match computed ones
        let stored_entry_set: HashSet<NodeId> = self.entry_nodes.iter().copied().collect();
        let computed_entry_set: HashSet<NodeId> = computed_entry.iter().copied().collect();
        if stored_entry_set != computed_entry_set {
            return Err("Stored entry nodes don't match computed entry nodes".to_string());
        }

        let stored_exit_set: HashSet<NodeId> = self.exit_nodes.iter().copied().collect();
        let computed_exit_set: HashSet<NodeId> = computed_exit.iter().copied().collect();
        if stored_exit_set != computed_exit_set {
            return Err("Stored exit nodes don't match computed exit nodes".to_string());
        }

        Ok(())
    }
}

impl Default for ExecutionDag {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AISOperationType;

    #[test]
    fn test_new_dag() {
        let dag = ExecutionDag::new();
        assert!(dag.nodes.is_empty());
        assert!(dag.edges.is_empty());
    }

    #[test]
    fn test_add_node() {
        let mut dag = ExecutionDag::new();
        let node = Node::new(1, AISOperationType::Inv);
        assert!(dag.add_node(node).is_ok());
        assert_eq!(dag.nodes.len(), 1);
    }

    #[test]
    fn test_add_duplicate_node() {
        let mut dag = ExecutionDag::new();
        let node1 = Node::new(1, AISOperationType::Inv);
        let node2 = Node::new(1, AISOperationType::Rsn);
        dag.add_node(node1).unwrap();
        assert!(dag.add_node(node2).is_err());
    }

    #[test]
    fn test_add_edge() {
        let mut dag = ExecutionDag::new();
        dag.add_node(Node::new(1, AISOperationType::Inv)).unwrap();
        dag.add_node(Node::new(2, AISOperationType::Rsn)).unwrap();
        let edge = Edge::new(1, 2, 10, crate::types::DependencyType::Data);
        assert!(dag.add_edge(edge).is_ok());
        assert_eq!(dag.edges.len(), 1);
    }

    #[test]
    fn test_add_edge_invalid_source() {
        let mut dag = ExecutionDag::new();
        dag.add_node(Node::new(2, AISOperationType::Rsn)).unwrap();
        let edge = Edge::new(1, 2, 10, crate::types::DependencyType::Data);
        assert!(dag.add_edge(edge).is_err());
    }

    #[test]
    fn test_get_node() {
        let mut dag = ExecutionDag::new();
        let node = Node::new(1, AISOperationType::Inv);
        dag.add_node(node.clone()).unwrap();
        assert_eq!(dag.get_node(1), Some(&node));
        assert_eq!(dag.get_node(999), None);
    }

    #[test]
    fn test_get_edges_from() {
        let mut dag = ExecutionDag::new();
        dag.add_node(Node::new(1, AISOperationType::Inv)).unwrap();
        dag.add_node(Node::new(2, AISOperationType::Rsn)).unwrap();
        dag.add_node(Node::new(3, AISOperationType::QMem)).unwrap();
        dag.add_edge(Edge::new(1, 2, 10, crate::types::DependencyType::Data))
            .unwrap();
        dag.add_edge(Edge::new(1, 3, 20, crate::types::DependencyType::Data))
            .unwrap();
        let edges = dag.get_edges_from(1);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_find_entry_nodes() {
        let mut dag = ExecutionDag::new();
        dag.add_node(Node::new(1, AISOperationType::Inv)).unwrap();
        dag.add_node(Node::new(2, AISOperationType::Rsn)).unwrap();
        dag.add_edge(Edge::new(1, 2, 10, crate::types::DependencyType::Data))
            .unwrap();
        let entry = dag.find_entry_nodes();
        assert_eq!(entry, vec![1]);
    }

    #[test]
    fn test_find_exit_nodes() {
        let mut dag = ExecutionDag::new();
        dag.add_node(Node::new(1, AISOperationType::Inv)).unwrap();
        dag.add_node(Node::new(2, AISOperationType::Rsn)).unwrap();
        dag.add_edge(Edge::new(1, 2, 10, crate::types::DependencyType::Data))
            .unwrap();
        let exit = dag.find_exit_nodes();
        assert_eq!(exit, vec![2]);
    }

    #[test]
    fn test_no_cycles_linear() {
        let mut dag = ExecutionDag::new();
        dag.add_node(Node::new(1, AISOperationType::Inv)).unwrap();
        dag.add_node(Node::new(2, AISOperationType::Rsn)).unwrap();
        dag.add_node(Node::new(3, AISOperationType::QMem)).unwrap();
        dag.add_edge(Edge::new(1, 2, 10, crate::types::DependencyType::Data))
            .unwrap();
        dag.add_edge(Edge::new(2, 3, 20, crate::types::DependencyType::Data))
            .unwrap();
        assert!(!dag.has_cycles());
    }

    #[test]
    fn test_has_cycles() {
        let mut dag = ExecutionDag::new();
        dag.add_node(Node::new(1, AISOperationType::Inv)).unwrap();
        dag.add_node(Node::new(2, AISOperationType::Rsn)).unwrap();
        dag.add_edge(Edge::new(1, 2, 10, crate::types::DependencyType::Data))
            .unwrap();
        dag.add_edge(Edge::new(2, 1, 20, crate::types::DependencyType::Data))
            .unwrap();
        assert!(dag.has_cycles());
    }

    #[test]
    fn test_validate_ok() {
        let mut dag = ExecutionDag::new();
        let mut node1 = Node::new(1, AISOperationType::Inv);
        node1.add_output_token(10);
        dag.add_node(node1).unwrap();
        let mut node2 = Node::new(2, AISOperationType::Rsn);
        node2.add_input_token(10);
        dag.add_node(node2).unwrap();
        dag.add_edge(Edge::new(1, 2, 10, crate::types::DependencyType::Data))
            .unwrap();
        dag.entry_nodes = dag.find_entry_nodes();
        dag.exit_nodes = dag.find_exit_nodes();
        assert!(dag.validate().is_ok());
    }

    #[test]
    fn test_validate_cycle() {
        let mut dag = ExecutionDag::new();
        dag.add_node(Node::new(1, AISOperationType::Inv)).unwrap();
        dag.add_node(Node::new(2, AISOperationType::Rsn)).unwrap();
        dag.add_edge(Edge::new(1, 2, 10, crate::types::DependencyType::Data))
            .unwrap();
        dag.add_edge(Edge::new(2, 1, 20, crate::types::DependencyType::Data))
            .unwrap();
        assert!(dag.validate().is_err());
    }
}
