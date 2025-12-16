//! Execution DAG edge representation.
//!
//! Edges represent dependencies between nodes in the execution DAG,
//! carrying tokens and indicating the type of dependency.

use serde::{Deserialize, Serialize};

use crate::types::{NodeId, TokenId};

/// Represents the type of dependency between nodes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyType {
    // Data dependency: the target node needs the value from the source node.
    Data,
    // Effect dependency: the target node must execute after the source node's side effects.
    Effect,
    // Control dependency: the target node's execution is controlled by the source node.
    Control,
}

/// Represents an edge in the execution DAG.
///
/// An edge connects two nodes and carries a token, indicating a dependency between the nodes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    // Source node ID (where the token is produced).
    pub from: NodeId,
    // Target node ID (where the token is consumed).
    pub to: NodeId,
    // Token ID that flows through this edge.
    pub token_id: TokenId,
    // Type of dependency this edge represents.
    pub dependency_type: DependencyType,
}

impl Edge {
    /// Creates a new edge.
    ///
    /// Examples
    ///
    /// ```
    /// use apxm_core::types::{Edge, DependencyType};
    ///
    /// let edge = Edge::new(1, 2, 10, DependencyType::Data);
    /// assert_eq!(edge.from, 1);
    /// assert_eq!(edge.to, 2);
    /// ```
    pub fn new(
        from: NodeId,
        to: NodeId,
        token_id: TokenId,
        dependency_type: DependencyType,
    ) -> Self {
        Edge {
            from,
            to,
            token_id,
            dependency_type,
        }
    }

    /// Checks if this is a data dependency.
    pub fn is_data_dependency(&self) -> bool {
        matches!(self.dependency_type, DependencyType::Data)
    }

    // Checks if this is an effect dependency.
    pub fn is_effect_dependency(&self) -> bool {
        matches!(self.dependency_type, DependencyType::Effect)
    }

    // Checks if this is a control dependency.
    pub fn is_control_dependency(&self) -> bool {
        matches!(self.dependency_type, DependencyType::Control)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_edge() {
        let edge = Edge::new(1, 2, 10, DependencyType::Data);
        assert_eq!(edge.from, 1);
        assert_eq!(edge.to, 2);
        assert_eq!(edge.token_id, 10);
        assert!(edge.is_data_dependency());
    }

    #[test]
    fn test_dependency_types() {
        let data_edge = Edge::new(1, 2, 10, DependencyType::Data);
        assert!(data_edge.is_data_dependency());
        assert!(!data_edge.is_effect_dependency());
        assert!(!data_edge.is_control_dependency());

        let effect_edge = Edge::new(1, 2, 10, DependencyType::Effect);
        assert!(!effect_edge.is_data_dependency());
        assert!(effect_edge.is_effect_dependency());
        assert!(!effect_edge.is_control_dependency());

        let control_edge = Edge::new(1, 2, 10, DependencyType::Control);
        assert!(!control_edge.is_data_dependency());
        assert!(!control_edge.is_effect_dependency());
        assert!(control_edge.is_control_dependency());
    }

    #[test]
    fn test_serialization() {
        let edge = Edge::new(1, 2, 10, DependencyType::Data);
        let json = serde_json::to_string(&edge).expect("serialize edge");
        assert!(json.contains("1"));
        assert!(json.contains("2"));
        assert!(json.contains("10"));
    }
}
