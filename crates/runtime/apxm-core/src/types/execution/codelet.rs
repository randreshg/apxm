//! Codelet -- the fundamental unit of AI work in APXM.
//!
//! A codelet represents a schedulable piece of AI computation, from a single
//! LLM prompt to a composed pipeline of operations.  Codelets fire when all
//! dependencies are satisfied, and independent codelets are automatically
//! parallelized by the dataflow scheduler.
//!
//! # Design
//!
//! A [`Codelet`] wraps one or more APXM [`Node`]s into a logical unit of AI
//! work.  A [`CodeletDag`] organises codelets into a dependency graph that can
//! be validated (cycle detection, missing-dependency checks) and then lowered
//! to an [`ExecutionDag`] for the runtime scheduler.
//!
//! # Background & Inspiration
//!
//! The name *codelet* draws on two traditions:
//!
//! - **HPC dataflow** (Gao et al., CAPSL) -- fine-grained, dependency-driven
//!   firing rules and automatic parallelism.
//! - **Cognitive science** (Baars & Franklin, Global Workspace Theory) --
//!   specialized processors that fire when conditions are met and broadcast
//!   results to a shared workspace.
//!
//! APXM defines its own concept -- the *AI Codelet* -- that is structurally
//! isomorphic to both: a unit of work that fires when its inputs are ready,
//! produces typed outputs, and composes into dependency graphs.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use super::{DagMetadata, DependencyType, Edge, ExecutionDag, Node, NodeId};
use crate::constants::graph::attrs as graph_attrs;
use crate::error::runtime::RuntimeError;
use crate::types::AISOperationType;

// ---------------------------------------------------------------------------
// CodeletId
// ---------------------------------------------------------------------------

/// Unique identifier for a codelet.
pub type CodeletId = u64;

// ---------------------------------------------------------------------------
// CodeletMetadata
// ---------------------------------------------------------------------------

/// Metadata associated with a [`Codelet`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CodeletMetadata {
    /// Scheduling priority (higher = more important).
    #[serde(default)]
    pub priority: u32,
    /// Optional JSON Schema describing the expected output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_output_schema: Option<String>,
}

// ---------------------------------------------------------------------------
// Codelet
// ---------------------------------------------------------------------------

/// The fundamental unit of AI work in APXM.
///
/// A codelet represents a schedulable piece of AI computation -- from a
/// single LLM prompt to a composed pipeline of operations.  Codelets fire
/// when all dependencies are satisfied, and independent codelets are
/// automatically parallelized by the dataflow scheduler.
///
/// Each codelet maps to one or more underlying APXM [`Node`]s.  The
/// [`CodeletDag`] manages inter-codelet dependencies and can be lowered to
/// an [`ExecutionDag`] for scheduling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Codelet {
    /// Unique identifier.
    pub id: CodeletId,
    /// Human-readable name.
    pub name: String,
    /// Description of what this codelet does.
    pub description: String,
    /// The underlying APXM node IDs that implement this codelet.
    pub nodes: Vec<NodeId>,
    /// Codelets that must complete before this one fires.
    pub depends_on: Vec<CodeletId>,
    /// Scheduling and output metadata.
    pub metadata: CodeletMetadata,
}

impl Codelet {
    /// Creates a new codelet with the given id, name, and description.
    pub fn new(id: CodeletId, name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: description.into(),
            nodes: Vec::new(),
            depends_on: Vec::new(),
            metadata: CodeletMetadata::default(),
        }
    }

    /// Adds an underlying APXM node to this codelet.
    pub fn add_node(mut self, node_id: NodeId) -> Self {
        self.nodes.push(node_id);
        self
    }

    /// Declares a dependency on another codelet.
    pub fn add_dependency(mut self, dep: CodeletId) -> Self {
        self.depends_on.push(dep);
        self
    }

    /// Sets scheduling priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.metadata.priority = priority;
        self
    }

    /// Sets the expected output JSON Schema.
    pub fn with_expected_output_schema(mut self, schema: impl Into<String>) -> Self {
        self.metadata.expected_output_schema = Some(schema.into());
        self
    }
}

// ---------------------------------------------------------------------------
// CodeletDag
// ---------------------------------------------------------------------------

/// A directed acyclic graph of codelets.
///
/// Validates dependencies, detects cycles at construction time, and can be
/// lowered to an [`ExecutionDag`] for the dataflow scheduler.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeletDag {
    /// Human-readable name for this DAG.
    pub name: String,
    /// The codelets in this graph.
    pub codelets: Vec<Codelet>,
    /// DAG-level metadata (forwarded when lowering to [`ExecutionDag`]).
    pub metadata: DagMetadata,
}

impl CodeletDag {
    /// Creates a new empty codelet DAG.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            codelets: Vec::new(),
            metadata: DagMetadata::default(),
        }
    }

    /// Adds a codelet to the DAG.
    pub fn add_codelet(mut self, codelet: Codelet) -> Self {
        self.codelets.push(codelet);
        self
    }

    /// Serialize this codelet DAG as JSON.
    pub fn to_json(&self) -> Result<String, RuntimeError> {
        serde_json::to_string(self).map_err(|e| {
            RuntimeError::Serialization(format!("failed to serialize CodeletDag: {e}"))
        })
    }

    /// Deserialize a codelet DAG from JSON.
    pub fn from_json(payload: &str) -> Result<Self, RuntimeError> {
        serde_json::from_str(payload).map_err(|e| {
            RuntimeError::Serialization(format!("failed to deserialize CodeletDag: {e}"))
        })
    }

    /// Validates the codelet DAG.
    ///
    /// Checks for:
    /// - Duplicate codelet IDs
    /// - Missing dependency references
    /// - Dependency cycles (via Kahn's algorithm)
    pub fn validate(&self) -> Result<(), RuntimeError> {
        let ids: HashMap<CodeletId, &Codelet> = self.codelets.iter().map(|c| (c.id, c)).collect();

        // Check for duplicate IDs
        if ids.len() != self.codelets.len() {
            return Err(RuntimeError::State(
                "CodeletDag contains duplicate codelet IDs".to_string(),
            ));
        }

        // Check all dependency references exist
        for codelet in &self.codelets {
            for dep in &codelet.depends_on {
                if !ids.contains_key(dep) {
                    return Err(RuntimeError::State(format!(
                        "codelet '{}' (id={}) depends on unknown codelet id={}",
                        codelet.name, codelet.id, dep
                    )));
                }
            }
        }

        // Cycle detection via topological sort (Kahn's algorithm)
        let mut in_degree: HashMap<CodeletId, usize> =
            self.codelets.iter().map(|c| (c.id, 0)).collect();
        let mut adjacency: HashMap<CodeletId, Vec<CodeletId>> =
            self.codelets.iter().map(|c| (c.id, Vec::new())).collect();

        for codelet in &self.codelets {
            for dep in &codelet.depends_on {
                adjacency.get_mut(dep).unwrap().push(codelet.id);
                *in_degree.get_mut(&codelet.id).unwrap() += 1;
            }
        }

        let mut queue: VecDeque<CodeletId> = in_degree
            .iter()
            .filter_map(|(&id, &deg)| (deg == 0).then_some(id))
            .collect();
        let mut visited = 0usize;

        while let Some(id) = queue.pop_front() {
            visited += 1;
            for &target in adjacency.get(&id).unwrap_or(&Vec::new()) {
                let deg = in_degree.get_mut(&target).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(target);
                }
            }
        }

        if visited != self.codelets.len() {
            return Err(RuntimeError::State(format!(
                "CodeletDag '{}' contains a dependency cycle",
                self.name
            )));
        }

        Ok(())
    }

    /// Lowers this codelet DAG to an [`ExecutionDag`] for the dataflow
    /// scheduler.
    ///
    /// Each codelet's nodes are added to the execution DAG, and dependency
    /// edges are created between the last node of a dependency codelet and
    /// the first node of the dependent codelet.
    ///
    /// Codelets without any underlying nodes get a synthetic `Ask` node
    /// created from their description.
    pub fn to_execution_dag(&self) -> Result<ExecutionDag, RuntimeError> {
        self.validate()?;

        let mut dag = ExecutionDag::new();
        dag.metadata = self.metadata.clone();
        dag.metadata.name = Some(self.name.clone());

        let mut token_counter: u64 = 1;
        // Track first and last node IDs for each codelet (for edge wiring).
        let mut codelet_first_node: HashMap<CodeletId, NodeId> = HashMap::new();
        let mut codelet_last_node: HashMap<CodeletId, NodeId> = HashMap::new();

        for codelet in &self.codelets {
            if codelet.nodes.is_empty() {
                // Synthesize a node from the codelet description.
                let node_id = codelet.id * 1000; // avoid collisions
                let mut node = Node::new(node_id, AISOperationType::Ask);
                node.set_attribute(
                    graph_attrs::PROMPT.to_string(),
                    crate::types::Value::String(codelet.description.clone()),
                );
                node.metadata.priority = codelet.metadata.priority;
                node.metadata.codelet_source_id = Some(codelet.id);
                dag.add_node(node)?;
                codelet_first_node.insert(codelet.id, node_id);
                codelet_last_node.insert(codelet.id, node_id);
            } else {
                for &node_id in &codelet.nodes {
                    if dag.get_node(node_id).is_none() {
                        let mut node = Node::new(node_id, AISOperationType::Ask);
                        node.set_attribute(
                            graph_attrs::PROMPT.to_string(),
                            crate::types::Value::String(codelet.description.clone()),
                        );
                        node.metadata.priority = codelet.metadata.priority;
                        node.metadata.codelet_source_id = Some(codelet.id);
                        dag.add_node(node)?;
                    } else if let Some(existing) = dag.get_node_mut(node_id) {
                        existing.metadata.codelet_source_id = Some(codelet.id);
                    }
                }

                // Add all specified nodes and wire them sequentially.
                let first = codelet.nodes[0];
                let last = *codelet.nodes.last().unwrap();
                codelet_first_node.insert(codelet.id, first);
                codelet_last_node.insert(codelet.id, last);

                // Wire sequential edges within the codelet.
                for pair in codelet.nodes.windows(2) {
                    let token_id = token_counter;
                    token_counter += 1;
                    // Add output token to source, input token to target.
                    if let Some(src) = dag.get_node_mut(pair[0]) {
                        src.add_output_token(token_id);
                    }
                    if let Some(tgt) = dag.get_node_mut(pair[1]) {
                        tgt.add_input_token(token_id);
                    }
                    dag.add_edge(Edge::new(pair[0], pair[1], token_id, DependencyType::Data))?;
                }
            }
        }

        // Wire inter-codelet dependency edges.
        for codelet in &self.codelets {
            for dep_id in &codelet.depends_on {
                let from_node = *codelet_last_node.get(dep_id).unwrap();
                let to_node = *codelet_first_node.get(&codelet.id).unwrap();
                let token_id = token_counter;
                token_counter += 1;

                if let Some(src) = dag.get_node_mut(from_node) {
                    src.add_output_token(token_id);
                }
                if let Some(tgt) = dag.get_node_mut(to_node) {
                    tgt.add_input_token(token_id);
                }
                dag.add_edge(Edge::new(
                    from_node,
                    to_node,
                    token_id,
                    DependencyType::Data,
                ))?;
            }
        }

        dag.entry_nodes = dag.find_entry_nodes();
        dag.exit_nodes = dag.find_exit_nodes();

        Ok(dag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codelet_builder() {
        let c = Codelet::new(1, "research", "Research a topic")
            .with_priority(10)
            .with_expected_output_schema(r#"{"type":"object"}"#)
            .add_dependency(0);

        assert_eq!(c.id, 1);
        assert_eq!(c.name, "research");
        assert_eq!(c.metadata.priority, 10);
        assert_eq!(c.depends_on, vec![0]);
    }

    #[test]
    fn codelet_dag_validates_missing_dependency() {
        let dag = CodeletDag::new("test").add_codelet(Codelet::new(1, "a", "A").add_dependency(99));

        let err = dag.validate().expect_err("should detect missing dep");
        assert!(err.to_string().contains("unknown codelet id=99"));
    }

    #[test]
    fn codelet_dag_validates_duplicate_ids() {
        let dag = CodeletDag::new("test")
            .add_codelet(Codelet::new(1, "a", "A"))
            .add_codelet(Codelet::new(1, "b", "B"));

        let err = dag.validate().expect_err("should detect duplicates");
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn codelet_dag_validates_cycles() {
        let dag = CodeletDag::new("test")
            .add_codelet(Codelet::new(1, "a", "A").add_dependency(2))
            .add_codelet(Codelet::new(2, "b", "B").add_dependency(1));

        let err = dag.validate().expect_err("should detect cycle");
        assert!(err.to_string().contains("dependency cycle"));
    }

    #[test]
    fn codelet_dag_validates_success() {
        let dag = CodeletDag::new("test")
            .add_codelet(Codelet::new(1, "a", "A"))
            .add_codelet(Codelet::new(2, "b", "B").add_dependency(1))
            .add_codelet(Codelet::new(3, "c", "C").add_dependency(2));

        dag.validate().expect("valid DAG should pass");
    }

    #[test]
    fn codelet_dag_to_execution_dag() {
        let dag = CodeletDag::new("research-pipeline")
            .add_codelet(Codelet::new(1, "research", "Research a topic"))
            .add_codelet(Codelet::new(2, "fact-check", "Verify claims").add_dependency(1))
            .add_codelet(
                Codelet::new(3, "write", "Write report")
                    .add_dependency(1)
                    .add_dependency(2),
            );

        let exec_dag = dag.to_execution_dag().expect("should lower successfully");

        // 3 synthetic nodes (one per codelet)
        assert_eq!(exec_dag.nodes.len(), 3);
        // 3 dependency edges (research->fact-check, research->write, fact-check->write)
        assert_eq!(exec_dag.edges.len(), 3);
        // 1 entry node (research)
        assert_eq!(exec_dag.entry_nodes.len(), 1);
        // 1 exit node (write)
        assert_eq!(exec_dag.exit_nodes.len(), 1);
    }

    #[test]
    fn codelet_dag_parallel_codelets() {
        // Two independent codelets with no deps should both be entry nodes
        let dag = CodeletDag::new("parallel")
            .add_codelet(Codelet::new(1, "a", "Task A"))
            .add_codelet(Codelet::new(2, "b", "Task B"));

        let exec_dag = dag.to_execution_dag().expect("should lower");
        assert_eq!(exec_dag.entry_nodes.len(), 2);
        assert_eq!(exec_dag.exit_nodes.len(), 2);
        assert_eq!(exec_dag.edges.len(), 0);
    }

    #[test]
    fn codelet_dag_json_roundtrip() {
        let dag = CodeletDag::new("json-test")
            .add_codelet(Codelet::new(1, "a", "Task A"))
            .add_codelet(Codelet::new(2, "b", "Task B").add_dependency(1));

        let json = dag.to_json().expect("serialize codelet dag");
        let restored = CodeletDag::from_json(&json).expect("deserialize codelet dag");
        assert_eq!(restored.name, "json-test");
        assert_eq!(restored.codelets.len(), 2);
    }

    #[test]
    fn codelet_source_id_stamped_on_nodes() {
        let dag = CodeletDag::new("source-id").add_codelet(Codelet::new(10, "a", "Task A"));
        let exec_dag = dag.to_execution_dag().expect("should lower");
        assert_eq!(exec_dag.nodes.len(), 1);
        assert_eq!(exec_dag.nodes[0].metadata.codelet_source_id, Some(10));
    }
}
