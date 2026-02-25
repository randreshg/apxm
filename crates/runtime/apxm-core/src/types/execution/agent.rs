//! Runtime agent model.
//!
//! Agents are the top-level runtime unit that own named flows. Each flow can
//! preserve its original codelet DAG representation and its lowered execution
//! DAG used by the scheduler.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{CodeletDag, ExecutionDag, FlowParameter};
use crate::error::runtime::RuntimeError;

/// Unique identifier for an agent.
pub type AgentId = String;

/// Runtime agent containing one or more flows.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Agent {
    /// Human-readable agent name.
    pub name: AgentId,
    /// Agent-level declarations (memory, capabilities, tools, context).
    #[serde(default)]
    pub metadata: AgentMetadata,
    /// Named flows owned by this agent.
    #[serde(default)]
    pub flows: HashMap<String, AgentFlow>,
}

impl Agent {
    /// Create a new runtime agent with no flows.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            metadata: AgentMetadata::default(),
            flows: HashMap::new(),
        }
    }

    /// Add (or replace) a flow in this agent.
    pub fn add_flow(mut self, flow: AgentFlow) -> Self {
        self.flows.insert(flow.name.clone(), flow);
        self
    }

    /// Get the designated entry flow, if present.
    pub fn entry_flow(&self) -> Option<&AgentFlow> {
        self.flows.values().find(|flow| flow.is_entry)
    }

    /// Get a named flow by name.
    pub fn get_flow(&self, flow_name: &str) -> Option<&AgentFlow> {
        self.flows.get(flow_name)
    }
}

/// A single flow within an agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentFlow {
    /// Flow name.
    pub name: String,
    /// Whether this flow is the entry flow for its artifact.
    #[serde(default)]
    pub is_entry: bool,
    /// Flow parameters (for entry argument validation).
    #[serde(default)]
    pub parameters: Vec<FlowParameter>,
    /// Optional original codelet DAG representation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codelet_dag: Option<CodeletDag>,
    /// Lowered execution DAG used by runtime scheduling.
    pub execution_dag: ExecutionDag,
}

impl AgentFlow {
    /// Build an agent flow from a codelet DAG while preserving both views.
    pub fn from_codelet_dag(
        name: impl Into<String>,
        codelet_dag: CodeletDag,
        is_entry: bool,
    ) -> Result<Self, RuntimeError> {
        let name = name.into();
        let mut execution_dag = codelet_dag.to_execution_dag()?;
        execution_dag.metadata.name = Some(name.clone());
        execution_dag.metadata.is_entry = is_entry;
        let parameters = execution_dag.metadata.parameters.clone();

        Ok(Self {
            name,
            is_entry,
            parameters,
            codelet_dag: Some(codelet_dag),
            execution_dag,
        })
    }
}

/// Runtime-level declarations associated with an agent.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AgentMetadata {
    /// Declared memories.
    #[serde(default)]
    pub memories: Vec<MemoryDeclaration>,
    /// Declared capabilities.
    #[serde(default)]
    pub capabilities: Vec<CapabilityDeclaration>,
    /// Declared tool names.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Discoverable capability/tool names.
    #[serde(default)]
    pub discoverable: Vec<String>,
    /// Optional contextual prompt/details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

/// Declares an agent memory by name and tier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryDeclaration {
    pub name: String,
    pub tier: String,
}

/// Declares an agent capability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapabilityDeclaration {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Codelet;

    fn sample_execution_dag(
        name: &str,
        is_entry: bool,
        parameters: Vec<FlowParameter>,
    ) -> ExecutionDag {
        let mut dag = ExecutionDag::new();
        dag.metadata.name = Some(name.to_string());
        dag.metadata.is_entry = is_entry;
        dag.metadata.parameters = parameters;
        dag
    }

    #[test]
    fn agent_builder_and_lookup() {
        let entry = AgentFlow {
            name: "main".to_string(),
            is_entry: true,
            parameters: vec![FlowParameter {
                name: "topic".to_string(),
                type_name: "str".to_string(),
            }],
            codelet_dag: None,
            execution_dag: sample_execution_dag(
                "research.main",
                true,
                vec![FlowParameter {
                    name: "topic".to_string(),
                    type_name: "str".to_string(),
                }],
            ),
        };

        let helper = AgentFlow {
            name: "helper".to_string(),
            is_entry: false,
            parameters: vec![],
            codelet_dag: None,
            execution_dag: sample_execution_dag("research.helper", false, vec![]),
        };

        let agent = Agent::new("research").add_flow(entry).add_flow(helper);
        assert_eq!(agent.name, "research");
        assert!(agent.get_flow("helper").is_some());
        assert_eq!(agent.entry_flow().map(|f| f.name.as_str()), Some("main"));
    }

    #[test]
    fn from_codelet_dag_preserves_and_lowers() {
        let mut codelet_dag = CodeletDag::new("research.main")
            .add_codelet(Codelet::new(1, "research", "Gather references"))
            .add_codelet(Codelet::new(2, "write", "Write summary").add_dependency(1));
        codelet_dag.metadata.parameters = vec![FlowParameter {
            name: "topic".to_string(),
            type_name: "str".to_string(),
        }];

        let flow = AgentFlow::from_codelet_dag("research.main", codelet_dag, true)
            .expect("codelet DAG should lower");

        assert_eq!(flow.name, "research.main");
        assert!(flow.is_entry);
        assert!(flow.codelet_dag.is_some());
        assert_eq!(
            flow.execution_dag.metadata.name.as_deref(),
            Some("research.main")
        );
        assert_eq!(flow.execution_dag.nodes.len(), 2);
        assert_eq!(flow.parameters.len(), 1);
        assert_eq!(flow.parameters[0].name, "topic");
    }

    #[test]
    fn agent_serde_roundtrip() {
        let agent = Agent::new("analyst")
            .add_flow(AgentFlow {
                name: "main".to_string(),
                is_entry: true,
                parameters: vec![],
                codelet_dag: None,
                execution_dag: sample_execution_dag("analyst.main", true, vec![]),
            })
            .add_flow(AgentFlow {
                name: "enrich".to_string(),
                is_entry: false,
                parameters: vec![],
                codelet_dag: None,
                execution_dag: sample_execution_dag("analyst.enrich", false, vec![]),
            });

        let encoded = serde_json::to_string(&agent).expect("serialize agent");
        let restored: Agent = serde_json::from_str(&encoded).expect("deserialize agent");

        assert_eq!(restored.name, "analyst");
        assert_eq!(restored.flows.len(), 2);
        assert!(restored.entry_flow().is_some());
        assert!(restored.get_flow("enrich").is_some());
    }
}
