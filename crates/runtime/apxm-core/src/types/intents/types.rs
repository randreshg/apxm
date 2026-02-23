//! Intent classification types for APXM.
//!
//! This module contains types for intent classification and entity extraction
//! that are shared across multiple crates (chat, REPL, tooling, etc.).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// User intent classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Intent {
    /// Execute a task (default for most queries)
    ExecuteTask {
        description: String,
        #[serde(default)]
        parameters: HashMap<String, String>,
    },

    /// Execute with explicit planning step
    ExecuteWithPlan { description: String },

    /// Query memory (AAM beliefs/goals)
    QueryMemory {
        query_type: MemoryQueryType,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        key: Option<String>,
    },

    /// Update memory (AAM beliefs/goals)
    UpdateMemory { key: String, value: String },

    /// Query available capabilities
    QueryCapabilities {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filter: Option<String>,
    },

    /// Query loaded modules
    QueryModules,

    /// Load a module
    LoadModule { module_name: String },

    /// Inspect system state
    InspectState { target: InspectTarget },

    /// Visualize execution DAG
    VisualizeDAG,

    /// Export conversation or program
    Export { format: ExportFormat },

    /// Modify existing program
    ModifyProgram {
        target: String,
        modification: String,
    },

    /// Build program step-by-step
    BuildProgram { step: ProgramBuildStep },

    /// Request help
    Help {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        topic: Option<String>,
    },

    /// Request clarification
    Clarification { ambiguity: String },

    /// Meta command (mode switch, config, etc.)
    MetaCommand { command: String, args: Vec<String> },
}

impl Intent {
    /// Get a human-readable description of this intent
    pub fn description(&self) -> &str {
        match self {
            Intent::ExecuteTask { .. } => "Execute task",
            Intent::ExecuteWithPlan { .. } => "Execute with planning",
            Intent::QueryMemory { .. } => "Query memory",
            Intent::UpdateMemory { .. } => "Update memory",
            Intent::QueryCapabilities { .. } => "Query capabilities",
            Intent::QueryModules => "Query modules",
            Intent::LoadModule { .. } => "Load module",
            Intent::InspectState { .. } => "Inspect state",
            Intent::VisualizeDAG => "Visualize DAG",
            Intent::Export { .. } => "Export",
            Intent::ModifyProgram { .. } => "Modify program",
            Intent::BuildProgram { .. } => "Build program",
            Intent::Help { .. } => "Help",
            Intent::Clarification { .. } => "Clarification needed",
            Intent::MetaCommand { .. } => "Meta command",
        }
    }

    /// Check if this intent requires execution
    pub fn requires_execution(&self) -> bool {
        matches!(
            self,
            Intent::ExecuteTask { .. }
                | Intent::ExecuteWithPlan { .. }
                | Intent::ModifyProgram { .. }
                | Intent::BuildProgram { .. }
        )
    }

    /// Check if this intent is a query
    pub fn is_query(&self) -> bool {
        matches!(
            self,
            Intent::QueryMemory { .. }
                | Intent::QueryCapabilities { .. }
                | Intent::QueryModules
                | Intent::InspectState { .. }
                | Intent::Help { .. }
        )
    }
}

/// Type of memory query
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryQueryType {
    /// Query beliefs
    Belief,
    /// Query goals
    Goal,
    /// Query all memory
    All,
    /// Query episodic memory
    EpisodicMemory,
}

/// Target for inspection commands
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum InspectTarget {
    /// Inspect AAM state
    AAM,
    /// Inspect DAG
    DAG,
    /// Inspect execution state
    Execution,
    /// Inspect capabilities
    Capabilities,
    /// Inspect modules
    Modules,
}

/// Format for export
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum ExportFormat {
    /// Export as AIS code
    AIS,
    /// Export as JSON
    JSON,
    /// Export as Markdown
    Markdown,
}

/// Step in program building process
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "step_type", rename_all = "snake_case")]
pub enum ProgramBuildStep {
    /// Start building a new program
    Start { description: String },
    /// Add a step to the program
    AddStep { description: String },
    /// Add error handling
    AddErrorHandling,
    /// Finalize the program
    Finalize,
}

/// Extracted entity from user input
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entity {
    /// The text of the entity
    pub text: String,
    /// The type of entity
    pub entity_type: EntityType,
    /// Start position in the original text
    pub start: usize,
    /// End position in the original text
    pub end: usize,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

impl Entity {
    /// Create a new entity
    pub fn new(
        text: impl Into<String>,
        entity_type: EntityType,
        start: usize,
        end: usize,
        confidence: f64,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type,
            start,
            end,
            confidence,
        }
    }

    /// Check if this entity has high confidence (>= 0.8)
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }
}

/// Type of entity extracted from text
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// Name of a capability/operation
    CapabilityName,
    /// Name of a module
    ModuleName,
    /// Name of a variable
    VariableName,
    /// Key for memory access
    MemoryKey,
    /// Parameter name
    Parameter,
    /// Value literal
    Value,
    /// File path
    FilePath,
}

impl EntityType {
    /// Get a human-readable name for this entity type
    pub fn name(&self) -> &'static str {
        match self {
            EntityType::CapabilityName => "Capability",
            EntityType::ModuleName => "Module",
            EntityType::VariableName => "Variable",
            EntityType::MemoryKey => "Memory Key",
            EntityType::Parameter => "Parameter",
            EntityType::Value => "Value",
            EntityType::FilePath => "File Path",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_requires_execution() {
        let execute = Intent::ExecuteTask {
            description: "test".to_string(),
            parameters: HashMap::new(),
        };
        assert!(execute.requires_execution());

        let query = Intent::QueryModules;
        assert!(!query.requires_execution());
    }

    #[test]
    fn test_intent_is_query() {
        let query = Intent::QueryModules;
        assert!(query.is_query());

        let execute = Intent::ExecuteTask {
            description: "test".to_string(),
            parameters: HashMap::new(),
        };
        assert!(!execute.is_query());
    }

}
