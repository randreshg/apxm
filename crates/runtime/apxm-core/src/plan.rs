//! Unified Plan structures shared across the system
//!
//! This module provides a single source of truth for plan-related types,
//! eliminating the previous inconsistency between OuterPlan (chat) and
//! PlanOutput (runtime).

use serde::{Deserialize, Serialize};

use crate::types::execution::CodeletDag;

/// Structured plan with steps, result summary, and optional inner plan
///
/// This structure is used throughout the system:
/// - In planning workflows that generate AIS subgraphs
/// - In apxm-runtime PLAN operation for LLM-based planning
/// - Supports multi-level planning with inner plan execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Ordered list of plan steps
    #[serde(rename = "plan")]
    pub steps: Vec<PlanStep>,

    /// Summary of what will be accomplished
    pub result: String,

    /// Optional inner plan (DSL code to be compiled and executed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_plan: Option<InnerPlanDsl>,
}

/// A single step in a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Clear, actionable step description
    pub description: String,

    /// Execution priority (0-100, higher = more urgent)
    #[serde(default)]
    pub priority: u32,

    /// List of step descriptions that must complete first
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<String>,
}

/// Inner plan DSL code
///
/// Represents either APXM DSL code or a structured `CodeletDag` that should
/// be compiled and executed as part of multi-level planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerPlanDsl {
    /// Raw APXM DSL code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsl: Option<String>,
    /// Optional structured codelet DAG.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codelet_dag: Option<CodeletDag>,
}

impl Plan {
    /// Create a new plan with steps and result
    pub fn new(steps: Vec<PlanStep>, result: String) -> Self {
        Self {
            steps,
            result,
            inner_plan: None,
        }
    }

    /// Create a plan with inner plan DSL
    pub fn with_inner_plan(mut self, dsl: String) -> Self {
        self.inner_plan = Some(InnerPlanDsl {
            dsl: Some(dsl),
            codelet_dag: None,
        });
        self
    }

    /// Create a plan with structured inner codelet DAG.
    pub fn with_inner_codelet_dag(mut self, codelet_dag: CodeletDag) -> Self {
        self.inner_plan = Some(InnerPlanDsl {
            dsl: None,
            codelet_dag: Some(codelet_dag),
        });
        self
    }

    /// Check if this plan has an inner plan
    pub fn has_inner_plan(&self) -> bool {
        self.inner_plan
            .as_ref()
            .map(InnerPlanDsl::has_payload)
            .unwrap_or(false)
    }
}

impl InnerPlanDsl {
    /// Returns true when this inner plan contains either DSL or a codelet DAG.
    pub fn has_payload(&self) -> bool {
        self.dsl
            .as_ref()
            .map(|dsl| !dsl.trim().is_empty())
            .unwrap_or(false)
            || self.codelet_dag.is_some()
    }
}

impl PlanStep {
    /// Create a new plan step
    pub fn new(description: String, priority: u32) -> Self {
        Self {
            description,
            priority,
            dependencies: Vec::new(),
        }
    }

    /// Add a dependency to this step
    pub fn with_dependency(mut self, dep: String) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Add multiple dependencies
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies.extend(deps);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_with_inner_dsl() {
        let plan = Plan::new(vec![], "Test plan".to_string()).with_inner_plan(
            "agent Test { on Message { user, text } => { return text; } }".to_string(),
        );

        assert!(plan.has_inner_plan());
        assert!(plan.inner_plan.unwrap().dsl.is_some());
    }

    #[test]
    fn test_plan_serialization() {
        let step = PlanStep::new("Test step".to_string(), 50);
        let plan = Plan::new(vec![step], "Test plan".to_string());

        let json = serde_json::to_string(&plan).unwrap();
        assert!(json.contains("Test step"));
        assert!(json.contains("Test plan"));

        let deserialized: Plan = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.steps.len(), 1);
        assert_eq!(deserialized.result, "Test plan");
    }

    #[test]
    fn test_plan_with_inner_codelet_dag() {
        let dag = CodeletDag::new("inner");
        let plan = Plan::new(vec![], "Test plan".to_string()).with_inner_codelet_dag(dag);
        assert!(plan.has_inner_plan());
        assert!(plan.inner_plan.unwrap().codelet_dag.is_some());
    }
}
