//! Unified Plan structures shared across the system
//!
//! This module provides a single source of truth for plan-related types,
//! eliminating the previous inconsistency between OuterPlan (chat) and
//! PlanOutput (runtime).

use serde::{Deserialize, Serialize};

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
/// Represents APXM DSL code that should be compiled and executed
/// as part of multi-level planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerPlanDsl {
    /// Raw APXM DSL code
    pub dsl: String,
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
        self.inner_plan = Some(InnerPlanDsl { dsl });
        self
    }

    /// Check if this plan has an inner plan
    pub fn has_inner_plan(&self) -> bool {
        self.inner_plan.is_some()
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
    fn test_plan_creation() {
        let step1 = PlanStep::new("Analyze requirements".to_string(), 100);
        let step2 = PlanStep::new("Implement solution".to_string(), 90)
            .with_dependency("Analyze requirements".to_string());

        let plan = Plan::new(vec![step1, step2], "Complete analysis and implementation".to_string());

        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].priority, 100);
        assert_eq!(plan.steps[1].dependencies.len(), 1);
        assert!(!plan.has_inner_plan());
    }

    #[test]
    fn test_plan_with_inner_dsl() {
        let plan = Plan::new(vec![], "Test plan".to_string())
            .with_inner_plan("agent Test { on Message { user, text } => { return text; } }".to_string());

        assert!(plan.has_inner_plan());
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
}
