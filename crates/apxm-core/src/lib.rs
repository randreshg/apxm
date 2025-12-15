//! APXM Core - Fundamental types and traits for the APXM system.
//!
//! This crate provides the foundational types, data structures, and traits
//! that all other APXM components depend on.

pub mod aam;
pub mod error;
pub mod logging;
pub mod memory;
pub mod traits;
pub mod types;

pub use aam::{
    AAMState, Belief, CapabilityExecutionProfile, CapabilityMetadata, CapabilitySchemas,
    EpisodicLog, Goal, GoalChange, GoalId, GoalStatus, Parameter, StateTransition,
};

pub use error::{
    common::{ErrorContext, ErrorContextExt, OpId, SourceLocation, TraceId},
    compile::CompileError,
    runtime::RuntimeError,
    security::SecurityError,
};

pub use memory::{
    EpisodicEntry, EpisodicQuery, LTMBackend, LTMQuery, LTMResult, MemorySpace, STMConfig, STMEntry,
};

pub use traits::{
    Capability, CapabilityError, ExecutionError, FinishReason, LLMBackend, LLMError, LLMRequest,
    LLMResponse, LLMResponseStream, OperationExecutor, TokenUsage,
};

pub use types::{
    AISOperation, AISOperationType, DependencyType, Edge, Node, NodeId, NodeMetadata, Number,
    Token, TokenId, TokenStatus, Value,
};

#[cfg(test)]
mod tests {
    use super::{
        AAMState, AISOperationType, Belief, CapabilityExecutionProfile, CapabilityMetadata,
        CapabilitySchemas, EpisodicEntry, EpisodicQuery, Goal, GoalChange, GoalStatus, LTMQuery,
        MemorySpace, Number, Parameter, STMConfig, STMEntry, StateTransition, Token, Value,
        error::{
            common::SourceLocation, compile::CompileError, runtime::RuntimeError,
            security::SecurityError,
        },
    };
    use std::{collections::HashMap, time::Duration};

    #[test]
    fn test_types_export() {
        let num = Number::Integer(42);
        assert!(matches!(num, Number::Integer(42)));

        let val = Value::from(true);
        assert!(matches!(val, Value::Bool(true)));

        let token = Token::new(1);
        assert_eq!(token.id, 1);

        let op = AISOperationType::Inv;
        assert_eq!(op.to_string(), "INV");
    }

    #[test]
    fn test_compile_error_export() {
        let location = SourceLocation::new("test.ais".to_string(), 3, 7);
        let err = CompileError::Parse {
            location: location.clone(),
            message: "Unexpected token".to_string(),
        };

        let display = err.to_string();
        assert!(display.contains("Parse error"));
        assert!(display.contains(&location.to_string()));
        assert!(display.contains("Unexpected token"));
    }

    #[test]
    fn test_runtime_error_from_security_error() {
        let security = SecurityError::Unauthorized {
            resource: "/secret".to_string(),
            reason: Some("Missing token".to_string()),
        };

        let runtime: RuntimeError = security.into();
        let display = runtime.to_string();
        assert!(matches!(runtime, RuntimeError::Security(_)));
        assert!(display.contains("Security error"));
        assert!(display.contains("Missing token"));
    }

    #[test]
    fn test_runtime_timeout_error_display() {
        let err = RuntimeError::Timeout {
            op_id: 7,
            timeout: Duration::from_secs(5),
        };

        let display = err.to_string();
        assert!(display.contains("Timeout"));
        assert!(display.contains("7"));
        assert!(display.contains("5s"));
    }

    #[test]
    fn test_aam_state_exports() {
        let state = AAMState::new();
        state.set_belief("status".to_string(), Value::String("idle".to_string()));

        let belief = Belief::new("flag".to_string(), Value::Bool(true));
        assert_eq!(belief.key(), "flag");

        let goal = Goal::new(1, "Demo".to_string(), 5);
        state.add_goal(goal.clone()).expect("goal added");
        let peeked = state
            .peek_next_goal()
            .expect("goal queue accessible")
            .expect("goal present");
        assert_eq!(peeked.id, goal.id);

        let change = GoalChange::StatusChanged(goal.id, GoalStatus::Pending, GoalStatus::Active);
        let mut before: HashMap<String, Value> = HashMap::new();
        before.insert("status".to_string(), Value::String("idle".to_string()));
        state
            .record_transition(
                AISOperationType::Plan,
                before,
                vec![change],
                Duration::from_millis(10),
            )
            .expect("transition recorded");
        assert_eq!(state.transition_count().expect("transition count"), 1);
    }

    #[test]
    fn test_capability_and_transition_exports() {
        let param_schema = serde_json::json!({"type": "string"});
        let param = Parameter::with_schema(
            "query".to_string(),
            "string".to_string(),
            true,
            param_schema.clone(),
        );
        assert!(param.required);
        assert_eq!(param.schema, param_schema);

        let schema = serde_json::json!({"type": "object"});
        let schemas = CapabilitySchemas::new(schema.clone(), schema.clone());
        let profile = CapabilityExecutionProfile::new(true, 2.5, 42);
        let mut capability = CapabilityMetadata::with_details(
            "search".to_string(),
            "Search capability".to_string(),
            "1.0".to_string(),
            schemas,
            profile,
            vec!["utility".to_string()],
        );
        capability.add_tag("utility".to_string());
        assert!(capability.has_tag("utility"));
        assert!(capability.requires_sandbox);

        let mut before = HashMap::new();
        before.insert("input".to_string(), Value::String("rust".to_string()));
        let mut after = before.clone();
        after.insert("result".to_string(), Value::Bool(true));
        let goal = Goal::new(9, "Find info".to_string(), 1);
        let mut transition = StateTransition::with_capability(
            AISOperationType::Inv,
            before,
            after,
            vec![GoalChange::Added(goal)],
            "search".to_string(),
            Duration::from_millis(5),
        );
        transition.add_metadata("note".to_string(), Value::String("ok".to_string()));
        assert_eq!(transition.capability_used.as_deref(), Some("search"));
        assert_eq!(
            transition.get_metadata("note"),
            Some(&Value::String("ok".to_string()))
        );
    }

    #[test]
    fn test_memory_exports() {
        let space = MemorySpace::Ltm;
        assert!(space.is_ltm());

        let ttl = Duration::from_secs(60);
        let entry = STMEntry::new("k", Value::Bool(true), ttl);
        assert_eq!(entry.key, "k");

        let cfg = STMConfig::default();
        assert!(cfg.max_size > 0);

        let query = LTMQuery::new("hello");
        assert!(query.validate().is_ok());

        let episode = EpisodicEntry::new(
            "exec",
            vec![Value::String("op".into())],
            vec![Value::Bool(true)],
        );
        let filter = EpisodicQuery::new().with_execution_id("exec");
        assert!(filter.matches(&episode));
    }
}
