//! Execution context - Holds runtime state and provides access to subsystems

use crate::{aam::Aam, capability::CapabilitySystem, memory::MemorySystem};
use apxm_models::registry::LLMRegistry;
use std::sync::Arc;

use super::dag_splicer::{DagSplicer, NoOpSplicer};
use super::inner_plan_linker::{InnerPlanLinker, NoOpLinker};

/// Execution context passed to all operation handlers
///
/// Provides access to:
/// - Memory system (STM, LTM, Episodic)
/// - LLM registry for reasoning operations
/// - Capability system for tool invocation
/// - Inner plan linker for compiling DSL during execution
/// - DAG splicer for dynamic inner/outer plan unification
/// - Execution metadata (ID, session, etc.)
#[derive(Clone)]
pub struct ExecutionContext {
    /// Unique execution ID for tracing
    pub execution_id: String,
    /// Optional session ID for multi-turn interactions
    pub session_id: Option<String>,
    /// Memory system (3-tier)
    pub memory: Arc<MemorySystem>,
    /// LLM registry for reasoning operations
    pub llm_registry: Arc<LLMRegistry>,
    /// Capability system for tool invocation
    pub capability_system: Arc<CapabilitySystem>,
    /// Agent Abstract Machine state handle
    pub aam: Aam,
    /// Inner plan linker for compiling DSL from LLMs
    pub inner_plan_linker: Arc<dyn InnerPlanLinker>,
    /// DAG splicer for dynamic inner/outer plan unification
    pub dag_splicer: Arc<dyn DagSplicer>,
    /// Start time of execution (for timing)
    pub start_time: std::time::Instant,
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl ExecutionContext {
    /// Create a new execution context with no-op inner plan support
    pub fn new(
        memory: Arc<MemorySystem>,
        llm_registry: Arc<LLMRegistry>,
        capability_system: Arc<CapabilitySystem>,
        aam: Aam,
    ) -> Self {
        Self {
            execution_id: uuid::Uuid::now_v7().to_string(),
            session_id: None,
            memory,
            llm_registry,
            capability_system,
            aam,
            inner_plan_linker: Arc::new(NoOpLinker),
            dag_splicer: Arc::new(NoOpSplicer),
            start_time: std::time::Instant::now(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Create execution context with full inner plan support (linker + splicer)
    pub fn with_inner_plan_support(
        memory: Arc<MemorySystem>,
        llm_registry: Arc<LLMRegistry>,
        capability_system: Arc<CapabilitySystem>,
        aam: Aam,
        inner_plan_linker: Arc<dyn InnerPlanLinker>,
        dag_splicer: Arc<dyn DagSplicer>,
    ) -> Self {
        Self {
            execution_id: uuid::Uuid::now_v7().to_string(),
            session_id: None,
            memory,
            llm_registry,
            capability_system,
            aam,
            inner_plan_linker,
            dag_splicer,
            start_time: std::time::Instant::now(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Get a reference to the memory subsystem
    pub fn memory(&self) -> &MemorySystem {
        &self.memory
    }

    /// Create context with specific execution ID
    pub fn with_execution_id(mut self, execution_id: String) -> Self {
        self.execution_id = execution_id;
        self
    }

    /// Create context with session ID
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get elapsed time since execution started
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Create a child context with new execution ID
    pub fn child(&self) -> Self {
        Self {
            execution_id: uuid::Uuid::now_v7().to_string(),
            session_id: self.session_id.clone(),
            memory: Arc::clone(&self.memory),
            llm_registry: Arc::clone(&self.llm_registry),
            capability_system: Arc::clone(&self.capability_system),
            aam: self.aam.clone(),
            inner_plan_linker: Arc::clone(&self.inner_plan_linker),
            dag_splicer: Arc::clone(&self.dag_splicer),
            start_time: std::time::Instant::now(),
            metadata: self.metadata.clone(),
        }
    }

    pub fn aam(&self) -> &Aam {
        &self.aam
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryConfig;

    #[tokio::test]
    async fn test_execution_context_creation() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());

        let ctx = ExecutionContext::new(memory, llm_registry, capability_system, Aam::new());

        assert!(!ctx.execution_id.is_empty());
        assert!(ctx.session_id.is_none());
        assert!(ctx.metadata.is_empty());
    }

    #[tokio::test]
    async fn test_execution_context_with_session() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());

        let ctx = ExecutionContext::new(memory, llm_registry, capability_system, Aam::new())
            .with_session_id("session_123".to_string())
            .with_metadata("key".to_string(), "value".to_string());

        assert_eq!(ctx.session_id, Some("session_123".to_string()));
        assert_eq!(ctx.metadata.get("key"), Some(&"value".to_string()));
    }

    #[tokio::test]
    async fn test_child_context() {
        let memory = Arc::new(
            MemorySystem::new(MemoryConfig::in_memory_ltm())
                .await
                .unwrap(),
        );
        let llm_registry = Arc::new(LLMRegistry::new());
        let capability_system = Arc::new(CapabilitySystem::new());

        let parent = ExecutionContext::new(memory, llm_registry, capability_system, Aam::new())
            .with_session_id("session_123".to_string());
        let child = parent.child();

        // Should have different execution ID
        assert_ne!(child.execution_id, parent.execution_id);
        // Should inherit session ID
        assert_eq!(child.session_id, parent.session_id);
    }
}
