//! Runtime orchestrator - Main entry point for the APxM runtime

use crate::{
    aam::Aam,
    capability::CapabilitySystem,
    executor::{ExecutionContext, ExecutionResult, ExecutorEngine},
    memory::{MemoryConfig, MemorySystem},
    scheduler::{DataflowScheduler, SchedulerConfig},
};
use apxm_artifact::Artifact;
use apxm_core::{
    error::RuntimeError,
    types::{
        execution::{ExecutionDag, ExecutionStats},
        values::Value,
    },
};
use apxm_models::registry::LLMRegistry;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

/// Result of DAG execution
#[derive(Debug, Clone)]
pub struct RuntimeExecutionResult {
    /// Final results mapped by token ID
    pub results: HashMap<u64, Value>,
    /// Execution statistics
    pub stats: ExecutionStats,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Memory system configuration
    pub memory_config: MemoryConfig,
    /// Scheduler configuration
    pub scheduler_config: SchedulerConfig,
    /// Enable tracing (default: true)
    pub enable_tracing: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            memory_config: MemoryConfig::default(),
            scheduler_config: SchedulerConfig::default(),
            enable_tracing: true,
        }
    }
}

impl RuntimeConfig {
    /// Create configuration with in-memory LTM (for testing)
    pub fn in_memory() -> Self {
        Self {
            memory_config: MemoryConfig::in_memory_ltm(),
            scheduler_config: SchedulerConfig::default(),
            enable_tracing: false,
        }
    }

    /// Set scheduler configuration
    pub fn with_scheduler_config(mut self, config: SchedulerConfig) -> Self {
        self.scheduler_config = config;
        self
    }
}

/// APxM Runtime - Main orchestrator
///
/// The runtime coordinates all subsystems and provides the main API
/// for executing APxM programs.
pub struct Runtime {
    config: RuntimeConfig,
    memory: Arc<MemorySystem>,
    llm_registry: Arc<LLMRegistry>,
    capability_system: Arc<CapabilitySystem>,
    aam: Aam,
    scheduler: DataflowScheduler,
}

impl Runtime {
    /// Create a new runtime with the given configuration
    pub async fn new(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        // Initialize tracing if enabled
        if config.enable_tracing {
            tracing_subscriber::fmt()
                .with_target(false)
                .with_level(true)
                .try_init()
                .ok(); // Ignore if already initialized
        }

        tracing::info!("Initializing APxM Runtime");

        // Initialize memory system
        let memory = Arc::new(
            MemorySystem::new(config.memory_config.clone())
                .await
                .map_err(|e| RuntimeError::State(format!("Failed to initialize memory: {}", e)))?,
        );

        // Initialize LLM registry
        let llm_registry = Arc::new(LLMRegistry::new());

        // Initialize AAM and capability system
        let aam = Aam::new();
        let capability_system = Arc::new(CapabilitySystem::with_aam(aam.clone()));

        // Initialize scheduler
        let scheduler = DataflowScheduler::new(config.scheduler_config.clone());

        tracing::info!("APxM Runtime initialized successfully");

        Ok(Self {
            config,
            memory,
            llm_registry,
            capability_system,
            aam,
            scheduler,
        })
    }

    /// Execute a DAG with parallel dataflow execution
    ///
    /// Note: This method does NOT support inner DAG execution (multi-level planning).
    /// If you need inner DAG support, wrap the Runtime in Arc and use
    /// `execute_with_inner_support()` instead.
    pub async fn execute(&self, dag: ExecutionDag) -> Result<RuntimeExecutionResult, RuntimeError> {
        tracing::info!(
            nodes = dag.nodes.len(),
            "Executing DAG with parallel scheduler"
        );

        // Create execution context
        let context = ExecutionContext::new(
            Arc::clone(&self.memory),
            Arc::clone(&self.llm_registry),
            Arc::clone(&self.capability_system),
            self.aam.clone(),
        );

        // Create executor
        let executor = Arc::new(ExecutorEngine::new(context.clone()));

        // Execute with dataflow scheduler for automatic parallelism
        let (results, stats) = self.scheduler.execute(dag, executor, context).await?;

        Ok(RuntimeExecutionResult { results, stats })
    }

    /// Execute a serialized artifact
    pub async fn execute_artifact(
        &self,
        artifact: Artifact,
    ) -> Result<RuntimeExecutionResult, RuntimeError> {
        self.execute(artifact.into_dag()).await
    }

    /// Execute an artifact from raw bytes
    pub async fn execute_artifact_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<RuntimeExecutionResult, RuntimeError> {
        let artifact = Artifact::from_bytes(bytes)
            .map_err(|e| RuntimeError::State(format!("Artifact parse error: {e}")))?;
        self.execute_artifact(artifact).await
    }

    /// Execute a DAG sequentially (for testing/debugging)
    pub async fn execute_sequential(
        &self,
        dag: ExecutionDag,
    ) -> Result<ExecutionResult, RuntimeError> {
        tracing::info!(nodes = dag.nodes.len(), "Executing DAG sequentially");

        let context = ExecutionContext::new(
            Arc::clone(&self.memory),
            Arc::clone(&self.llm_registry),
            Arc::clone(&self.capability_system),
            self.aam.clone(),
        );

        let executor = ExecutorEngine::new(context);
        executor.execute_dag(dag).await
    }

    /// Get memory system reference
    pub fn memory(&self) -> &MemorySystem {
        &self.memory
    }

    /// Get LLM registry reference
    pub fn llm_registry(&self) -> &LLMRegistry {
        &self.llm_registry
    }

    /// Get capability system reference
    pub fn capability_system(&self) -> &CapabilitySystem {
        &self.capability_system
    }

    /// Get AAM handle
    pub fn aam(&self) -> &Aam {
        &self.aam
    }

    /// Get configuration
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Create a new execution context
    pub fn create_context(&self) -> ExecutionContext {
        ExecutionContext::new(
            Arc::clone(&self.memory),
            Arc::clone(&self.llm_registry),
            Arc::clone(&self.capability_system),
            self.aam.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxm_core::types::{
        execution::{Node, NodeMetadata},
        operations::AISOperationType,
        values::Value,
    };
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_runtime_creation() {
        let config = RuntimeConfig::in_memory();
        let runtime = Runtime::new(config).await.unwrap();

        assert!(runtime.memory().stm().len().await == 0);
    }

    #[tokio::test]
    async fn test_runtime_execute_simple_dag() {
        let config = RuntimeConfig::in_memory();
        let runtime = Runtime::new(config).await.unwrap();

        // Create a simple DAG with one CONST_STR node
        let mut node = Node {
            id: 1,
            op_type: AISOperationType::ConstStr,
            attributes: HashMap::new(),
            input_tokens: vec![],
            output_tokens: vec![100],
            metadata: NodeMetadata::default(),
        };
        node.attributes.insert(
            "value".to_string(),
            Value::String("Hello from runtime!".to_string()),
        );

        let dag = ExecutionDag {
            nodes: vec![node],
            edges: vec![],
            entry_nodes: vec![1],
            exit_nodes: vec![1],
            metadata: Default::default(),
        };

        let result = runtime.execute(dag).await.unwrap();

        assert_eq!(result.stats.executed_nodes, 1);
        assert_eq!(result.stats.failed_nodes, 0);
        assert!(result.results.contains_key(&100));
    }

    #[tokio::test]
    async fn test_runtime_memory_persistence() {
        let config = RuntimeConfig::in_memory();
        let runtime = Runtime::new(config).await.unwrap();

        // Write to memory
        runtime
            .memory()
            .write(
                crate::memory::MemorySpace::Stm,
                "test_key".to_string(),
                Value::String("test_value".to_string()),
            )
            .await
            .unwrap();

        // Read back
        let result = runtime
            .memory()
            .read(crate::memory::MemorySpace::Stm, "test_key")
            .await
            .unwrap();

        assert_eq!(result, Some(Value::String("test_value".to_string())));
    }
}
