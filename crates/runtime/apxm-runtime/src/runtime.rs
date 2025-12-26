//! Runtime orchestrator - Main entry point for the APxM runtime

use crate::{
    aam::Aam,
    capability::CapabilitySystem,
    executor::{ExecutionContext, ExecutionResult, ExecutorEngine, InnerPlanLinker, NoOpLinker},
    memory::{MemoryConfig, MemorySystem},
    scheduler::{DataflowScheduler, SchedulerConfig},
};
use apxm_artifact::Artifact;
use apxm_core::log_info;
use apxm_core::{
    error::RuntimeError,
    types::{
        execution::{ExecutionDag, ExecutionStats},
        values::Value,
    },
};
use apxm_backends::LLMRegistry;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

/// Result of DAG execution
#[derive(Debug, Clone)]
pub struct RuntimeExecutionResult {
    /// Final results mapped by token ID
    pub results: HashMap<u64, Value>,
    /// Execution statistics
    pub stats: ExecutionStats,
    /// LLM request metrics (when enabled)
    #[cfg(feature = "metrics")]
    pub llm_metrics: apxm_backends::AggregatedMetrics,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeConfig {
    /// Memory system configuration
    pub memory_config: MemoryConfig,
    /// Scheduler configuration
    pub scheduler_config: SchedulerConfig,
}

impl RuntimeConfig {
    /// Create configuration with in-memory LTM (for testing)
    pub fn in_memory() -> Self {
        Self {
            memory_config: MemoryConfig::in_memory_ltm(),
            scheduler_config: SchedulerConfig::default(),
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
    inner_plan_linker: Arc<dyn InnerPlanLinker>,
}

impl Runtime {
    /// Create a new runtime with the given configuration
    pub async fn new(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        log_info!("runtime", "Initializing APxM Runtime");

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

        log_info!("runtime", "APxM Runtime initialized successfully");

        Ok(Self {
            config,
            memory,
            llm_registry,
            capability_system,
            aam,
            scheduler,
            inner_plan_linker: Arc::new(NoOpLinker),
        })
    }

    /// Attach a custom inner plan linker implementation to the runtime.
    pub fn set_inner_plan_linker(&mut self, linker: Arc<dyn InnerPlanLinker>) {
        self.inner_plan_linker = linker;
    }

    /// Execute a DAG with parallel dataflow execution
    ///
    /// Note: This method does NOT support inner DAG execution (multi-level planning).
    /// If you need inner DAG support, wrap the Runtime in Arc and use
    /// `execute_with_inner_support()` instead.
    pub async fn execute(&self, dag: ExecutionDag) -> Result<RuntimeExecutionResult, RuntimeError> {
        log_info!(
            "runtime",
            nodes = dag.nodes.len(),
            "Executing DAG with parallel scheduler"
        );

        #[cfg(feature = "metrics")]
        self.llm_registry.metrics().reset();

        // Create execution context
        let mut context = ExecutionContext::new(
            Arc::clone(&self.memory),
            Arc::clone(&self.llm_registry),
            Arc::clone(&self.capability_system),
            self.aam.clone(),
        );
        context.inner_plan_linker = Arc::clone(&self.inner_plan_linker);

        // Create executor
        let executor = Arc::new(ExecutorEngine::new(context.clone()));

        // Execute with dataflow scheduler for automatic parallelism
        let (results, stats) = self.scheduler.execute(dag, executor, context).await?;

        #[cfg(feature = "metrics")]
        let llm_metrics = self.llm_registry.metrics().aggregate();

        Ok(RuntimeExecutionResult {
            results,
            stats,
            #[cfg(feature = "metrics")]
            llm_metrics,
        })
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
        log_info!(
            "runtime",
            nodes = dag.nodes.len(),
            "Executing DAG sequentially"
        );

        let mut context = ExecutionContext::new(
            Arc::clone(&self.memory),
            Arc::clone(&self.llm_registry),
            Arc::clone(&self.capability_system),
            self.aam.clone(),
        );
        context.inner_plan_linker = Arc::clone(&self.inner_plan_linker);

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

    /// Get LLM registry as Arc (clones the Arc)
    pub fn llm_registry_arc(&self) -> Arc<LLMRegistry> {
        Arc::clone(&self.llm_registry)
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
        let mut ctx = ExecutionContext::new(
            Arc::clone(&self.memory),
            Arc::clone(&self.llm_registry),
            Arc::clone(&self.capability_system),
            self.aam.clone(),
        );
        ctx.inner_plan_linker = Arc::clone(&self.inner_plan_linker);
        ctx
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
