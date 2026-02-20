//! Runtime orchestrator - Main entry point for the APxM runtime

use crate::{
    aam::Aam,
    capability::{CapabilitySystem, flow_registry::FlowRegistry},
    executor::{ExecutionContext, ExecutionResult, ExecutorEngine, InnerPlanLinker, NoOpLinker},
    memory::{MemoryConfig, MemorySystem},
    scheduler::{DataflowScheduler, SchedulerConfig},
};
use apxm_artifact::Artifact;
use apxm_backends::LLMRegistry;
use apxm_core::log_info;
use apxm_core::{
    error::RuntimeError,
    types::{
        execution::{ExecutionDag, ExecutionStats},
        values::Value,
    },
};
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
    /// Scheduler overhead metrics
    pub scheduler_metrics: crate::observability::SchedulerMetrics,
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
    flow_registry: Arc<FlowRegistry>,
    aam: Aam,
    scheduler: DataflowScheduler,
    inner_plan_linker: Arc<dyn InnerPlanLinker>,
    instruction_config: apxm_core::InstructionConfig,
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

        // NOTE: No default capabilities are registered here.
        // Capabilities should be registered by the executor based on what
        // the workflow declares in its `tools: [...]` list.

        // Initialize flow registry for cross-agent flow calls
        let flow_registry = Arc::new(FlowRegistry::new());

        // Initialize scheduler
        let scheduler = DataflowScheduler::new(config.scheduler_config.clone());

        log_info!("runtime", "APxM Runtime initialized successfully");

        Ok(Self {
            config,
            memory,
            llm_registry,
            capability_system,
            flow_registry,
            aam,
            scheduler,
            inner_plan_linker: Arc::new(NoOpLinker),
            instruction_config: apxm_core::InstructionConfig::default(),
        })
    }

    /// Attach a custom inner plan linker implementation to the runtime.
    pub fn set_inner_plan_linker(&mut self, linker: Arc<dyn InnerPlanLinker>) {
        self.inner_plan_linker = linker;
    }

    /// Set the instruction configuration for system prompts.
    ///
    /// The instruction config is used by LLM handlers to get system prompts
    /// for operations like ask, think, reason, plan, and reflect.
    pub fn set_instruction_config(&mut self, config: apxm_core::InstructionConfig) {
        self.instruction_config = config;
    }

    /// Get the instruction configuration.
    pub fn instruction_config(&self) -> &apxm_core::InstructionConfig {
        &self.instruction_config
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
        let context = ExecutionContext {
            execution_id: uuid::Uuid::now_v7().to_string(),
            session_id: None,
            memory: Arc::clone(&self.memory),
            llm_registry: Arc::clone(&self.llm_registry),
            capability_system: Arc::clone(&self.capability_system),
            aam: self.aam.clone(),
            inner_plan_linker: Arc::clone(&self.inner_plan_linker),
            dag_splicer: Arc::new(crate::executor::NoOpSplicer),
            flow_registry: Arc::clone(&self.flow_registry),
            instruction_config: self.instruction_config.clone(),
            start_time: std::time::Instant::now(),
            metadata: std::collections::HashMap::new(),
        };

        // Create executor
        let executor = Arc::new(ExecutorEngine::new(context.clone()));

        // Execute with dataflow scheduler for automatic parallelism
        let (results, stats, scheduler_metrics) = self
            .scheduler
            .execute(dag, executor, context, vec![])
            .await?;

        #[cfg(feature = "metrics")]
        let llm_metrics = self.llm_registry.metrics().aggregate();

        Ok(RuntimeExecutionResult {
            results,
            stats,
            #[cfg(feature = "metrics")]
            llm_metrics,
            scheduler_metrics,
        })
    }

    /// Execute a serialized artifact
    pub async fn execute_artifact(
        &self,
        artifact: Artifact,
    ) -> Result<RuntimeExecutionResult, RuntimeError> {
        self.execute(artifact.into_dag()).await
    }

    /// Execute a serialized artifact with automatic entry point detection and flow registration.
    ///
    /// This method:
    /// 1. Checks for an `@entry` DAG in the artifact
    /// 2. Auto-registers all other flows in the FlowRegistry
    /// 3. Executes the @entry DAG
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::State` if the artifact does not have an `@entry` flow.
    pub async fn execute_artifact_auto(
        &self,
        artifact: Artifact,
    ) -> Result<RuntimeExecutionResult, RuntimeError> {
        // 1. Find @entry DAG
        let entry_dag = artifact.entry_dag().cloned().ok_or_else(|| {
            let name = artifact
                .dags()
                .first()
                .and_then(|d| d.metadata.name.as_deref())
                .unwrap_or("<unnamed>");
            RuntimeError::State(format!(
                "No @entry flow found in artifact '{}'. Mark a flow with @entry to designate the entry point.",
                name
            ))
        })?;

        // 2. Auto-register all non-entry flows
        let num_registered = {
            let mut count = 0;
            for dag in artifact.flow_dags() {
                if let Some(ref name) = dag.metadata.name {
                    let (agent_name, flow_name) = parse_flow_name(name);
                    log_info!(
                        "runtime",
                        agent = %agent_name,
                        flow = %flow_name,
                        "Auto-registering flow from artifact"
                    );
                    self.flow_registry
                        .register_flow(&agent_name, &flow_name, dag.clone());
                    count += 1;
                }
            }
            count
        };

        log_info!(
            "runtime",
            name = entry_dag.metadata.name.as_deref().unwrap_or("<unnamed>"),
            num_flows = artifact.dags().len(),
            "Executing @entry flow with {} registered flows",
            num_registered
        );

        // 3. Execute @entry DAG
        self.execute(entry_dag).await
    }

    /// Execute an artifact with provided arguments for entry flow parameters.
    ///
    /// This method validates that the number of arguments matches the entry flow's
    /// parameter count, then injects the argument values into the entry tokens.
    pub async fn execute_artifact_with_args(
        &self,
        artifact: Artifact,
        args: Vec<String>,
    ) -> Result<RuntimeExecutionResult, RuntimeError> {
        // 1. Find @entry DAG
        let entry_dag = artifact.entry_dag().cloned().ok_or_else(|| {
            let name = artifact
                .dags()
                .first()
                .and_then(|d| d.metadata.name.as_deref())
                .unwrap_or("<unnamed>");
            RuntimeError::State(format!(
                "No @entry flow found in artifact '{}'. Mark a flow with @entry to designate the entry point.",
                name
            ))
        })?;

        // 2. Validate argument count against parameters
        let params = &entry_dag.metadata.parameters;
        if args.len() != params.len() {
            let flow_name = entry_dag.metadata.name.as_deref().unwrap_or("<unnamed>");
            let param_desc = if params.is_empty() {
                "no parameters".to_string()
            } else {
                params
                    .iter()
                    .map(|p| format!("{}: {}", p.name, p.type_name))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(RuntimeError::State(format!(
                "Flow '{}' requires {} argument(s) ({}) but {} were provided",
                flow_name,
                params.len(),
                param_desc,
                args.len()
            )));
        }

        // 3. Auto-register all non-entry flows
        for dag in artifact.flow_dags() {
            if let Some(ref name) = dag.metadata.name {
                let (agent_name, flow_name) = parse_flow_name(name);
                log_info!(
                    "runtime",
                    agent = %agent_name,
                    flow = %flow_name,
                    "Auto-registering flow from artifact"
                );
                self.flow_registry
                    .register_flow(&agent_name, &flow_name, dag.clone());
            }
        }

        // 4. Convert args to Values and execute with inputs
        let arg_values: Vec<Value> = args.into_iter().map(Value::String).collect();

        log_info!(
            "runtime",
            nodes = entry_dag.nodes.len(),
            inputs = arg_values.len(),
            "Executing DAG with {} input values",
            arg_values.len()
        );

        #[cfg(feature = "metrics")]
        self.llm_registry.metrics().reset();

        let context = ExecutionContext {
            execution_id: uuid::Uuid::now_v7().to_string(),
            session_id: None,
            memory: Arc::clone(&self.memory),
            llm_registry: Arc::clone(&self.llm_registry),
            capability_system: Arc::clone(&self.capability_system),
            aam: self.aam.clone(),
            inner_plan_linker: Arc::clone(&self.inner_plan_linker),
            dag_splicer: Arc::new(crate::executor::NoOpSplicer),
            flow_registry: Arc::clone(&self.flow_registry),
            instruction_config: self.instruction_config.clone(),
            start_time: std::time::Instant::now(),
            metadata: std::collections::HashMap::new(),
        };

        let executor = Arc::new(ExecutorEngine::new(context.clone()));

        let (results, stats, scheduler_metrics) = self
            .scheduler
            .execute(entry_dag, executor, context, arg_values)
            .await?;

        Ok(RuntimeExecutionResult {
            results,
            stats,
            #[cfg(feature = "metrics")]
            llm_metrics: self.llm_registry.metrics().aggregate(),
            scheduler_metrics,
        })
    }

    /// Execute an artifact with positional arguments and an optional session identifier.
    ///
    /// Identical to [`execute_artifact_with_args`] but forwards `session_id` into the
    /// [`ExecutionContext`] so downstream capability executors can correlate invocations.
    pub async fn execute_artifact_with_session(
        &self,
        artifact: Artifact,
        args: Vec<String>,
        session_id: Option<String>,
    ) -> Result<RuntimeExecutionResult, RuntimeError> {
        // 1. Find @entry DAG
        let entry_dag = artifact.entry_dag().cloned().ok_or_else(|| {
            let name = artifact
                .dags()
                .first()
                .and_then(|d| d.metadata.name.as_deref())
                .unwrap_or("<unnamed>");
            RuntimeError::State(format!(
                "No @entry flow found in artifact '{}'. Mark a flow with @entry to designate the entry point.",
                name
            ))
        })?;

        // 2. Validate argument count against parameters
        let params = &entry_dag.metadata.parameters;
        if args.len() != params.len() {
            let flow_name = entry_dag.metadata.name.as_deref().unwrap_or("<unnamed>");
            let param_desc = if params.is_empty() {
                "no parameters".to_string()
            } else {
                params
                    .iter()
                    .map(|p| format!("{}: {}", p.name, p.type_name))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(RuntimeError::State(format!(
                "Flow '{}' requires {} argument(s) ({}) but {} were provided",
                flow_name,
                params.len(),
                param_desc,
                args.len()
            )));
        }

        // 3. Auto-register all non-entry flows
        for dag in artifact.flow_dags() {
            if let Some(ref name) = dag.metadata.name {
                let (agent_name, flow_name) = parse_flow_name(name);
                self.flow_registry
                    .register_flow(&agent_name, &flow_name, dag.clone());
            }
        }

        // 4. Convert args to Values and execute with inputs
        let arg_values: Vec<Value> = args.into_iter().map(Value::String).collect();

        #[cfg(feature = "metrics")]
        self.llm_registry.metrics().reset();

        let context = ExecutionContext {
            execution_id: uuid::Uuid::now_v7().to_string(),
            session_id,
            memory: Arc::clone(&self.memory),
            llm_registry: Arc::clone(&self.llm_registry),
            capability_system: Arc::clone(&self.capability_system),
            aam: self.aam.clone(),
            inner_plan_linker: Arc::clone(&self.inner_plan_linker),
            dag_splicer: Arc::new(crate::executor::NoOpSplicer),
            flow_registry: Arc::clone(&self.flow_registry),
            instruction_config: self.instruction_config.clone(),
            start_time: std::time::Instant::now(),
            metadata: std::collections::HashMap::new(),
        };

        let executor = Arc::new(ExecutorEngine::new(context.clone()));

        let (results, stats, scheduler_metrics) = self
            .scheduler
            .execute(entry_dag, executor, context, arg_values)
            .await?;

        Ok(RuntimeExecutionResult {
            results,
            stats,
            #[cfg(feature = "metrics")]
            llm_metrics: self.llm_registry.metrics().aggregate(),
            scheduler_metrics,
        })
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

    /// Execute an artifact from raw bytes with entry point validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the artifact does not have an `@entry` flow.
    pub async fn execute_artifact_bytes_auto(
        &self,
        bytes: &[u8],
    ) -> Result<RuntimeExecutionResult, RuntimeError> {
        let artifact = Artifact::from_bytes(bytes)
            .map_err(|e| RuntimeError::State(format!("Artifact parse error: {e}")))?;
        self.execute_artifact_auto(artifact).await
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

        let context = ExecutionContext {
            execution_id: uuid::Uuid::now_v7().to_string(),
            session_id: None,
            memory: Arc::clone(&self.memory),
            llm_registry: Arc::clone(&self.llm_registry),
            capability_system: Arc::clone(&self.capability_system),
            aam: self.aam.clone(),
            inner_plan_linker: Arc::clone(&self.inner_plan_linker),
            dag_splicer: Arc::new(crate::executor::NoOpSplicer),
            flow_registry: Arc::clone(&self.flow_registry),
            instruction_config: self.instruction_config.clone(),
            start_time: std::time::Instant::now(),
            metadata: std::collections::HashMap::new(),
        };

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
        ExecutionContext {
            execution_id: uuid::Uuid::now_v7().to_string(),
            session_id: None,
            memory: Arc::clone(&self.memory),
            llm_registry: Arc::clone(&self.llm_registry),
            capability_system: Arc::clone(&self.capability_system),
            aam: self.aam.clone(),
            inner_plan_linker: Arc::clone(&self.inner_plan_linker),
            dag_splicer: Arc::new(crate::executor::NoOpSplicer),
            flow_registry: Arc::clone(&self.flow_registry),
            instruction_config: self.instruction_config.clone(),
            start_time: std::time::Instant::now(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Get flow registry reference
    pub fn flow_registry(&self) -> &FlowRegistry {
        &self.flow_registry
    }

    /// Get flow registry as Arc (clones the Arc)
    pub fn flow_registry_arc(&self) -> Arc<FlowRegistry> {
        Arc::clone(&self.flow_registry)
    }
}

/// Parse flow name in "Agent.flow" format
fn parse_flow_name(name: &str) -> (String, String) {
    if let Some((agent, flow)) = name.split_once('.') {
        (agent.to_string(), flow.to_string())
    } else {
        ("default".to_string(), name.to_string())
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
