//! APxM Runtime - Production-ready execution engine for APxM programs
//!
//! The runtime provides:
//! - **Memory System**: Three-tier memory (STM, LTM, Episodic)
//! - **Executor**: Operation dispatch and execution
//! - **Observability**: Tracing and metrics integration
//!
//! # Example
//!
//! ```no_run
//! use apxm_runtime::{Runtime, RuntimeConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = RuntimeConfig::default();
//!     let runtime = Runtime::new(config).await?;
//!
//!     // Execute a DAG
//!     // let result = runtime.execute(dag).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod aam;
pub mod capability;
pub mod executor;
pub mod memory;
pub mod observability;
mod runtime;
pub mod scheduler;

pub use aam::{
    Aam, CapabilityRecord, Goal, GoalId, GoalStatus, STAGED_BELIEF_PREFIX, TransitionLabel,
    effects::{AamComponent, OperationEffects, operation_effects},
};
pub use capability::{CapabilitySystem, flow_registry::FlowRegistry};
pub use executor::{ExecutionContext, ExecutorEngine, InnerPlanLinker, NoOpLinker};
pub use memory::{MemoryConfig, MemorySpace, MemorySystem};
pub use observability::{MetricsCollector, SchedulerMetrics};
pub use runtime::{Runtime, RuntimeConfig, RuntimeExecutionResult};
pub use scheduler::{DataflowScheduler, SchedulerConfig};

pub type RuntimeResult<T> = std::result::Result<T, RuntimeError>;

// Re-export commonly used types
pub use apxm_core::{
    error::RuntimeError,
    types::{
        execution::{ExecutionDag, ExecutionStats, Node},
        values::Value,
    },
};
