//! Dataflow Scheduler.
//!
//! This module implements a work-stealing dataflow scheduler with:
//! - Token-based automatic parallelism
//! - 4-level priority queues (Critical, High, Normal, Low)
//! - O(1) readiness tracking
//! - Exponential backoff retry logic
//! - Deadlock detection
//! - Semaphore-based backpressure

pub mod concurrency_control;
pub mod config;
pub mod dataflow;
pub mod queue;
pub mod ready_set;
pub mod splicing;
pub mod state;
pub mod work_stealing;
pub mod worker;

// Internal state types (not part of public API)
pub(crate) mod internal_state;

// Public exports
pub use config::SchedulerConfig;
pub use dataflow::DataflowScheduler;
pub use queue::{Priority, PriorityQueue};
pub use splicing::SpliceConfig;

// Re-export execution types from apxm-core
pub use apxm_core::types::{ExecutionStats, NodeStatus, OpStatus};
