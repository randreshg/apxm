//! Observability - Zero-overhead metrics and tracing for runtime
//!
//! This module provides metrics collection that can be completely disabled
//! at compile time for zero runtime overhead in production.
//!
//! # Feature Flags
//!
//! - `metrics`: Enables full metrics collection with scheduler overhead breakdown
//!
//! When the `metrics` feature is disabled, all metrics calls compile to nothing.

mod metrics;

pub use metrics::{MetricsCollector, OverheadBreakdown, SchedulerMetrics};

// Re-export the timing macros at crate root
pub use crate::{timed, timed_async};
