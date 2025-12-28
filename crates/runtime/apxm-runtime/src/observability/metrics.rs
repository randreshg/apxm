//! Zero-overhead metrics collection for runtime observability.
//!
//! This module provides metrics collection that compiles to nothing when the
//! `metrics` feature is disabled, ensuring ZERO runtime overhead in production.
//!
//! # Usage
//!
//! ```bash
//! # Production build - no metrics overhead
//! cargo build --release -p apxm-runtime
//!
//! # Benchmarking build - with metrics
//! cargo build --release -p apxm-runtime --features metrics
//! ```

// ============================================================================
// METRICS ENABLED: Full implementation with overhead breakdown
// ============================================================================
#[cfg(feature = "metrics")]
mod enabled {
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::Duration;

    /// Metrics collector with scheduler overhead breakdown.
    ///
    /// All timing is in nanoseconds for precision. Atomic operations ensure
    /// thread-safe updates without locks on the hot path.
    #[derive(Debug)]
    pub struct MetricsCollector {
        // === Scheduler overhead breakdown (nanoseconds) ===
        /// Time spent updating the ready set
        pub ready_set_update_ns: AtomicU64,
        /// Time spent in work stealing
        pub work_stealing_ns: AtomicU64,
        /// Time spent collecting inputs
        pub input_collection_ns: AtomicU64,
        /// Time spent dispatching operations
        pub operation_dispatch_ns: AtomicU64,
        /// Time spent routing output tokens
        pub token_routing_ns: AtomicU64,

        // === Parallelism tracking ===
        /// Maximum concurrent operations observed
        pub max_concurrent_ops: AtomicUsize,
        /// Current operations in flight
        pub operations_in_flight: AtomicUsize,
        /// Samples of parallelism factor for averaging
        pub parallelism_samples: Mutex<Vec<usize>>,

        // === Operation counters ===
        /// Total operations executed
        pub operations_executed: AtomicUsize,
        /// Total operations failed
        pub operations_failed: AtomicUsize,
        /// Total tokens published
        pub tokens_published: AtomicUsize,
        /// Total retries attempted
        pub retries_attempted: AtomicUsize,
        /// Total execution time (microseconds)
        pub total_execution_time_us: AtomicU64,

        // === Overhead measurement counters ===
        /// Count of ready set updates (for averaging)
        pub ready_set_update_count: AtomicU64,
        /// Count of work stealing attempts
        pub work_stealing_count: AtomicU64,
        /// Count of input collections
        pub input_collection_count: AtomicU64,
        /// Count of operation dispatches
        pub operation_dispatch_count: AtomicU64,
        /// Count of token routings
        pub token_routing_count: AtomicU64,
    }

    impl Default for MetricsCollector {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MetricsCollector {
        pub fn new() -> Self {
            Self {
                ready_set_update_ns: AtomicU64::new(0),
                work_stealing_ns: AtomicU64::new(0),
                input_collection_ns: AtomicU64::new(0),
                operation_dispatch_ns: AtomicU64::new(0),
                token_routing_ns: AtomicU64::new(0),
                max_concurrent_ops: AtomicUsize::new(0),
                operations_in_flight: AtomicUsize::new(0),
                parallelism_samples: Mutex::new(Vec::with_capacity(1024)),
                operations_executed: AtomicUsize::new(0),
                operations_failed: AtomicUsize::new(0),
                tokens_published: AtomicUsize::new(0),
                retries_attempted: AtomicUsize::new(0),
                total_execution_time_us: AtomicU64::new(0),
                ready_set_update_count: AtomicU64::new(0),
                work_stealing_count: AtomicU64::new(0),
                input_collection_count: AtomicU64::new(0),
                operation_dispatch_count: AtomicU64::new(0),
                token_routing_count: AtomicU64::new(0),
            }
        }

        // === Scheduler overhead recording ===

        #[inline]
        pub fn record_ready_set_update(&self, duration: Duration) {
            self.ready_set_update_ns
                .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
            self.ready_set_update_count.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        pub fn record_work_stealing(&self, duration: Duration) {
            self.work_stealing_ns
                .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
            self.work_stealing_count.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        pub fn record_input_collection(&self, duration: Duration) {
            self.input_collection_ns
                .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
            self.input_collection_count.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        pub fn record_operation_dispatch(&self, duration: Duration) {
            self.operation_dispatch_ns
                .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
            self.operation_dispatch_count.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        pub fn record_token_routing(&self, duration: Duration) {
            self.token_routing_ns
                .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
            self.token_routing_count.fetch_add(1, Ordering::Relaxed);
        }

        // === Parallelism tracking ===

        #[inline]
        pub fn record_schedule(&self) {
            let current = self.operations_in_flight.fetch_add(1, Ordering::Relaxed) + 1;

            // Update max if needed
            let mut max = self.max_concurrent_ops.load(Ordering::Relaxed);
            while current > max {
                match self.max_concurrent_ops.compare_exchange_weak(
                    max,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => max = actual,
                }
            }

            // Sample parallelism (don't lock on every call - sample every 16th)
            if current > 1 && self.operations_executed.load(Ordering::Relaxed) % 16 == 0 {
                if let Some(mut samples) = self.parallelism_samples.try_lock() {
                    samples.push(current);
                }
            }
        }

        #[inline]
        pub fn record_completion(&self) {
            self.operations_executed.fetch_add(1, Ordering::Relaxed);
            self.operations_in_flight.fetch_sub(1, Ordering::Relaxed);
        }

        #[inline]
        pub fn record_failure(&self) {
            self.operations_failed.fetch_add(1, Ordering::Relaxed);
            self.operations_in_flight.fetch_sub(1, Ordering::Relaxed);
        }

        // === Legacy methods for compatibility ===

        #[inline]
        pub fn increment_executed(&self) {
            self.operations_executed.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        pub fn increment_failed(&self) {
            self.operations_failed.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        pub fn increment_tokens(&self) {
            self.tokens_published.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        pub fn increment_retries(&self) {
            self.retries_attempted.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        pub fn record_execution_time(&self, duration: Duration) {
            self.total_execution_time_us
                .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
        }

        #[inline]
        pub fn increment_in_flight(&self) {
            self.operations_in_flight.fetch_add(1, Ordering::Relaxed);
        }

        #[inline]
        pub fn decrement_in_flight(&self) {
            self.operations_in_flight.fetch_sub(1, Ordering::Relaxed);
        }

        // === Getters ===

        pub fn get_executed(&self) -> usize {
            self.operations_executed.load(Ordering::Relaxed)
        }

        pub fn get_failed(&self) -> usize {
            self.operations_failed.load(Ordering::Relaxed)
        }

        pub fn get_in_flight(&self) -> usize {
            self.operations_in_flight.load(Ordering::Relaxed)
        }

        // === Reporting methods ===

        /// Get per-operation overhead breakdown in microseconds.
        pub fn overhead_breakdown_us(&self) -> OverheadBreakdown {
            let ops = self.operations_executed.load(Ordering::Relaxed).max(1) as f64;

            OverheadBreakdown {
                ready_set_update_us: self.ready_set_update_ns.load(Ordering::Relaxed) as f64
                    / 1000.0
                    / ops,
                work_stealing_us: self.work_stealing_ns.load(Ordering::Relaxed) as f64
                    / 1000.0
                    / ops,
                input_collection_us: self.input_collection_ns.load(Ordering::Relaxed) as f64
                    / 1000.0
                    / ops,
                operation_dispatch_us: self.operation_dispatch_ns.load(Ordering::Relaxed) as f64
                    / 1000.0
                    / ops,
                token_routing_us: self.token_routing_ns.load(Ordering::Relaxed) as f64
                    / 1000.0
                    / ops,
            }
        }

        /// Get total scheduler overhead per operation in microseconds.
        pub fn total_overhead_per_op_us(&self) -> f64 {
            let breakdown = self.overhead_breakdown_us();
            breakdown.ready_set_update_us
                + breakdown.work_stealing_us
                + breakdown.input_collection_us
                + breakdown.operation_dispatch_us
                + breakdown.token_routing_us
        }

        /// Get average parallelism factor.
        pub fn average_parallelism(&self) -> f64 {
            let samples = self.parallelism_samples.lock();
            if samples.is_empty() {
                1.0
            } else {
                samples.iter().sum::<usize>() as f64 / samples.len() as f64
            }
        }

        /// Get maximum observed parallelism.
        pub fn max_parallelism(&self) -> usize {
            self.max_concurrent_ops.load(Ordering::Relaxed)
        }

        /// Print a summary report to stdout.
        pub fn print_report(&self) {
            let breakdown = self.overhead_breakdown_us();
            let total = self.total_overhead_per_op_us();

            println!("╭─ Scheduler Overhead Breakdown ─────────────────────────────╮");
            println!(
                "│ Ready set update:    {:>8.2} μs/op ({:>5.1}%)              │",
                breakdown.ready_set_update_us,
                breakdown.ready_set_update_us / total * 100.0
            );
            println!(
                "│ Work stealing:       {:>8.2} μs/op ({:>5.1}%)              │",
                breakdown.work_stealing_us,
                breakdown.work_stealing_us / total * 100.0
            );
            println!(
                "│ Input collection:    {:>8.2} μs/op ({:>5.1}%)              │",
                breakdown.input_collection_us,
                breakdown.input_collection_us / total * 100.0
            );
            println!(
                "│ Operation dispatch:  {:>8.2} μs/op ({:>5.1}%)              │",
                breakdown.operation_dispatch_us,
                breakdown.operation_dispatch_us / total * 100.0
            );
            println!(
                "│ Token routing:       {:>8.2} μs/op ({:>5.1}%)              │",
                breakdown.token_routing_us,
                breakdown.token_routing_us / total * 100.0
            );
            println!("├─────────────────────────────────────────────────────────────┤");
            println!(
                "│ TOTAL OVERHEAD:      {:>8.2} μs/op                        │",
                total
            );
            println!("│                                                             │");
            println!(
                "│ Operations executed: {:>8}                               │",
                self.get_executed()
            );
            println!(
                "│ Max parallelism:     {:>8}                               │",
                self.max_parallelism()
            );
            println!(
                "│ Avg parallelism:     {:>8.2}                               │",
                self.average_parallelism()
            );
            println!("╰─────────────────────────────────────────────────────────────╯");
        }

        /// Export metrics as JSON value.
        pub fn to_json(&self) -> serde_json::Value {
            let breakdown = self.overhead_breakdown_us();
            serde_json::json!({
                "overhead_breakdown": {
                    "ready_set_update_us": breakdown.ready_set_update_us,
                    "work_stealing_us": breakdown.work_stealing_us,
                    "input_collection_us": breakdown.input_collection_us,
                    "operation_dispatch_us": breakdown.operation_dispatch_us,
                    "token_routing_us": breakdown.token_routing_us
                },
                "total_overhead_per_op_us": self.total_overhead_per_op_us(),
                "operations_executed": self.get_executed(),
                "operations_failed": self.get_failed(),
                "max_parallelism": self.max_parallelism(),
                "average_parallelism": self.average_parallelism()
            })
        }
    }

    /// Overhead breakdown per operation.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct OverheadBreakdown {
        pub ready_set_update_us: f64,
        pub work_stealing_us: f64,
        pub input_collection_us: f64,
        pub operation_dispatch_us: f64,
        pub token_routing_us: f64,
    }

    impl OverheadBreakdown {
        /// Calculate total overhead in microseconds.
        pub fn total_us(&self) -> f64 {
            self.ready_set_update_us
                + self.work_stealing_us
                + self.input_collection_us
                + self.operation_dispatch_us
                + self.token_routing_us
        }
    }
}

// ============================================================================
// METRICS DISABLED: Empty stub that compiles to nothing
// ============================================================================
#[cfg(not(feature = "metrics"))]
mod disabled {
    use std::time::Duration;

    /// No-op metrics collector.
    ///
    /// All methods are `#[inline(always)]` and empty, so they are completely
    /// optimized away by the compiler. Zero overhead in production.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct MetricsCollector;

    impl MetricsCollector {
        #[inline(always)]
        pub fn new() -> Self {
            Self
        }

        // Scheduler overhead recording - all no-ops
        #[inline(always)]
        pub fn record_ready_set_update(&self, _: Duration) {}
        #[inline(always)]
        pub fn record_work_stealing(&self, _: Duration) {}
        #[inline(always)]
        pub fn record_input_collection(&self, _: Duration) {}
        #[inline(always)]
        pub fn record_operation_dispatch(&self, _: Duration) {}
        #[inline(always)]
        pub fn record_token_routing(&self, _: Duration) {}

        // Parallelism tracking - all no-ops
        #[inline(always)]
        pub fn record_schedule(&self) {}
        #[inline(always)]
        pub fn record_completion(&self) {}
        #[inline(always)]
        pub fn record_failure(&self) {}

        // Legacy methods - all no-ops
        #[inline(always)]
        pub fn increment_executed(&self) {}
        #[inline(always)]
        pub fn increment_failed(&self) {}
        #[inline(always)]
        pub fn increment_tokens(&self) {}
        #[inline(always)]
        pub fn increment_retries(&self) {}
        #[inline(always)]
        pub fn record_execution_time(&self, _: Duration) {}
        #[inline(always)]
        pub fn increment_in_flight(&self) {}
        #[inline(always)]
        pub fn decrement_in_flight(&self) {}

        // Getters return defaults
        #[inline(always)]
        pub fn get_executed(&self) -> usize {
            0
        }
        #[inline(always)]
        pub fn get_failed(&self) -> usize {
            0
        }
        #[inline(always)]
        pub fn get_in_flight(&self) -> usize {
            0
        }
    }

    /// Placeholder overhead breakdown.
    #[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
    pub struct OverheadBreakdown {
        pub ready_set_update_us: f64,
        pub work_stealing_us: f64,
        pub input_collection_us: f64,
        pub operation_dispatch_us: f64,
        pub token_routing_us: f64,
    }

    impl OverheadBreakdown {
        /// Calculate total overhead in microseconds.
        pub fn total_us(&self) -> f64 {
            self.ready_set_update_us
                + self.work_stealing_us
                + self.input_collection_us
                + self.operation_dispatch_us
                + self.token_routing_us
        }
    }
}

// ============================================================================
// Public exports
// ============================================================================
#[cfg(feature = "metrics")]
pub use enabled::{MetricsCollector, OverheadBreakdown};

#[cfg(not(feature = "metrics"))]
pub use disabled::{MetricsCollector, OverheadBreakdown};

// ============================================================================
// Zero-overhead timing macro
// ============================================================================

/// Times a block only when the `metrics` feature is enabled.
///
/// When metrics are disabled, this compiles to just the block with zero overhead.
///
/// # Example
///
/// ```ignore
/// use apxm_runtime::timed;
///
/// let result = timed!(metrics, record_work_stealing, {
///     state.work_stealing.steal_next()
/// });
/// ```
#[macro_export]
macro_rules! timed {
    ($metrics:expr, $method:ident, $block:expr) => {{
        #[cfg(feature = "metrics")]
        let __start = std::time::Instant::now();

        let __result = $block;

        #[cfg(feature = "metrics")]
        $metrics.$method(__start.elapsed());

        __result
    }};
}

/// Times an async block only when the `metrics` feature is enabled.
#[macro_export]
macro_rules! timed_async {
    ($metrics:expr, $method:ident, $block:expr) => {{
        #[cfg(feature = "metrics")]
        let __start = std::time::Instant::now();

        let __result = $block.await;

        #[cfg(feature = "metrics")]
        $metrics.$method(__start.elapsed());

        __result
    }};
}
