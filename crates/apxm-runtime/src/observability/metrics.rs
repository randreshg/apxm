//! Metrics collection for runtime observability

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

/// Metrics collector for runtime execution
#[derive(Debug, Default)]
pub struct MetricsCollector {
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

    /// Operations currently in flight
    pub operations_in_flight: AtomicUsize,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_schedule(&self) {
        self.increment_in_flight();
    }

    pub fn record_completion(&self) {
        self.increment_executed();
        self.decrement_in_flight();
    }

    pub fn record_failure(&self) {
        self.increment_failed();
        self.decrement_in_flight();
    }

    pub fn increment_executed(&self) {
        self.operations_executed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_failed(&self) {
        self.operations_failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_tokens(&self) {
        self.tokens_published.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_retries(&self) {
        self.retries_attempted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_execution_time(&self, duration: Duration) {
        self.total_execution_time_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn increment_in_flight(&self) {
        self.operations_in_flight.fetch_add(1, Ordering::Relaxed);
    }

    pub fn decrement_in_flight(&self) {
        self.operations_in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn get_executed(&self) -> usize {
        self.operations_executed.load(Ordering::Relaxed)
    }

    pub fn get_failed(&self) -> usize {
        self.operations_failed.load(Ordering::Relaxed)
    }

    pub fn get_in_flight(&self) -> usize {
        self.operations_in_flight.load(Ordering::Relaxed)
    }
}
