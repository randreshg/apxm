//! Scheduler configuration.
//!
//! This module defines configuration options for the dataflow scheduler,
//! including parallelism settings, retry behavior, and resource limits.

use serde::{Deserialize, Serialize};

/// Configuration for the dataflow scheduler.
///
/// Controls parallelism, work-stealing behavior, retry logic, and resource limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum number of concurrent worker threads.
    ///
    /// This controls the level of parallelism. Generally should be set to
    /// the number of CPU cores available.
    ///
    /// Default: Number of logical CPUs
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,

    /// Maximum number of operations that can be in-flight simultaneously.
    ///
    /// This provides backpressure to prevent overwhelming the system.
    /// Should be >= max_concurrency.
    ///
    /// Default: max_concurrency * 2
    #[serde(default = "default_max_inflight")]
    pub max_inflight: usize,

    /// Maximum number of retry attempts for failed operations.
    ///
    /// 0 means no retries, operations fail immediately.
    ///
    /// Default: 3
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Initial backoff delay in milliseconds for retries.
    ///
    /// The actual delay grows exponentially: initial_delay * 2^attempt
    ///
    /// Default: 500ms
    #[serde(default = "default_retry_backoff_ms")]
    pub retry_backoff_ms: u64,

    /// Maximum backoff delay in milliseconds.
    ///
    /// Caps the exponential growth of retry delays.
    ///
    /// Default: 60000ms (60 seconds)
    #[serde(default = "default_retry_backoff_max_ms")]
    pub retry_backoff_max_ms: u64,

    /// Watchdog interval in milliseconds.
    ///
    /// How often the watchdog checks for deadlocks.
    ///
    /// Default: 1000ms (1 second)
    #[serde(default = "default_watchdog_interval_ms")]
    pub watchdog_interval_ms: u64,

    /// Deadlock detection timeout in milliseconds.
    ///
    /// If no progress is made for this duration, execution is considered deadlocked.
    ///
    /// Default: 30000ms (30 seconds)
    #[serde(default = "default_deadlock_timeout_ms")]
    pub deadlock_timeout_ms: u64,

    /// Maximum total cost budget for the DAG.
    ///
    /// 0 means no limit. Cost is estimated from operation metadata.
    ///
    /// Default: 0 (unlimited)
    #[serde(default)]
    pub max_cost: usize,

    /// Work queue capacity per priority level.
    ///
    /// Limits the size of work queues to prevent unbounded memory growth.
    ///
    /// Default: 10000
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrency: default_max_concurrency(),
            max_inflight: default_max_inflight(),
            max_retries: default_max_retries(),
            retry_backoff_ms: default_retry_backoff_ms(),
            retry_backoff_max_ms: default_retry_backoff_max_ms(),
            watchdog_interval_ms: default_watchdog_interval_ms(),
            deadlock_timeout_ms: default_deadlock_timeout_ms(),
            max_cost: 0,
            queue_capacity: default_queue_capacity(),
        }
    }
}

impl SchedulerConfig {
    /// Create a new scheduler configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum concurrency level.
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Set the maximum in-flight operations.
    pub fn with_max_inflight(mut self, max_inflight: usize) -> Self {
        self.max_inflight = max_inflight;
        self
    }

    /// Set the maximum retry attempts.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set the retry backoff parameters.
    pub fn with_retry_backoff(mut self, initial_ms: u64, max_ms: u64) -> Self {
        self.retry_backoff_ms = initial_ms;
        self.retry_backoff_max_ms = max_ms;
        self
    }

    /// Set the deadlock detection timeout.
    pub fn with_deadlock_timeout(mut self, timeout_ms: u64) -> Self {
        self.deadlock_timeout_ms = timeout_ms;
        self
    }

    /// Set the maximum cost budget.
    pub fn with_max_cost(mut self, max_cost: usize) -> Self {
        self.max_cost = max_cost;
        self
    }

    /// Validate the configuration.
    ///
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_concurrency == 0 {
            return Err("max_concurrency must be > 0".to_string());
        }

        if self.max_inflight == 0 {
            return Err("max_inflight must be > 0".to_string());
        }

        if self.retry_backoff_ms == 0 {
            return Err("retry_backoff_ms must be > 0".to_string());
        }

        if self.retry_backoff_max_ms < self.retry_backoff_ms {
            return Err("retry_backoff_max_ms must be >= retry_backoff_ms".to_string());
        }

        if self.watchdog_interval_ms == 0 {
            return Err("watchdog_interval_ms must be > 0".to_string());
        }

        if self.deadlock_timeout_ms < self.watchdog_interval_ms {
            return Err("deadlock_timeout_ms must be >= watchdog_interval_ms".to_string());
        }

        Ok(())
    }
}

// Default functions for serde
fn default_max_concurrency() -> usize {
    num_cpus::get().max(1)
}

fn default_max_inflight() -> usize {
    default_max_concurrency() * 2
}

fn default_max_retries() -> u32 {
    3
}

fn default_retry_backoff_ms() -> u64 {
    500
}

fn default_retry_backoff_max_ms() -> u64 {
    60_000
}

fn default_watchdog_interval_ms() -> u64 {
    1_000
}

fn default_deadlock_timeout_ms() -> u64 {
    30_000
}

fn default_queue_capacity() -> usize {
    10_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SchedulerConfig::default();
        assert!(config.max_concurrency > 0);
        assert!(config.max_inflight >= config.max_concurrency);
        assert_eq!(config.max_retries, 3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let config = SchedulerConfig::new()
            .with_max_concurrency(8)
            .with_max_inflight(16)
            .with_max_retries(5)
            .with_retry_backoff(1000, 30_000)
            .with_deadlock_timeout(60_000);

        assert_eq!(config.max_concurrency, 8);
        assert_eq!(config.max_inflight, 16);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.retry_backoff_ms, 1000);
        assert_eq!(config.retry_backoff_max_ms, 30_000);
        assert_eq!(config.deadlock_timeout_ms, 60_000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_zero_concurrency() {
        let config = SchedulerConfig {
            max_concurrency: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_invalid_backoff() {
        let config = SchedulerConfig {
            retry_backoff_ms: 10_000,
            retry_backoff_max_ms: 1_000,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_invalid_deadlock_timeout() {
        let config = SchedulerConfig {
            watchdog_interval_ms: 5_000,
            deadlock_timeout_ms: 1_000,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
