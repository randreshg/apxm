//! Health monitoring for backends.
//!
//! Tracks backend health status based on recent request success/failure rates
//! and response latencies.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Health status of a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HealthStatus {
    /// Backend is healthy (success rate >= 90%)
    Healthy,
    /// Backend is degraded (success rate 50-90%)
    Degraded,
    /// Backend is unhealthy (success rate < 50%)
    Unhealthy,
    /// Health status unknown (insufficient data)
    #[default]
    Unknown,
}

/// Health statistics for a backend.
#[derive(Debug, Clone)]
struct HealthStats {
    /// Total requests
    total_requests: usize,
    /// Successful requests
    successful_requests: usize,
    /// Failed requests
    failed_requests: usize,
    /// Recent latencies (last 10 requests)
    recent_latencies: Vec<Duration>,
    /// Last update timestamp
    last_updated: Instant,
    /// Explicit status override
    status_override: Option<HealthStatus>,
}

impl HealthStats {
    fn new() -> Self {
        HealthStats {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            recent_latencies: Vec::new(),
            last_updated: Instant::now(),
            status_override: None,
        }
    }

    fn record_success(&mut self, latency: Duration) {
        self.total_requests += 1;
        self.successful_requests += 1;
        self.add_latency(latency);
        self.last_updated = Instant::now();
    }

    fn record_failure(&mut self, latency: Duration) {
        self.total_requests += 1;
        self.failed_requests += 1;
        self.add_latency(latency);
        self.last_updated = Instant::now();
    }

    fn add_latency(&mut self, latency: Duration) {
        self.recent_latencies.push(latency);
        if self.recent_latencies.len() > 10 {
            self.recent_latencies.remove(0);
        }
    }

    fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        (self.successful_requests as f64) / (self.total_requests as f64)
    }

    fn average_latency(&self) -> Option<Duration> {
        if self.recent_latencies.is_empty() {
            return None;
        }

        let total: Duration = self.recent_latencies.iter().sum();
        Some(total / self.recent_latencies.len() as u32)
    }

    fn compute_status(&self) -> HealthStatus {
        // Use override if set
        if let Some(status) = self.status_override {
            return status;
        }

        // Need at least 3 requests for meaningful health check
        if self.total_requests < 3 {
            return HealthStatus::Unknown;
        }

        let success_rate = self.success_rate();

        if success_rate >= 0.9 {
            HealthStatus::Healthy
        } else if success_rate >= 0.5 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        }
    }
}

/// Health monitor tracks backend health.
pub struct HealthMonitor {
    /// Health statistics per backend
    stats: Arc<DashMap<String, parking_lot::Mutex<HealthStats>>>,
}

impl HealthMonitor {
    /// Create a new health monitor.
    pub fn new() -> Self {
        HealthMonitor {
            stats: Arc::new(DashMap::new()),
        }
    }

    /// Register a backend for health tracking.
    pub fn register_backend(&self, name: &str) {
        self.stats.insert(
            name.to_string(),
            parking_lot::Mutex::new(HealthStats::new()),
        );
    }

    /// Unregister a backend from health tracking.
    pub fn unregister_backend(&self, name: &str) {
        self.stats.remove(name);
    }

    /// Record a successful request.
    pub fn record_success(&self, name: &str, latency: Duration) {
        if let Some(entry) = self.stats.get(name) {
            entry.value().lock().record_success(latency);
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self, name: &str, latency: Duration) {
        if let Some(entry) = self.stats.get(name) {
            entry.value().lock().record_failure(latency);
        }
    }

    /// Get the health status of a backend.
    pub fn status(&self, name: &str) -> HealthStatus {
        self.stats
            .get(name)
            .map(|entry| entry.value().lock().compute_status())
            .unwrap_or(HealthStatus::Unknown)
    }

    /// Explicitly set the health status of a backend (override).
    pub fn set_status(&self, name: &str, status: HealthStatus) {
        if let Some(entry) = self.stats.get(name) {
            entry.value().lock().status_override = Some(status);
        }
    }

    /// Get success rate for a backend.
    pub fn success_rate(&self, name: &str) -> Option<f64> {
        self.stats
            .get(name)
            .map(|entry| entry.value().lock().success_rate())
    }

    /// Get average latency for a backend.
    pub fn average_latency(&self, name: &str) -> Option<Duration> {
        self.stats
            .get(name)
            .and_then(|entry| entry.value().lock().average_latency())
    }

    /// Get total request count for a backend.
    pub fn total_requests(&self, name: &str) -> Option<usize> {
        self.stats
            .get(name)
            .map(|entry| entry.value().lock().total_requests)
    }

    /// Reset statistics for a backend.
    pub fn reset(&self, name: &str) {
        if let Some(entry) = self.stats.get(name) {
            *entry.value().lock() = HealthStats::new();
        }
    }

    /// Get all backend names being monitored.
    pub fn monitored_backends(&self) -> Vec<String> {
        self.stats.iter().map(|entry| entry.key().clone()).collect()
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_stats_success_rate() {
        let mut stats = HealthStats::new();

        // Initially unknown
        assert_eq!(stats.compute_status(), HealthStatus::Unknown);

        // Record successes
        for _ in 0..9 {
            stats.record_success(Duration::from_millis(100));
        }
        stats.record_failure(Duration::from_millis(100));

        // 90% success rate = Healthy
        assert_eq!(stats.success_rate(), 0.9);
        assert_eq!(stats.compute_status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_health_monitor() -> Result<(), Box<dyn std::error::Error>> {
        let monitor = HealthMonitor::new();

        monitor.register_backend("test");
        assert_eq!(monitor.status("test"), HealthStatus::Unknown);

        // Record enough successes to be healthy
        for _ in 0..10 {
            monitor.record_success("test", Duration::from_millis(50));
        }

        assert_eq!(monitor.status("test"), HealthStatus::Healthy);

        // Handle Option return without unwrap()
        let rate = monitor
            .success_rate("test")
            .ok_or_else(|| "missing success rate for 'test'".to_string())
            .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>)?;
        assert!(rate > 0.99, "unexpected success_rate: {}", rate);

        Ok(())
    }

    #[test]
    fn test_health_degraded() {
        let monitor = HealthMonitor::new();
        monitor.register_backend("test");

        // 60% success rate = Degraded
        for _ in 0..6 {
            monitor.record_success("test", Duration::from_millis(50));
        }
        for _ in 0..4 {
            monitor.record_failure("test", Duration::from_millis(50));
        }

        assert_eq!(monitor.status("test"), HealthStatus::Degraded);
    }

    #[test]
    fn test_health_unhealthy() {
        let monitor = HealthMonitor::new();
        monitor.register_backend("test");

        // 30% success rate = Unhealthy
        for _ in 0..3 {
            monitor.record_success("test", Duration::from_millis(50));
        }
        for _ in 0..7 {
            monitor.record_failure("test", Duration::from_millis(50));
        }

        assert_eq!(monitor.status("test"), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_status_override() {
        let monitor = HealthMonitor::new();
        monitor.register_backend("test");

        // Initially unknown
        assert_eq!(monitor.status("test"), HealthStatus::Unknown);

        // Set override
        monitor.set_status("test", HealthStatus::Healthy);
        assert_eq!(monitor.status("test"), HealthStatus::Healthy);

        // Override persists even with failures
        monitor.record_failure("test", Duration::from_millis(50));
        monitor.record_failure("test", Duration::from_millis(50));
        monitor.record_failure("test", Duration::from_millis(50));
        assert_eq!(monitor.status("test"), HealthStatus::Healthy);
    }
}
