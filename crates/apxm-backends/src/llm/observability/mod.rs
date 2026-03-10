//! Observability and metrics tracking for LLM requests.
//!
//! Provides request tracing, metrics collection, and performance monitoring.

use apxm_core::types::TokenUsage;
use dashmap::DashMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Metrics for a single request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// Backend used for the request
    pub backend: String,
    /// Model used
    pub model: String,
    /// Request latency
    pub latency: Duration,
    /// Token usage
    pub usage: TokenUsage,
    /// Whether the request succeeded
    pub success: bool,
    /// Number of retry attempts
    pub retry_count: usize,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

impl RequestMetrics {
    /// Create new request metrics.
    pub fn new(
        backend: String,
        model: String,
        latency: Duration,
        usage: TokenUsage,
        success: bool,
        retry_count: usize,
    ) -> Self {
        RequestMetrics {
            backend,
            model,
            latency,
            usage,
            success,
            retry_count,
            timestamp: std::time::SystemTime::now(),
        }
    }
}

/// Aggregated metrics for analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Total requests
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Total input tokens
    pub total_input_tokens: usize,
    /// Total output tokens
    pub total_output_tokens: usize,
    /// Average latency
    pub average_latency: Duration,
    /// P50 latency
    pub p50_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
    /// Total retries
    pub total_retries: usize,
}

/// Metrics tracker for observability.
#[derive(Clone)]
pub struct MetricsTracker {
    inner: Arc<Mutex<MetricsTrackerInner>>,
}

struct MetricsTrackerInner {
    /// All request metrics
    requests: Vec<RequestMetrics>,
    /// Per-backend metrics
    backend_metrics: DashMap<String, Vec<RequestMetrics>>,
    /// Per-model metrics
    model_metrics: DashMap<String, Vec<RequestMetrics>>,
}

impl MetricsTracker {
    /// Create a new metrics tracker.
    pub fn new() -> Self {
        MetricsTracker {
            inner: Arc::new(Mutex::new(MetricsTrackerInner {
                requests: Vec::new(),
                backend_metrics: DashMap::new(),
                model_metrics: DashMap::new(),
            })),
        }
    }

    /// Record a request.
    pub fn record(&self, metrics: RequestMetrics) {
        let mut inner = self.inner.lock();

        // Add to per-backend metrics
        inner
            .backend_metrics
            .entry(metrics.backend.clone())
            .or_default()
            .push(metrics.clone());

        // Add to per-model metrics
        inner
            .model_metrics
            .entry(metrics.model.clone())
            .or_default()
            .push(metrics.clone());

        // Add to global metrics
        inner.requests.push(metrics);
    }

    /// Get aggregated metrics for all requests.
    pub fn aggregate(&self) -> AggregatedMetrics {
        let inner = self.inner.lock();
        Self::compute_aggregated(&inner.requests)
    }

    /// Get aggregated metrics for a specific backend.
    pub fn aggregate_backend(&self, backend: &str) -> Option<AggregatedMetrics> {
        let inner = self.inner.lock();
        inner
            .backend_metrics
            .get(backend)
            .map(|metrics| Self::compute_aggregated(metrics.value()))
    }

    /// Get aggregated metrics for a specific model.
    pub fn aggregate_model(&self, model: &str) -> Option<AggregatedMetrics> {
        let inner = self.inner.lock();
        inner
            .model_metrics
            .get(model)
            .map(|metrics| Self::compute_aggregated(metrics.value()))
    }

    /// Compute aggregated metrics from a list of requests.
    fn compute_aggregated(requests: &[RequestMetrics]) -> AggregatedMetrics {
        if requests.is_empty() {
            return AggregatedMetrics::default();
        }

        let total_requests = requests.len();
        let successful_requests = requests.iter().filter(|r| r.success).count();
        let failed_requests = total_requests - successful_requests;

        let total_input_tokens: usize = requests.iter().map(|r| r.usage.input_tokens).sum();
        let total_output_tokens: usize = requests.iter().map(|r| r.usage.output_tokens).sum();

        let total_latency: Duration = requests.iter().map(|r| r.latency).sum();
        let average_latency = total_latency / total_requests as u32;

        let total_retries: usize = requests.iter().map(|r| r.retry_count).sum();

        // Calculate percentiles
        let mut latencies: Vec<Duration> = requests.iter().map(|r| r.latency).collect();
        latencies.sort();

        let p50_index = (total_requests as f64 * 0.5) as usize;
        let p99_index = (total_requests as f64 * 0.99) as usize;

        let p50_latency = latencies.get(p50_index).copied().unwrap_or_default();
        let p99_latency = latencies.get(p99_index).copied().unwrap_or_default();

        AggregatedMetrics {
            total_requests,
            successful_requests,
            failed_requests,
            total_input_tokens,
            total_output_tokens,
            average_latency,
            p50_latency,
            p99_latency,
            total_retries,
        }
    }

    /// Get success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let metrics = self.aggregate();
        if metrics.total_requests == 0 {
            return 0.0;
        }
        (metrics.successful_requests as f64 / metrics.total_requests as f64) * 100.0
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        let mut inner = self.inner.lock();
        inner.requests.clear();
        inner.backend_metrics.clear();
        inner.model_metrics.clear();
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Request tracer for debugging and logging.
pub struct RequestTracer {
    start: Instant,
    backend: String,
    model: String,
    retry_count: usize,
}

impl RequestTracer {
    /// Start tracing a new request.
    pub fn start(backend: String, model: String) -> Self {
        RequestTracer {
            start: Instant::now(),
            backend,
            model,
            retry_count: 0,
        }
    }

    /// Record a retry attempt.
    pub fn record_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Finish tracing and return metrics.
    pub fn finish(self, usage: TokenUsage, success: bool) -> RequestMetrics {
        let latency = self.start.elapsed();
        RequestMetrics::new(
            self.backend,
            self.model,
            latency,
            usage,
            success,
            self.retry_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_tracker() {
        let tracker = MetricsTracker::new();

        let metrics1 = RequestMetrics::new(
            "openai".to_string(),
            "gpt-4".to_string(),
            Duration::from_millis(500),
            TokenUsage::new(100, 50),
            true,
            0,
        );

        let metrics2 = RequestMetrics::new(
            "openai".to_string(),
            "gpt-4".to_string(),
            Duration::from_millis(300),
            TokenUsage::new(200, 75),
            true,
            1,
        );

        tracker.record(metrics1);
        tracker.record(metrics2);

        let aggregated = tracker.aggregate();
        assert_eq!(aggregated.total_requests, 2);
        assert_eq!(aggregated.successful_requests, 2);
        assert_eq!(aggregated.total_input_tokens, 300);
        assert_eq!(aggregated.total_output_tokens, 125);
        assert_eq!(aggregated.total_retries, 1);
    }

    #[test]
    fn test_backend_specific_metrics() -> Result<(), Box<dyn std::error::Error>> {
        let tracker = MetricsTracker::new();

        let openai_metrics = RequestMetrics::new(
            "openai".to_string(),
            "gpt-4".to_string(),
            Duration::from_millis(500),
            TokenUsage::new(100, 50),
            true,
            0,
        );

        let anthropic_metrics = RequestMetrics::new(
            "anthropic".to_string(),
            "claude-3".to_string(),
            Duration::from_millis(600),
            TokenUsage::new(150, 60),
            true,
            0,
        );

        tracker.record(openai_metrics);
        tracker.record(anthropic_metrics);

        let openai_agg = match tracker.aggregate_backend("openai") {
            Some(agg) => agg,
            None => return Err("expected aggregated metrics for 'openai'".into()),
        };
        assert_eq!(openai_agg.total_requests, 1);
        assert_eq!(openai_agg.total_input_tokens, 100);

        let anthropic_agg = match tracker.aggregate_backend("anthropic") {
            Some(agg) => agg,
            None => return Err("expected aggregated metrics for 'anthropic'".into()),
        };
        assert_eq!(anthropic_agg.total_requests, 1);
        assert_eq!(anthropic_agg.total_input_tokens, 150);

        Ok(())
    }

    #[test]
    fn test_success_rate() {
        let tracker = MetricsTracker::new();

        tracker.record(RequestMetrics::new(
            "test".to_string(),
            "model".to_string(),
            Duration::from_millis(100),
            TokenUsage::new(10, 10),
            true,
            0,
        ));

        tracker.record(RequestMetrics::new(
            "test".to_string(),
            "model".to_string(),
            Duration::from_millis(100),
            TokenUsage::new(10, 10),
            false,
            0,
        ));

        assert!((tracker.success_rate() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_request_tracer() {
        let mut tracer = RequestTracer::start("test".to_string(), "model".to_string());
        tracer.record_retry();
        tracer.record_retry();

        let metrics = tracer.finish(TokenUsage::new(100, 50), true);

        assert_eq!(metrics.backend, "test");
        assert_eq!(metrics.model, "model");
        assert_eq!(metrics.retry_count, 2);
        assert!(metrics.success);
    }
}
