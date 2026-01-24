//! APxM Models - Unified LLM Provider Integration Library
//!
//! Provides a declarative, minimalistic interface for integrating multiple LLM providers
//! (OpenAI, Anthropic, Google, local Ollama) with intelligent routing, cost tracking,
//! retry logic, and schema validation.
//!
//! # Design Principles
//!
//! - **Minimal scope**: Only manages model connections and requests, not storage or sessions
//! - **Declarative**: Configuration-driven behavior
//! - **Modular**: Each provider is independently extensible
//! - **DRY**: No duplication across providers
//! - **Type-safe**: Leverages Rust's type system
//! ```

pub mod backends;
#[cfg(feature = "metrics")]
pub mod observability;
#[cfg(not(feature = "metrics"))]
pub mod observability {
    use apxm_core::types::TokenUsage;
    use serde::{Deserialize, Serialize};
    use std::time::Duration;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RequestMetrics {
        pub backend: String,
        pub model: String,
        pub latency: Duration,
        pub usage: TokenUsage,
        pub success: bool,
        pub retry_count: usize,
        pub timestamp: std::time::SystemTime,
    }

    impl Default for RequestMetrics {
        fn default() -> Self {
            Self {
                backend: String::new(),
                model: String::new(),
                latency: Duration::default(),
                usage: TokenUsage::new(0, 0),
                success: false,
                retry_count: 0,
                timestamp: std::time::SystemTime::UNIX_EPOCH,
            }
        }
    }

    impl RequestMetrics {
        pub fn new(
            backend: String,
            model: String,
            latency: Duration,
            usage: TokenUsage,
            success: bool,
            retry_count: usize,
        ) -> Self {
            Self {
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

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct AggregatedMetrics {
        pub total_requests: usize,
        pub successful_requests: usize,
        pub failed_requests: usize,
        pub total_input_tokens: usize,
        pub total_output_tokens: usize,
        pub average_latency: Duration,
        pub p50_latency: Duration,
        pub p99_latency: Duration,
        pub total_retries: usize,
    }

    #[derive(Clone, Default)]
    pub struct MetricsTracker;

    impl MetricsTracker {
        pub fn new() -> Self {
            Self
        }

        pub fn record(&self, _metrics: RequestMetrics) {}

        pub fn aggregate(&self) -> AggregatedMetrics {
            AggregatedMetrics::default()
        }

        pub fn aggregate_backend(&self, _backend: &str) -> Option<AggregatedMetrics> {
            None
        }

        pub fn aggregate_model(&self, _model: &str) -> Option<AggregatedMetrics> {
            None
        }

        pub fn success_rate(&self) -> f64 {
            0.0
        }

        pub fn reset(&self) {}
    }

    pub struct RequestTracer;

    impl RequestTracer {
        pub fn start(_backend: String, _model: String) -> Self {
            Self
        }

        pub fn record_retry(&mut self) {}

        pub fn finish(self, usage: TokenUsage, success: bool) -> RequestMetrics {
            RequestMetrics::new(
                String::new(),
                String::new(),
                Duration::default(),
                usage,
                success,
                0,
            )
        }
    }
}
pub mod provider;
pub mod registry;
pub mod retry;
pub mod schema;

// Re-export key public API types
pub use backends::{
    AnthropicModel, GenerationConfig, GoogleModel, LLMBackend, LLMRequest, LLMResponse,
    OllamaModel, OpenAIModel, RequestBuilder, TokenUsage, ToolChoice, ToolDefinition,
};
pub use observability::{AggregatedMetrics, MetricsTracker, RequestMetrics, RequestTracer};
pub use provider::{Provider, ProviderId};
pub use registry::{HealthMonitor, HealthStatus, LLMRegistry};
pub use retry::{ErrorClass, RetryConfig, RetryStrategy};
pub use schema::{JsonSchema, OutputParser};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
