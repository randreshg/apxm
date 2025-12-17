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
pub mod observability;
pub mod provider;
pub mod registry;
pub mod retry;
pub mod schema;

// Re-export key public API types
pub use backends::{
    AnthropicModel, GenerationConfig, GoogleModel, LLMBackend, LLMRequest, LLMResponse,
    OllamaModel, OpenAIModel, RequestBuilder, TokenUsage,
};
pub use observability::{AggregatedMetrics, MetricsTracker, RequestMetrics, RequestTracer};
pub use provider::{Provider, ProviderId};
pub use registry::{HealthMonitor, HealthStatus, LLMRegistry};
pub use retry::{ErrorClass, RetryConfig, RetryStrategy};
pub use schema::{JsonSchema, OutputParser};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
