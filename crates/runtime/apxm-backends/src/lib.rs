//! APXM Backends - LLM providers, storage backends, and prompt templates.
//!
//! This crate consolidates three backend systems:
//!
//! - **`llm`**: Unified LLM provider integration (OpenAI, Anthropic, Google, Ollama)
//! - **`storage`**: Pluggable storage backends (in-memory, SQLite, embedded KV)
//! - **`prompts`**: Compile-time embedded prompt templates with MiniJinja
//!
//! # Architecture
//!
//! ```text
//!                      ┌─────────────────┐
//!                      │  apxm-backends  │
//!                      └────────┬────────┘
//!                               │
//!        ┌──────────────────────┼──────────────────────┐
//!        ▼                      ▼                      ▼
//!   ┌─────────┐           ┌──────────┐           ┌──────────┐
//!   │   llm   │           │  storage │           │ prompts  │
//!   │ OpenAI  │           │  SQLite  │           │ MiniJinja│
//!   │Anthropic│           │  Memory  │           │ Templates│
//!   │ Google  │           │  Redb    │           │          │
//!   │ Ollama  │           │          │           │          │
//!   └─────────┘           └──────────┘           └──────────┘
//! ```

// Module declarations
pub mod llm;
pub mod prompts;
pub mod storage;

// ═══════════════════════════════════════════════════════════════════════════
// LLM Re-exports
// ═══════════════════════════════════════════════════════════════════════════

pub use llm::{
    // Backend traits and types
    AnthropicModel, GenerationConfig, GoogleModel, LLMBackend, LLMRequest, LLMResponse,
    OllamaModel, OpenAIModel, RequestBuilder, TokenUsage,
    // Observability
    AggregatedMetrics, MetricsTracker, RequestMetrics, RequestTracer,
    // Provider management
    Provider, ProviderId,
    // Registry and health
    HealthMonitor, HealthStatus, LLMRegistry,
    // Retry logic
    ErrorClass, RetryConfig, RetryStrategy,
    // Schema validation
    JsonSchema, OutputParser,
};

// ═══════════════════════════════════════════════════════════════════════════
// Storage Re-exports
// ═══════════════════════════════════════════════════════════════════════════

pub use storage::{
    // Backend trait and types
    BackendStats, SearchResult, StorageBackend, StorageResult,
    // Implementations
    InMemoryBackend, RedbBackend, SqliteBackend,
};

// ═══════════════════════════════════════════════════════════════════════════
// Prompts Re-exports
// ═══════════════════════════════════════════════════════════════════════════

pub use prompts::{list_prompts, render_inline, render_prompt};

// Re-export core types used across modules
pub use apxm_core::{error::RuntimeError, types::values::Value};
