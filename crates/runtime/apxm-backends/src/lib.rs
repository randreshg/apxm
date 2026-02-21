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
    // Observability
    AggregatedMetrics,
    // Backend traits and types
    AnthropicModel,
    // Retry logic
    ErrorClass,
    GenerationConfig,
    GoogleModel,
    // Registry and health
    HealthMonitor,
    HealthStatus,
    // Schema validation
    JsonSchema,
    LLMBackend,
    LLMRegistry,
    LLMRequest,
    LLMResponse,
    MetricsTracker,
    OllamaModel,
    OpenAIModel,
    OutputParser,
    // Provider management
    Provider,
    ProviderId,
    RequestBuilder,
    RequestMetrics,
    RequestTracer,
    RetryConfig,
    RetryStrategy,
    TokenUsage,
    // Tool types
    ToolChoice,
    ToolDefinition,
};

// ═══════════════════════════════════════════════════════════════════════════
// Storage Re-exports
// ═══════════════════════════════════════════════════════════════════════════

pub use storage::{
    // Backend trait and types
    BackendStats,
    // Embeddings
    Embedder,
    // Implementations
    InMemoryBackend,
    RedbBackend,
    SearchResult,
    SqliteBackend,
    StorageBackend,
    StorageResult,
    cosine_similarity,
};

#[cfg(feature = "embeddings")]
pub use storage::LocalEmbedder;

// ═══════════════════════════════════════════════════════════════════════════
// Prompts Re-exports
// ═══════════════════════════════════════════════════════════════════════════

pub use prompts::{list_prompts, render_inline, render_prompt};

// Re-export core types used across modules
pub use apxm_core::{error::RuntimeError, types::values::Value};
