# apxm-backends

LLM providers, storage backends, and prompt templates.

## Overview

`apxm-backends` consolidates three backend systems:
- **LLM** - Unified interface to OpenAI, Anthropic, Google, Ollama
- **Storage** - Pluggable backends (SQLite, in-memory, Redb)
- **Prompts** - MiniJinja template rendering

## Responsibilities

- Provide LLM provider implementations and routing
- Expose storage backends used by the runtime
- Centralize prompt rendering utilities

## How It Fits

The runtime depends on `apxm-backends` for LLM calls and memory backends.
The driver configures providers and passes a registry into the runtime.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      apxm-backends                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │     LLM      │  │   Storage    │  │     Prompts      │  │
│  ├──────────────┤  ├──────────────┤  ├──────────────────┤  │
│  │ • OpenAI     │  │ • SQLite     │  │ • MiniJinja      │  │
│  │ • Anthropic  │  │ • InMemory   │  │ • Embedded       │  │
│  │ • Google     │  │ • Redb       │  │                  │  │
│  │ • Ollama     │  │              │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## LLM Module

### Providers

```rust
use apxm_backends::{Provider, ProviderId, LLMRegistry};

// Create a provider
let provider = Provider::new(
    ProviderId::OpenAI,
    "sk-...",
    Some(json!({"model": "gpt-4"})),
).await?;

// Register in registry
let registry = LLMRegistry::new();
registry.register("gpt4", provider)?;
registry.set_default("gpt4")?;
```

### Making Requests

```rust
use apxm_backends::{LLMRequest, RequestBuilder, GenerationConfig};

let request = RequestBuilder::new()
    .system("You are a helpful assistant")
    .user("Explain MLIR")
    .config(GenerationConfig {
        max_tokens: Some(1000),
        temperature: Some(0.7),
        ..Default::default()
    })
    .build();

let response = registry.generate(request).await?;
println!("{}", response.content);
```

### Observability

```rust
use apxm_backends::{MetricsTracker, RequestTracer, TokenUsage};

// Track request metrics
let metrics = MetricsTracker::new();
let tracer = RequestTracer::start("backend".to_string(), "model".to_string());

// ... after a request completes
let record = tracer.finish(TokenUsage::new(10, 5), true);
metrics.record(record);

// Get aggregated stats
let stats = metrics.aggregate();
println!(
    "Total tokens: {}",
    stats.total_input_tokens + stats.total_output_tokens
);
```

Metrics are compiled out unless the `metrics` feature is enabled.

## Storage Module

```rust
use apxm_backends::{StorageBackend, SqliteBackend, InMemoryBackend};

// SQLite backend
let sqlite = SqliteBackend::new("data.db").await?;
sqlite.store("key", &value).await?;
let result = sqlite.retrieve("key").await?;

// In-memory backend
let memory = InMemoryBackend::new();
memory.store("key", &value).await?;
```

## Prompts Module

```rust
use apxm_backends::{render_prompt, render_inline, list_prompts};

// Render embedded template
let output = render_prompt("reasoning", &context)?;

// Render inline template
let output = render_inline("Hello {{ name }}!", &json!({"name": "World"}))?;

// List available templates
for name in list_prompts() {
    println!("Template: {}", name);
}
```

## Key Types

| Type | Description |
|------|-------------|
| `LLMRegistry` | Provider registry with health monitoring |
| `Provider` | Single LLM provider instance |
| `LLMRequest` / `LLMResponse` | Request/response types |
| `StorageBackend` | Trait for storage implementations |
| `GenerationConfig` | LLM generation parameters |

## Dependencies

| Crate | Purpose |
|-------|---------|
| apxm-core | Error types, Value type |

## Building

```bash
cargo build -p apxm-backends
```

## Testing

```bash
cargo test -p apxm-backends
```
