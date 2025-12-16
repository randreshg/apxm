//! APxM Storage - High-performance storage backends
//!
//! Provides pluggable storage implementations optimized for:
//! - In-memory caching (HashMap-based)
//! - Persistent key-value storage (SQLite with connection pooling)
//! - Future: Embedded KV stores (RocksDB, sled)

mod backend;
mod kv;
mod memory;
mod sqlite;

pub use backend::{BackendStats, SearchResult, StorageBackend};
pub use kv::RedbBackend;
pub use memory::InMemoryBackend;
pub use sqlite::SqliteBackend;

pub use apxm_core::{error::RuntimeError, types::values::Value};

/// Result type using RuntimeError from core
pub type StorageResult<T> = Result<T, RuntimeError>;
