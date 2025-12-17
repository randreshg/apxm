//! Long-Term Memory (LTM) - Persistent semantic storage
//!
//! Uses persistent backends (SQLite, Redb) for durable storage.
//! Supports semantic search and long-term knowledge retention.

use super::config::{LtmBackend, LtmConfig};
use apxm_core::{error::RuntimeError, types::values::Value};
use apxm_storage::{InMemoryBackend, RedbBackend, SearchResult, SqliteBackend, StorageBackend};
use std::sync::Arc;

type Result<T> = std::result::Result<T, RuntimeError>;

/// Long-Term Memory layer with pluggable backend
pub struct LongTermMemory {
    backend: Arc<dyn StorageBackend + Send + Sync>,
}

impl LongTermMemory {
    /// Create a new LTM instance with the given configuration
    pub async fn new(config: LtmConfig) -> Result<Self> {
        let backend: Arc<dyn StorageBackend + Send + Sync> = match config.backend {
            LtmBackend::Memory => Arc::new(InMemoryBackend::unlimited()),
            LtmBackend::Sqlite => {
                let path = config.path.ok_or_else(|| RuntimeError::Memory {
                    message: "SQLite backend requires a path".to_string(),
                    space: Some("ltm".to_string()),
                })?;
                Arc::new(SqliteBackend::new(path, config.max_connections).await?)
            }
            LtmBackend::Redb => {
                let path = config.path.ok_or_else(|| RuntimeError::Memory {
                    message: "Redb backend requires a path".to_string(),
                    space: Some("ltm".to_string()),
                })?;
                Arc::new(RedbBackend::new(path).await?)
            }
        };

        Ok(Self { backend })
    }

    /// Create in-memory LTM (for testing)
    pub async fn in_memory() -> Result<Self> {
        Ok(Self {
            backend: Arc::new(InMemoryBackend::unlimited()),
        })
    }

    /// Store a value in LTM
    pub async fn put(&self, key: &str, value: Value) -> Result<()> {
        self.backend.put(key, value).await
    }

    /// Retrieve a value from LTM
    pub async fn get(&self, key: &str) -> Result<Option<Value>> {
        self.backend.get(key).await
    }

    /// Delete a value from LTM
    pub async fn delete(&self, key: &str) -> Result<()> {
        self.backend.delete(key).await
    }

    /// Check if a key exists in LTM
    pub async fn exists(&self, key: &str) -> Result<bool> {
        self.backend.exists(key).await
    }

    /// List all keys in LTM
    pub async fn list_keys(&self) -> Result<Vec<String>> {
        self.backend.list_keys().await
    }

    /// Search LTM by key substring
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.backend.search(query, limit).await
    }

    /// Semantic/vector search (currently falls back to substring search)
    pub async fn search_semantic(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.backend.search_vector(query, limit).await
    }

    /// Clear all entries from LTM
    pub async fn clear(&self) -> Result<()> {
        self.backend.clear().await
    }

    /// Get LTM statistics
    pub async fn stats(&self) -> Result<apxm_storage::BackendStats> {
        self.backend.stats().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxm_core::types::values::Number;

    #[tokio::test]
    async fn test_ltm_basic_operations() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let ltm = LongTermMemory::in_memory().await?;

        // Put and get
        ltm.put("key1", Value::String("value1".to_string())).await?;
        let result = ltm.get("key1").await?;
        assert_eq!(result, Some(Value::String("value1".to_string())));

        // Exists
        assert!(ltm.exists("key1").await?);
        assert!(!ltm.exists("nonexistent").await?);

        // Delete
        ltm.delete("key1").await?;
        assert!(!ltm.exists("key1").await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_ltm_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let ltm = LongTermMemory::in_memory().await?;

        ltm.put("fact:earth", Value::String("third planet".to_string()))
            .await?;
        ltm.put("fact:mars", Value::String("fourth planet".to_string()))
            .await?;
        ltm.put("note:meeting", Value::String("tomorrow at 3pm".to_string()))
            .await?;

        let results = ltm.search("fact", 10).await?;
        assert_eq!(results.len(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_ltm_sqlite_backend() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = LtmConfig::sqlite(":memory:");
        let ltm = LongTermMemory::new(config).await?;

        ltm.put("persistent", Value::Number(Number::Integer(42)))
            .await?;

        let result = ltm.get("persistent").await?;
        assert_eq!(result, Some(Value::Number(Number::Integer(42))));

        Ok(())
    }

    #[tokio::test]
    async fn test_ltm_stats() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let ltm = LongTermMemory::in_memory().await?;

        ltm.put("key1", Value::String("value1".to_string())).await?;
        ltm.put("key2", Value::String("value2".to_string())).await?;

        let stats = ltm.stats().await?;
        assert_eq!(stats.total_keys, 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_ltm_clear() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let ltm = LongTermMemory::in_memory().await?;

        ltm.put("key1", Value::String("value1".to_string())).await?;
        ltm.put("key2", Value::String("value2".to_string())).await?;

        ltm.clear().await?;

        let stats = ltm.stats().await?;
        assert_eq!(stats.total_keys, 0);

        Ok(())
    }
}
