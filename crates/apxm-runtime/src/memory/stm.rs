//! Short-Term Memory (STM) - Fast, volatile working memory
//!
//! Uses in-memory backend for microsecond-latency operations.
//! Ideal for temporary results, intermediate values, and caching.

use super::config::StmConfig;
use apxm_core::{error::RuntimeError, types::values::Value};
use apxm_storage::{InMemoryBackend, SearchResult, StorageBackend};

type Result<T> = std::result::Result<T, RuntimeError>;

/// Short-Term Memory layer using in-memory backend
pub struct ShortTermMemory {
    backend: InMemoryBackend,
}

impl ShortTermMemory {
    /// Create a new STM instance with the given configuration
    pub fn new(config: StmConfig) -> Result<Self> {
        let backend = InMemoryBackend::new(config.max_entries);
        Ok(Self { backend })
    }

    /// Create unlimited-capacity STM (for testing)
    pub fn unlimited() -> Self {
        Self {
            backend: InMemoryBackend::unlimited(),
        }
    }

    /// Store a value in STM
    pub async fn put(&self, key: &str, value: Value) -> Result<()> {
        self.backend.put(key, value).await
    }

    /// Retrieve a value from STM
    pub async fn get(&self, key: &str) -> Result<Option<Value>> {
        self.backend.get(key).await
    }

    /// Delete a value from STM
    pub async fn delete(&self, key: &str) -> Result<()> {
        self.backend.delete(key).await
    }

    /// Check if a key exists in STM
    pub async fn exists(&self, key: &str) -> Result<bool> {
        self.backend.exists(key).await
    }

    /// List all keys in STM
    pub async fn list_keys(&self) -> Result<Vec<String>> {
        self.backend.list_keys().await
    }

    /// Search STM by key substring
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.backend.search(query, limit).await
    }

    /// Clear all entries from STM
    pub async fn clear(&self) -> Result<()> {
        self.backend.clear().await
    }

    /// Get current number of entries
    pub async fn len(&self) -> usize {
        self.backend.len().await
    }

    /// Check if STM is empty
    pub async fn is_empty(&self) -> bool {
        self.backend.is_empty().await
    }

    /// Get STM statistics
    pub async fn stats(&self) -> Result<apxm_storage::BackendStats> {
        self.backend.stats().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxm_core::types::values::Number;
    use std::error::Error;

    #[tokio::test]
    async fn test_stm_basic_operations() -> std::result::Result<(), Box<dyn Error>> {
        let stm = ShortTermMemory::unlimited();

        // Put and get
        stm.put("key1", Value::String("value1".to_string())).await?;
        let result = stm.get("key1").await?;
        assert_eq!(result, Some(Value::String("value1".to_string())));

        // Exists
        assert!(stm.exists("key1").await?);
        assert!(!stm.exists("nonexistent").await?);

        // Delete
        stm.delete("key1").await?;
        assert!(!stm.exists("key1").await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_stm_capacity_limit() -> std::result::Result<(), Box<dyn Error>> {
        let config = StmConfig {
            max_entries: Some(2),
        };
        let stm = ShortTermMemory::new(config)?;

        stm.put("key1", Value::String("value1".to_string())).await?;
        stm.put("key2", Value::String("value2".to_string())).await?;

        // Should fail on third insert (do not propagate)
        let result = stm.put("key3", Value::String("value3".to_string())).await;
        assert!(matches!(result, Err(RuntimeError::Memory { .. })));

        // Should allow update of existing key
        stm.put("key1", Value::String("updated".to_string()))
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_stm_search() -> std::result::Result<(), Box<dyn Error>> {
        let stm = ShortTermMemory::unlimited();

        stm.put("user:1", Value::String("alice".to_string()))
            .await?;
        stm.put("user:2", Value::String("bob".to_string())).await?;
        stm.put("post:1", Value::String("hello".to_string()))
            .await?;

        let results = stm.search("user", 10).await?;
        assert_eq!(results.len(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_stm_list_keys() -> std::result::Result<(), Box<dyn Error>> {
        let stm = ShortTermMemory::unlimited();

        stm.put("a", Value::Number(Number::Integer(1))).await?;
        stm.put("b", Value::Number(Number::Integer(2))).await?;
        stm.put("c", Value::Number(Number::Integer(3))).await?;

        let keys = stm.list_keys().await?;
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
        assert!(keys.contains(&"c".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_stm_clear() -> std::result::Result<(), Box<dyn Error>> {
        let stm = ShortTermMemory::unlimited();

        stm.put("key1", Value::String("value1".to_string())).await?;
        stm.put("key2", Value::String("value2".to_string())).await?;

        assert_eq!(stm.len().await, 2);

        stm.clear().await?;

        assert_eq!(stm.len().await, 0);
        assert!(stm.is_empty().await);

        Ok(())
    }
}
