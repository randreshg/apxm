use crate::storage::StorageResult;
use apxm_core::types::values::Value;
use async_trait::async_trait;

/// Result of a search query
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub key: String,
    pub value: Value,
    pub score: f64,
}

/// Storage backend trait for pluggable storage implementations
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store a key-value pair
    async fn put(&self, key: &str, value: Value) -> StorageResult<()>;

    /// Retrieve a value by key
    async fn get(&self, key: &str) -> StorageResult<Option<Value>>;

    /// Delete a key-value pair
    async fn delete(&self, key: &str) -> StorageResult<()>;

    /// Check if a key exists
    async fn exists(&self, key: &str) -> StorageResult<bool> {
        Ok(self.get(key).await?.is_some())
    }

    /// List all keys (optional, may not be supported by all backends)
    async fn list_keys(&self) -> StorageResult<Vec<String>> {
        Err(apxm_core::error::RuntimeError::Memory {
            message: "list_keys not supported by this backend".to_string(),
            space: None,
        })
    }

    /// Search for keys/values matching a query
    /// Default implementation does substring matching on keys
    async fn search(&self, query: &str, limit: usize) -> StorageResult<Vec<SearchResult>> {
        if limit == 0 || query.is_empty() {
            return Ok(Vec::new());
        }

        let keys = self.list_keys().await?;
        let mut results = Vec::with_capacity(limit);

        for key in keys.into_iter().filter(|k| k.contains(query)) {
            if let Some(value) = self.get(&key).await? {
                results.push(SearchResult {
                    key,
                    value,
                    score: 1.0,
                });

                if results.len() == limit {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Vector/semantic search (placeholder for future implementation)
    async fn search_vector(&self, _query: &str, limit: usize) -> StorageResult<Vec<SearchResult>> {
        // Default to regular search for now
        self.search(_query, limit).await
    }

    /// Clear all data (useful for testing)
    async fn clear(&self) -> StorageResult<()>;

    /// Get backend metadata/stats
    async fn stats(&self) -> StorageResult<BackendStats>;
}

/// Backend statistics
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    pub total_keys: usize,
    pub total_size_bytes: usize,
    pub backend_type: String,
}
