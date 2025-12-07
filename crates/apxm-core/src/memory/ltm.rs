//! Long-term memory (LTM) types and trait definitions.
//!
//! These types describe the contract for queryable, persistent memory backends.
//! Only the data shapes and trait are defined here; concrete implementations
//! live in runtime crates.

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::memory::MemorySpace;
use crate::types::Value;

/// Query parameters for LTM operations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LTMQuery {
    /// The search query (exact or semantic depending on backend).
    pub query: String,
    /// Target memory space; defaults to LTM but kept explicit for routing.
    pub space: MemorySpace,
    /// Maximum number of results to return.
    pub limit: Option<usize>,
}

impl LTMQuery {
    /// Default limit applied when none is provided.
    pub const DEFAULT_LIMIT: usize = 10;

    /// Creates a new query targeting LTM by default.
    pub fn new(query: impl Into<String>) -> Self {
        LTMQuery {
            query: query.into(),
            space: MemorySpace::Ltm,
            limit: None,
        }
    }

    /// Sets a limit for the query.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Changes the target memory space.
    pub fn with_space(mut self, space: MemorySpace) -> Self {
        self.space = space;
        self
    }

    /// Returns the effective limit, applying the provided default when needed.
    pub fn effective_limit(&self) -> usize {
        self.limit.unwrap_or(Self::DEFAULT_LIMIT).max(1)
    }

    /// Performs basic validation of the query parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.query.trim().is_empty() {
            return Err("LTM query cannot be empty".to_string());
        }
        if let Some(limit) = self.limit
            && limit == 0
        {
            return Err("LTM query limit must be greater than zero".to_string());
        }
        Ok(())
    }
}

/// Result item from an LTM query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LTMResult {
    /// Optional key or identifier for the stored item.
    pub key: Option<String>,
    /// Retrieved value.
    pub value: Value,
    /// Backend-specific relevance score (higher is better).
    pub score: f64,
    /// Optional latency reported by the backend, useful for observability.
    pub latency: Option<Duration>,
}

impl LTMResult {
    /// Creates a new result with a value and score.
    pub fn new(value: Value, score: f64) -> Self {
        LTMResult {
            key: None,
            value,
            score,
            latency: None,
        }
    }

    /// Attaches a key to the result.
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Records an optional latency measurement.
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = Some(latency);
        self
    }
}

/// Backend contract for LTM storage engines.
pub trait LTMBackend: Send + Sync {
    /// Error type produced by the backend.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Stores a value under a given key.
    fn store(&self, key: String, value: Value) -> Result<(), Self::Error>;

    /// Performs a query, returning ranked results.
    fn query(&self, query: &LTMQuery) -> Result<Vec<LTMResult>, Self::Error>;

    /// Deletes a value by key.
    fn delete(&self, key: &str) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct MockError;

    impl fmt::Display for MockError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "mock backend error")
        }
    }

    impl std::error::Error for MockError {}

    #[derive(Default)]
    struct MockBackend {
        data: Mutex<HashMap<String, Value>>,
    }

    impl LTMBackend for MockBackend {
        type Error = MockError;

        fn store(&self, key: String, value: Value) -> Result<(), Self::Error> {
            self.data.lock().unwrap().insert(key, value);
            Ok(())
        }

        fn query(&self, query: &LTMQuery) -> Result<Vec<LTMResult>, Self::Error> {
            let data = self.data.lock().unwrap();
            let mut results: Vec<LTMResult> = data
                .iter()
                .filter(|(k, _)| k.contains(&query.query))
                .map(|(k, v)| LTMResult::new(v.clone(), 1.0).with_key(k.clone()))
                .collect();
            results.truncate(query.effective_limit());
            Ok(results)
        }

        fn delete(&self, key: &str) -> Result<(), Self::Error> {
            self.data.lock().unwrap().remove(key);
            Ok(())
        }
    }

    #[test]
    fn test_query_validation() {
        let valid = LTMQuery::new("hello").with_limit(5);
        assert!(valid.validate().is_ok());
        assert_eq!(valid.effective_limit(), 5);

        let empty = LTMQuery::new("   ");
        assert!(empty.validate().is_err());

        let zero_limit = LTMQuery::new("q").with_limit(0);
        assert!(zero_limit.validate().is_err());
    }

    #[test]
    fn test_result_builder_and_serialization() {
        let result = LTMResult::new(Value::String("value".into()), 0.9)
            .with_key("id")
            .with_latency(Duration::from_millis(5));
        let json = serde_json::to_string(&result).expect("serialize");
        let restored: LTMResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.key.as_deref(), Some("id"));
        assert_eq!(restored.value, Value::String("value".into()));
    }

    #[test]
    fn test_mock_backend_behaviour() {
        let backend = MockBackend::default();
        backend
            .store("alpha".into(), Value::String("one".into()))
            .expect("store alpha");
        backend
            .store("beta".into(), Value::String("two".into()))
            .expect("store beta");

        let query = LTMQuery::new("a");
        let results = backend.query(&query).expect("query");
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .any(|item| item.key.as_deref() == Some("alpha"))
        );
        assert!(
            results
                .iter()
                .any(|item| item.key.as_deref() == Some("beta"))
        );

        backend.delete("alpha").expect("delete alpha");
        let after_delete = backend.query(&query).expect("query");
        assert_eq!(after_delete.len(), 1);
        assert_eq!(after_delete[0].key.as_deref(), Some("beta"));
    }

    #[test]
    fn test_ltmquery_serialization() {
        let query = LTMQuery::new("rust")
            .with_limit(3)
            .with_space(MemorySpace::Ltm);
        let json = serde_json::to_string(&query).expect("serialize");
        let restored: LTMQuery = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.query, "rust");
        assert_eq!(restored.limit, Some(3));
        assert_eq!(restored.space, MemorySpace::Ltm);
    }
}
