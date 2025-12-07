//! Episodic memory types.
//!
//! Episodic memory captures execution traces in an append-only log. These types
//! define the shape of stored entries and queries used to retrieve them.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::types::Value;

/// A single episodic memory entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpisodicEntry {
    /// Correlates entries to an execution.
    pub execution_id: String,
    /// When the entry was recorded.
    pub timestamp: DateTime<Utc>,
    /// Operations captured in this episode.
    pub operations: Vec<Value>,
    /// Results or outputs produced by the operations.
    pub results: Vec<Value>,
    /// Optional metadata associated with the entry.
    pub metadata: Option<Value>,
}

impl EpisodicEntry {
    /// Creates a new entry using the current time.
    pub fn new(
        execution_id: impl Into<String>,
        operations: Vec<Value>,
        results: Vec<Value>,
    ) -> Self {
        EpisodicEntry {
            execution_id: execution_id.into(),
            timestamp: Utc::now(),
            operations,
            results,
            metadata: None,
        }
    }

    /// Creates an entry with an explicit timestamp.
    pub fn with_timestamp(
        execution_id: impl Into<String>,
        timestamp: DateTime<Utc>,
        operations: Vec<Value>,
        results: Vec<Value>,
    ) -> Self {
        EpisodicEntry {
            execution_id: execution_id.into(),
            timestamp,
            operations,
            results,
            metadata: None,
        }
    }

    /// Attaches metadata to the entry.
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Query parameters used to filter episodic entries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct EpisodicQuery {
    /// Specific execution identifier to match.
    pub execution_id: Option<String>,
    /// Inclusive start time filter.
    pub start_time: Option<DateTime<Utc>>,
    /// Inclusive end time filter.
    pub end_time: Option<DateTime<Utc>>,
    /// Substring to match inside operation values (string-like).
    pub operation_contains: Option<String>,
    /// Optional limit for returned entries.
    pub limit: Option<usize>,
}

impl EpisodicQuery {
    /// Creates an empty query that matches everything.
    pub fn new() -> Self {
        EpisodicQuery {
            execution_id: None,
            start_time: None,
            end_time: None,
            operation_contains: None,
            limit: None,
        }
    }

    /// Sets the execution id filter.
    pub fn with_execution_id(mut self, execution_id: impl Into<String>) -> Self {
        self.execution_id = Some(execution_id.into());
        self
    }

    /// Sets a time range filter (inclusive).
    pub fn with_time_range(
        mut self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Self {
        self.start_time = start;
        self.end_time = end;
        self
    }

    /// Sets an operation substring filter.
    pub fn with_operation_contains(mut self, fragment: impl Into<String>) -> Self {
        self.operation_contains = Some(fragment.into());
        self
    }

    /// Sets a limit on the number of returned entries.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Applies basic validation to the query.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(limit) = self.limit
            && limit == 0
        {
            return Err("Episodic query limit must be greater than zero".to_string());
        }
        if let (Some(start), Some(end)) = (self.start_time, self.end_time)
            && start > end
        {
            return Err("Episodic query start_time must be before end_time".to_string());
        }
        Ok(())
    }

    /// Returns the effective limit, falling back to the provided default.
    pub fn effective_limit(&self, default: usize) -> usize {
        self.limit.unwrap_or(default).max(1)
    }

    /// Returns true if the entry matches all configured filters.
    pub fn matches(&self, entry: &EpisodicEntry) -> bool {
        if let Some(exec) = &self.execution_id
            && &entry.execution_id != exec
        {
            return false;
        }

        if let Some(start) = self.start_time
            && entry.timestamp < start
        {
            return false;
        }

        if let Some(end) = self.end_time
            && entry.timestamp > end
        {
            return false;
        }

        if let Some(fragment) = &self.operation_contains {
            let matches_fragment = entry.operations.iter().any(|op| match op {
                Value::String(s) => s.contains(fragment),
                Value::Array(values) => values.iter().any(|nested| {
                    if let Value::String(s) = nested {
                        s.contains(fragment)
                    } else {
                        false
                    }
                }),
                _ => false,
            });

            if !matches_fragment {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    #[test]
    fn test_entry_construction_and_metadata() {
        let operations = vec![Value::String("op".into())];
        let results = vec![Value::Bool(true)];
        let entry = EpisodicEntry::new("exec-1", operations.clone(), results.clone())
            .with_metadata(Value::String("meta".into()));

        assert_eq!(entry.execution_id, "exec-1");
        assert_eq!(entry.operations, operations);
        assert_eq!(entry.results, results);
        assert_eq!(entry.metadata, Some(Value::String("meta".into())));

        let json = serde_json::to_string(&entry).expect("serialize");
        let restored: EpisodicEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.execution_id, "exec-1");
    }

    #[test]
    fn test_query_matching_by_execution_and_time() {
        let timestamp = Utc::now();
        let entry = EpisodicEntry::with_timestamp(
            "run-42",
            timestamp,
            vec![Value::String("plan".into())],
            vec![Value::String("ok".into())],
        );

        let query = EpisodicQuery::new()
            .with_execution_id("run-42")
            .with_time_range(
                Some(timestamp - ChronoDuration::seconds(1)),
                Some(timestamp),
            );
        assert!(query.matches(&entry));

        let non_matching = query.with_execution_id("other");
        assert!(!non_matching.matches(&entry));
    }

    #[test]
    fn test_query_matching_by_operation_fragment() {
        let entry = EpisodicEntry::with_timestamp(
            "run-1",
            Utc::now(),
            vec![Value::Array(vec![Value::String("search-web".into())])],
            vec![],
        );

        let query = EpisodicQuery::new().with_operation_contains("web");
        assert!(query.matches(&entry));

        let query_none = EpisodicQuery::new().with_operation_contains("db");
        assert!(!query_none.matches(&entry));
    }

    #[test]
    fn test_query_validation_and_limit() {
        let query = EpisodicQuery::new().with_limit(5);
        assert_eq!(query.effective_limit(10), 5);
        assert!(query.validate().is_ok());

        let invalid_limit = EpisodicQuery::new().with_limit(0);
        assert!(invalid_limit.validate().is_err());

        let start = Utc::now();
        let end = start - ChronoDuration::seconds(1);
        let invalid_range = EpisodicQuery::new().with_time_range(Some(start), Some(end));
        assert!(invalid_range.validate().is_err());
    }

    #[test]
    fn test_query_serialization_round_trip() {
        let query = EpisodicQuery::new()
            .with_execution_id("exec")
            .with_time_range(None, None)
            .with_operation_contains("op")
            .with_limit(2);
        let json = serde_json::to_string(&query).expect("serialize");
        let restored: EpisodicQuery = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.execution_id, Some("exec".into()));
        assert_eq!(restored.limit, Some(2));
    }
}
