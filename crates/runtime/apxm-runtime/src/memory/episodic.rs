//! Episodic Memory - Append-only execution trace log
//!
//! Records execution history for debugging, auditing, and agent reflection.
//! Maintains temporal order and supports querying by execution ID.

use super::config::EpisodicConfig;
use apxm_core::{error::RuntimeError, types::values::Value};
use apxm_backends::SearchResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::RwLock;

type Result<T> = std::result::Result<T, RuntimeError>;

/// A single episodic memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEntry {
    /// Unique entry ID
    pub id: String,
    /// Timestamp when entry was created
    pub timestamp: DateTime<Utc>,
    /// Type of event (e.g., "operation_started", "llm_call", "error")
    pub event_type: String,
    /// Event payload/data
    pub payload: Value,
    /// Execution ID this entry belongs to
    pub execution_id: String,
}

impl EpisodicEntry {
    /// Create a new episodic entry
    pub fn new(event_type: String, payload: Value, execution_id: String) -> Self {
        Self {
            id: uuid::Uuid::now_v7().to_string(),
            timestamp: Utc::now(),
            event_type,
            payload,
            execution_id,
        }
    }
}

/// Episodic Memory - ring buffer of execution traces
pub struct EpisodicMemory {
    entries: RwLock<VecDeque<EpisodicEntry>>,
    max_entries: Option<usize>,
}

impl EpisodicMemory {
    /// Create a new episodic memory with the given configuration
    pub fn new(config: EpisodicConfig) -> Self {
        Self {
            entries: RwLock::new(VecDeque::new()),
            max_entries: config.max_entries,
        }
    }

    /// Create unlimited episodic memory (for testing)
    pub fn unlimited() -> Self {
        Self {
            entries: RwLock::new(VecDeque::new()),
            max_entries: None,
        }
    }

    /// Record a new episodic entry
    pub async fn record(
        &self,
        event_type: String,
        payload: Value,
        execution_id: String,
    ) -> Result<String> {
        let entry = EpisodicEntry::new(event_type, payload, execution_id);
        let entry_id = entry.id.clone();

        let mut entries = self.entries.write().await;

        // Evict oldest entry if at capacity
        if let Some(max) = self.max_entries {
            while entries.len() >= max {
                entries.pop_front();
            }
        }

        entries.push_back(entry);

        Ok(entry_id)
    }

    /// Get all entries for a specific execution ID
    pub async fn get_by_execution(&self, execution_id: &str) -> Result<Vec<EpisodicEntry>> {
        let entries = self.entries.read().await;
        Ok(entries
            .iter()
            .filter(|e| e.execution_id == execution_id)
            .cloned()
            .collect())
    }

    /// Get entry by ID
    pub async fn get_by_id(&self, id: &str) -> Result<Option<EpisodicEntry>> {
        let entries = self.entries.read().await;
        Ok(entries.iter().find(|e| e.id == id).cloned())
    }

    /// Get all entries (ordered by timestamp)
    pub async fn get_all(&self) -> Result<Vec<EpisodicEntry>> {
        let entries = self.entries.read().await;
        Ok(entries.iter().cloned().collect())
    }

    /// Get recent entries (last N)
    pub async fn get_recent(&self, limit: usize) -> Result<Vec<EpisodicEntry>> {
        let entries = self.entries.read().await;
        let start = entries.len().saturating_sub(limit);
        Ok(entries.range(start..).cloned().collect())
    }

    /// Search episodic memory by event type or execution ID
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let entries = self.entries.read().await;
        let mut results = Vec::new();

        for entry in entries.iter() {
            if entry.event_type.contains(query) || entry.execution_id.contains(query) {
                results.push(SearchResult {
                    key: entry.id.clone(),
                    value: serde_json::to_value(entry)
                        .map_err(|e| {
                            RuntimeError::Serialization(format!(
                                "Failed to serialize episodic entry: {}",
                                e
                            ))
                        })?
                        .try_into()
                        .unwrap_or(Value::Null),
                    score: 1.0,
                });

                if results.len() >= limit {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Get current number of entries
    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    /// Check if empty
    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
    }

    /// Clear all entries
    pub async fn clear(&self) -> Result<()> {
        self.entries.write().await.clear();
        Ok(())
    }

    /// Get statistics
    pub async fn stats(&self) -> Result<EpisodicStats> {
        let entries = self.entries.read().await;

        let total_entries = entries.len();
        let unique_executions = entries
            .iter()
            .map(|e| &e.execution_id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let oldest_timestamp = entries.front().map(|e| e.timestamp);
        let newest_timestamp = entries.back().map(|e| e.timestamp);

        Ok(EpisodicStats {
            total_entries,
            unique_executions,
            oldest_timestamp,
            newest_timestamp,
        })
    }
}

/// Statistics for episodic memory
#[derive(Debug, Clone)]
pub struct EpisodicStats {
    pub total_entries: usize,
    pub unique_executions: usize,
    pub oldest_timestamp: Option<DateTime<Utc>>,
    pub newest_timestamp: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_episodic_record() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let episodic = EpisodicMemory::unlimited();

        let entry_id = episodic
            .record(
                "test_event".to_string(),
                Value::String("test_data".to_string()),
                "exec_123".to_string(),
            )
            .await?;

        assert!(!entry_id.is_empty());
        assert_eq!(episodic.len().await, 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_episodic_get_by_execution() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let episodic = EpisodicMemory::unlimited();

        episodic
            .record("event1".to_string(), Value::Null, "exec_1".to_string())
            .await?;
        episodic
            .record("event2".to_string(), Value::Null, "exec_1".to_string())
            .await?;
        episodic
            .record("event3".to_string(), Value::Null, "exec_2".to_string())
            .await?;

        let exec_1_entries = episodic.get_by_execution("exec_1").await?;
        assert_eq!(exec_1_entries.len(), 2);

        let exec_2_entries = episodic.get_by_execution("exec_2").await?;
        assert_eq!(exec_2_entries.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_episodic_capacity_limit() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = EpisodicConfig {
            max_entries: Some(3),
        };
        let episodic = EpisodicMemory::new(config);

        for i in 0..5 {
            episodic
                .record(format!("event{}", i), Value::Null, "exec_123".to_string())
                .await?;
        }

        // Should only keep last 3 entries
        assert_eq!(episodic.len().await, 3);

        let all_entries = episodic.get_all().await?;
        assert_eq!(all_entries[0].event_type, "event2");
        assert_eq!(all_entries[2].event_type, "event4");

        Ok(())
    }

    #[tokio::test]
    async fn test_episodic_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let episodic = EpisodicMemory::unlimited();

        episodic
            .record(
                "operation_started".to_string(),
                Value::Null,
                "exec_1".to_string(),
            )
            .await?;
        episodic
            .record("llm_call".to_string(), Value::Null, "exec_1".to_string())
            .await?;
        episodic
            .record(
                "operation_completed".to_string(),
                Value::Null,
                "exec_1".to_string(),
            )
            .await?;

        let results = episodic.search("operation", 10).await?;
        assert_eq!(results.len(), 2);

        let results = episodic.search("exec_1", 10).await?;
        assert_eq!(results.len(), 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_episodic_recent() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let episodic = EpisodicMemory::unlimited();

        for i in 0..10 {
            episodic
                .record(format!("event{}", i), Value::Null, "exec_123".to_string())
                .await?;
        }

        let recent = episodic.get_recent(3).await?;
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].event_type, "event7");
        assert_eq!(recent[2].event_type, "event9");

        Ok(())
    }

    #[tokio::test]
    async fn test_episodic_stats() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let episodic = EpisodicMemory::unlimited();

        episodic
            .record("event1".to_string(), Value::Null, "exec_1".to_string())
            .await?;
        episodic
            .record("event2".to_string(), Value::Null, "exec_1".to_string())
            .await?;
        episodic
            .record("event3".to_string(), Value::Null, "exec_2".to_string())
            .await?;

        let stats = episodic.stats().await?;
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.unique_executions, 2);
        assert!(stats.oldest_timestamp.is_some());
        assert!(stats.newest_timestamp.is_some());

        Ok(())
    }
}
