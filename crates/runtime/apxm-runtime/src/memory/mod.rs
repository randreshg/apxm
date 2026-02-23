//! Three-tier memory system for APxM runtime
//!
//! Provides:
//! - **STM** (Short-Term Memory): Fast, volatile working memory
//! - **LTM** (Long-Term Memory): Persistent semantic storage
//! - **Episodic**: Append-only execution trace

mod config;
mod episodic;
mod facts;
mod ltm;
mod stm;

pub use config::MemoryConfig;
pub use episodic::{EpisodicEntry, EpisodicMemory};
pub use facts::{Fact, FactFilter, FactResult};
pub use ltm::LongTermMemory;
pub use stm::ShortTermMemory;

use apxm_core::error::RuntimeError;
use std::sync::Arc;

type Result<T> = std::result::Result<T, RuntimeError>;

/// Memory space identifier for routing operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    /// Short-term, volatile memory (fast cache)
    Stm,
    /// Long-term, persistent memory (durable storage)
    Ltm,
    /// Episodic trace memory (execution history)
    Episodic,
}

/// Unified memory system coordinating all three tiers
#[derive(Clone)]
pub struct MemorySystem {
    stm: Arc<ShortTermMemory>,
    ltm: Arc<LongTermMemory>,
    episodic: Arc<EpisodicMemory>,
}

impl MemorySystem {
    /// Create a new memory system with the given configuration
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        let stm = Arc::new(ShortTermMemory::new(config.stm_config)?);
        let ltm = Arc::new(LongTermMemory::new(config.ltm_config).await?);
        let episodic = Arc::new(EpisodicMemory::new(config.episodic_config));

        Ok(Self { stm, ltm, episodic })
    }

    /// Read a value from the specified memory space
    pub async fn read(
        &self,
        space: MemorySpace,
        key: &str,
    ) -> Result<Option<apxm_core::types::values::Value>> {
        match space {
            MemorySpace::Stm => self.stm.get(key).await,
            MemorySpace::Ltm => self.ltm.get(key).await,
            MemorySpace::Episodic => Err(RuntimeError::Memory {
                message: "Episodic memory is append-only, use query instead".to_string(),
                space: Some("episodic".to_string()),
            }),
        }
    }

    /// Write a value to the specified memory space
    pub async fn write(
        &self,
        space: MemorySpace,
        key: String,
        value: apxm_core::types::values::Value,
    ) -> Result<()> {
        match space {
            MemorySpace::Stm => self.stm.put(&key, value).await,
            MemorySpace::Ltm => self.ltm.put(&key, value).await,
            MemorySpace::Episodic => Err(RuntimeError::Memory {
                message: "Episodic memory is append-only, use record instead".to_string(),
                space: Some("episodic".to_string()),
            }),
        }
    }

    /// Delete a key from the specified memory space
    pub async fn delete(&self, space: MemorySpace, key: &str) -> Result<()> {
        match space {
            MemorySpace::Stm => self.stm.delete(key).await,
            MemorySpace::Ltm => self.ltm.delete(key).await,
            MemorySpace::Episodic => Err(RuntimeError::Memory {
                message: "Episodic memory is append-only, cannot delete".to_string(),
                space: Some("episodic".to_string()),
            }),
        }
    }

    /// Search memory space (substring matching on keys)
    pub async fn search(
        &self,
        space: MemorySpace,
        query: &str,
        limit: usize,
    ) -> Result<Vec<apxm_backends::SearchResult>> {
        match space {
            MemorySpace::Stm => self.stm.search(query, limit).await,
            MemorySpace::Ltm => self.ltm.search(query, limit).await,
            MemorySpace::Episodic => self.episodic.search(query, limit).await,
        }
    }

    /// Record an entry in episodic memory
    pub async fn record_episode(
        &self,
        event_type: String,
        payload: apxm_core::types::values::Value,
        execution_id: String,
    ) -> Result<String> {
        self.episodic
            .record(event_type, payload, execution_id)
            .await
    }

    /// Record an entry with explicit event type and payload map
    pub async fn record_episodic_event(
        &self,
        execution_id: String,
        event_type: &str,
        payload: apxm_core::types::values::Value,
    ) -> Result<String> {
        self.record_episode(event_type.to_string(), payload, execution_id)
            .await
    }

    /// Query episodic memory by execution ID
    pub async fn query_episodes(&self, execution_id: &str) -> Result<Vec<EpisodicEntry>> {
        self.episodic.get_by_execution(execution_id).await
    }

    /// Get STM reference (for advanced use cases)
    pub fn stm(&self) -> &ShortTermMemory {
        &self.stm
    }

    /// Get LTM reference (for advanced use cases)
    pub fn ltm(&self) -> &LongTermMemory {
        &self.ltm
    }

    /// Get Episodic reference (for advanced use cases)
    pub fn episodic(&self) -> &EpisodicMemory {
        &self.episodic
    }

    /// Clear all memory tiers (useful for testing)
    pub async fn clear_all(&self) -> Result<()> {
        self.stm.clear().await?;
        self.ltm.clear().await?;
        self.episodic.clear().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxm_core::types::values::Value;

    #[tokio::test]
    async fn test_memory_system_stm() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = MemoryConfig::in_memory_ltm();
        let system = MemorySystem::new(config).await?;

        // Write to STM
        system
            .write(
                MemorySpace::Stm,
                "test_key".to_string(),
                Value::String("test_value".to_string()),
            )
            .await?;

        // Read from STM
        let result = system.read(MemorySpace::Stm, "test_key").await?;
        assert_eq!(result, Some(Value::String("test_value".to_string())));

        // Delete from STM
        system.delete(MemorySpace::Stm, "test_key").await?;
        let result = system.read(MemorySpace::Stm, "test_key").await?;
        assert_eq!(result, None);

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_system_ltm() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = MemoryConfig::in_memory_ltm();
        let system = MemorySystem::new(config).await?;

        // Write to LTM
        system
            .write(
                MemorySpace::Ltm,
                "persistent_key".to_string(),
                Value::String("persistent_value".to_string()),
            )
            .await?;

        // Read from LTM
        let result = system.read(MemorySpace::Ltm, "persistent_key").await?;
        assert_eq!(result, Some(Value::String("persistent_value".to_string())));

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_system_episodic() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = MemoryConfig::in_memory_ltm();
        let system = MemorySystem::new(config).await?;

        // Record episode
        let entry_id = system
            .record_episode(
                "test_event".to_string(),
                Value::String("event_data".to_string()),
                "exec_123".to_string(),
            )
            .await?;

        assert!(!entry_id.is_empty());

        // Query episodes
        let episodes = system.query_episodes("exec_123").await?;
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].event_type, "test_event");

        Ok(())
    }
}
