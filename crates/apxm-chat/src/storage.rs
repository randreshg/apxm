//! Session storage layer using apxm-storage

use crate::error::{ChatError, ChatResult};
use apxm_core::types::values::Value;
use apxm_storage::{SqliteBackend, StorageBackend};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    /// Unique session ID
    pub id: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,

    /// Number of messages in session
    pub message_count: usize,

    /// Model used for this session
    pub model: String,
}

/// Message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message role: "user", "assistant", or "system"
    pub role: String,

    /// Message content
    pub content: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Storage layer for chat sessions
pub struct SessionStorage {
    backend: SqliteBackend,
}

impl SessionStorage {
    /// Create new session storage
    pub async fn new(storage_path: &Path) -> ChatResult<Self> {
        // Ensure parent directory exists
        if let Some(parent) = storage_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Create SQLite backend with connection pool
        let backend = SqliteBackend::new(storage_path, Some(4)).await?;

        Ok(Self { backend })
    }

    /// Create a new session
    pub async fn create_session(&self, info: &SessionInfo) -> ChatResult<()> {
        let key = format!("session:{}", info.id);
        let value = Value::String(serde_json::to_string(info)?);
        self.backend.put(&key, value).await?;
        Ok(())
    }

    /// Get session metadata
    pub async fn get_session(&self, id: &str) -> ChatResult<Option<SessionInfo>> {
        let key = format!("session:{}", id);

        match self.backend.get(&key).await? {
            Some(Value::String(json)) => {
                let info: SessionInfo = serde_json::from_str(&json)?;
                Ok(Some(info))
            }
            Some(_) => Err(ChatError::Storage(apxm_core::error::RuntimeError::Memory {
                message: "Session data is not a string".to_string(),
                space: None,
            })),
            None => Ok(None),
        }
    }

    /// List all sessions
    pub async fn list_sessions(&self) -> ChatResult<Vec<SessionInfo>> {
        let mut sessions = Vec::new();

        // Search for all session keys
        let results = self.backend.search("session:", 1000).await?;

        for result in results {
            if let Value::String(json) = result.value {
                if let Ok(info) = serde_json::from_str::<SessionInfo>(&json) {
                    sessions.push(info);
                }
            }
        }

        // Sort by updated_at (most recent first)
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        Ok(sessions)
    }

    /// Update session metadata
    pub async fn update_session(&self, info: &SessionInfo) -> ChatResult<()> {
        self.create_session(info).await
    }

    /// Add a message to a session
    pub async fn add_message(&self, session_id: &str, message: &Message) -> ChatResult<()> {
        let key = format!("session:{}:messages", session_id);

        // Get existing messages
        let mut messages = self.get_messages(session_id).await?;

        // Append new message
        messages.push(message.clone());

        // Store updated messages
        let value = Value::String(serde_json::to_string(&messages)?);
        self.backend.put(&key, value).await?;

        Ok(())
    }

    /// Get all messages for a session
    pub async fn get_messages(&self, session_id: &str) -> ChatResult<Vec<Message>> {
        let key = format!("session:{}:messages", session_id);

        match self.backend.get(&key).await? {
            Some(Value::String(json)) => {
                let messages: Vec<Message> = serde_json::from_str(&json)?;
                Ok(messages)
            }
            Some(_) => Err(ChatError::Storage(apxm_core::error::RuntimeError::Memory {
                message: "Messages data is not a string".to_string(),
                space: None,
            })),
            None => Ok(Vec::new()),
        }
    }

    /// Delete a session and its messages
    pub async fn delete_session(&self, id: &str) -> ChatResult<()> {
        let session_key = format!("session:{}", id);
        let messages_key = format!("session:{}:messages", id);

        self.backend.delete(&session_key).await?;
        self.backend.delete(&messages_key).await?;

        Ok(())
    }
}
