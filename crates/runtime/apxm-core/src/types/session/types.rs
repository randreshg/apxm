//! Session and conversation types for APXM.
//!
//! This module contains types related to chat sessions, messages, and conversation management
//! that are shared across multiple crates (chat, REPL, tooling, etc.).

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::types::MessageId;

/// Role of a message in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// Message from the user
    User,
    /// Message from the AI assistant
    Assistant,
    /// System-level message (prompts, instructions)
    System,
    /// Error message
    Error,
}

impl fmt::Display for MessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::System => write!(f, "system"),
            MessageRole::Error => write!(f, "error"),
        }
    }
}

impl MessageRole {
    /// Check if this is a user message
    pub fn is_user(&self) -> bool {
        matches!(self, MessageRole::User)
    }

    /// Check if this is an assistant message
    pub fn is_assistant(&self) -> bool {
        matches!(self, MessageRole::Assistant)
    }

    /// Check if this is a system message
    pub fn is_system(&self) -> bool {
        matches!(self, MessageRole::System)
    }

    /// Check if this is an error message
    pub fn is_error(&self) -> bool {
        matches!(self, MessageRole::Error)
    }
}

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique identifier for this message
    pub id: MessageId,
    /// Role of the message sender
    pub role: MessageRole,
    /// Content of the message
    pub content: String,
    /// Timestamp when the message was created (Unix timestamp)
    pub timestamp: i64,
    /// Optional metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MessageMetadata>,
}

/// Metadata associated with a message
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageMetadata {
    /// Token count for this message
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_count: Option<usize>,
    /// Model used to generate this message (for assistant messages)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Finish reason (for assistant messages)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Associated intent (if classified)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    /// Confidence score for intent classification
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl Message {
    /// Create a new message
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            id: MessageId::generate(),
            role,
            content: content.into(),
            timestamp: chrono::Utc::now().timestamp(),
            metadata: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }

    /// Create an error message
    pub fn error(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Error, content)
    }

    /// Set metadata for this message
    pub fn with_metadata(mut self, metadata: MessageMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get token count from metadata
    pub fn token_count(&self) -> Option<usize> {
        self.metadata.as_ref().and_then(|m| m.token_count)
    }
}

/// An example for prompt engineering and documentation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Example {
    /// Description of what this example demonstrates
    pub description: String,
    /// User input text
    pub user_input: String,
    /// Expected AIS output
    pub ais_output: String,
}

impl Example {
    /// Create a new example
    pub fn new(
        description: impl Into<String>,
        user_input: impl Into<String>,
        ais_output: impl Into<String>,
    ) -> Self {
        Self {
            description: description.into(),
            user_input: user_input.into(),
            ais_output: ais_output.into(),
        }
    }

    /// Format this example as markdown
    pub fn to_markdown(&self) -> String {
        format!(
            "### {}\n\n**Input:**\n```\n{}\n```\n\n**Output:**\n```ais\n{}\n```\n",
            self.description, self.user_input, self.ais_output
        )
    }
}

impl fmt::Display for Example {
    /// Canonical prompt-friendly formatting for Example
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "**Example: {}**\nUser: {}\nAIS:\n```ais\n{}\n```\n",
            self.description, self.user_input, self.ais_output
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "Hello");
        assert!(msg.metadata.is_none());
    }

    #[test]
    fn test_message_with_metadata() {
        let metadata = MessageMetadata {
            token_count: Some(42),
            model: Some("gpt-4".to_string()),
            ..Default::default()
        };

        let msg = Message::assistant("Response").with_metadata(metadata);
        assert_eq!(msg.token_count(), Some(42));
        assert_eq!(
            msg.metadata.as_ref().and_then(|m| m.model.as_deref()),
            Some("gpt-4")
        );
    }

    #[test]
    fn test_example_creation() {
        let example = Example::new("Simple addition", "Add 2 and 3", "add(2, 3)");
        assert_eq!(example.description, "Simple addition");
        assert_eq!(example.user_input, "Add 2 and 3");
        assert_eq!(example.ais_output, "add(2, 3)");
    }

}
