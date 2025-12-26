//! LLM response types.

use super::{FinishReason, TokenUsage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Response from an LLM backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    /// The generated content/text
    pub content: String,
    /// Which model produced this response
    pub model: String,
    /// Token usage information
    pub usage: TokenUsage,
    /// Why generation stopped
    pub finish_reason: FinishReason,
    /// Provider-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl LLMResponse {
    /// Create a new response.
    pub fn new(
        content: impl Into<String>,
        model: impl Into<String>,
        usage: TokenUsage,
        finish_reason: FinishReason,
    ) -> Self {
        LLMResponse {
            content: content.into(),
            model: model.into(),
            usage,
            finish_reason,
            metadata: HashMap::new(),
        }
    }

    /// Add provider metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Check if generation completed normally (not via error/truncation).
    pub fn completed_normally(&self) -> bool {
        matches!(
            self.finish_reason,
            FinishReason::Stop | FinishReason::Length
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_creation() {
        let usage = TokenUsage::new(100, 50);
        let response = LLMResponse::new("Hello", "gpt-4", usage, FinishReason::Stop);

        assert_eq!(response.content, "Hello");
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.usage.total_tokens, 150);
        assert!(response.completed_normally());
    }

    #[test]
    fn test_completion_status() {
        let usage = TokenUsage::new(100, 50);

        let stop_response = LLMResponse::new("text", "model", usage.clone(), FinishReason::Stop);
        assert!(stop_response.completed_normally());

        let error_response =
            LLMResponse::new("partial", "model", usage.clone(), FinishReason::Error);
        assert!(!error_response.completed_normally());
    }
}
