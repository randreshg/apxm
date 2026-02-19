//! LLM response types.

use super::{FinishReason, TokenUsage, ToolCall};
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
    /// Tool calls requested by the model (if any)
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
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
            tool_calls: Vec::new(),
        }
    }

    /// Add provider metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Add tool calls to the response.
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Check if the response contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Check if generation completed normally (not via error/truncation).
    pub fn completed_normally(&self) -> bool {
        matches!(
            self.finish_reason,
            FinishReason::Stop | FinishReason::Length
        )
    }

    /// Check if the model wants to use tools.
    pub fn wants_tool_use(&self) -> bool {
        self.finish_reason == FinishReason::ToolUse || self.has_tool_calls()
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
        assert!(!response.has_tool_calls());
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

    #[test]
    fn test_response_with_tool_calls() {
        let usage = TokenUsage::new(100, 50);
        let tool_calls = vec![
            ToolCall::new("call_1", "bash", serde_json::json!({"command": "ls"})),
            ToolCall::new("call_2", "read", serde_json::json!({"path": "file.txt"})),
        ];

        let response =
            LLMResponse::new("", "gpt-4", usage, FinishReason::ToolUse).with_tool_calls(tool_calls);

        assert!(response.has_tool_calls());
        assert!(response.wants_tool_use());
        assert_eq!(response.tool_calls.len(), 2);
        assert_eq!(response.tool_calls[0].name, "bash");
    }

    #[test]
    fn test_wants_tool_use() {
        let usage = TokenUsage::new(100, 50);

        // With ToolUse finish reason
        let response1 = LLMResponse::new("", "model", usage.clone(), FinishReason::ToolUse);
        assert!(response1.wants_tool_use());

        // With tool_calls even if finish_reason is Stop
        let response2 = LLMResponse::new("", "model", usage.clone(), FinishReason::Stop)
            .with_tool_calls(vec![ToolCall::new("id", "tool", serde_json::json!({}))]);
        assert!(response2.wants_tool_use());

        // No tools
        let response3 = LLMResponse::new("text", "model", usage.clone(), FinishReason::Stop);
        assert!(!response3.wants_tool_use());
    }
}
