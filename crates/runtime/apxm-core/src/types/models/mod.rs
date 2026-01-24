//! LLM model provider types.
//!
//! Core types for LLM backends, requests, and responses that are shared
//! across the apxm ecosystem.

mod response;
pub use response::LLMResponse;

use serde::{Deserialize, Serialize};

/// A tool call made by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call (used to match results)
    pub id: String,
    /// Name of the tool to invoke
    pub name: String,
    /// Arguments for the tool as JSON
    pub args: serde_json::Value,
}

impl ToolCall {
    /// Create a new tool call.
    pub fn new(id: impl Into<String>, name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            args,
        }
    }
}

/// Result of a tool execution to send back to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// ID matching the original tool call
    pub tool_call_id: String,
    /// Result content (usually stringified)
    pub content: String,
    /// Whether the tool execution was successful
    #[serde(default = "default_success")]
    pub success: bool,
}

fn default_success() -> bool {
    true
}

impl ToolResult {
    /// Create a successful tool result.
    pub fn success(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            success: true,
        }
    }

    /// Create a failed tool result.
    pub fn error(tool_call_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: error.into(),
            success: false,
        }
    }
}

/// Information about an LLM model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (e.g., "gpt-4", "claude-3-opus-20240229")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Context window size in tokens
    pub context_window: usize,
    /// Whether this model supports vision/images
    pub supports_vision: bool,
    /// Whether this model supports function calling
    pub supports_functions: bool,
}

/// Capabilities supported by an LLM backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Supports streaming responses
    pub streaming: bool,
    /// Supports vision/image inputs
    pub vision: bool,
    /// Supports function/tool calling
    pub functions: bool,
    /// Supports batch API (for cost optimization)
    pub batch: bool,
    /// Supports fine-tuning
    pub fine_tuning: bool,
}

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of tokens in the input
    pub input_tokens: usize,
    /// Number of tokens in the output
    pub output_tokens: usize,
    /// Total tokens used (input + output)
    pub total_tokens: usize,
}

impl TokenUsage {
    /// Create token usage with the given counts.
    pub fn new(input: usize, output: usize) -> Self {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
        }
    }
}

/// Reason why generation finished.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FinishReason {
    /// Model reached the stop sequence
    Stop,
    /// Maximum tokens reached
    Length,
    /// Request timeout
    Timeout,
    /// Error during generation
    Error,
    /// Content filter triggered
    ContentFilter,
    /// Model wants to use tools
    ToolUse,
    /// Unknown reason
    Unknown,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::Timeout => write!(f, "timeout"),
            FinishReason::Error => write!(f, "error"),
            FinishReason::ContentFilter => write!(f, "content_filter"),
            FinishReason::ToolUse => write!(f, "tool_use"),
            FinishReason::Unknown => write!(f, "unknown"),
        }
    }
}

impl FinishReason {
    /// Parse finish reason from string.
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "stop" | "end_turn" => FinishReason::Stop,
            "length" | "max_tokens" => FinishReason::Length,
            "timeout" => FinishReason::Timeout,
            "error" => FinishReason::Error,
            "content_filter" => FinishReason::ContentFilter,
            "tool_use" | "tool_calls" | "function_call" => FinishReason::ToolUse,
            _ => FinishReason::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage_calculation() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
    }

    #[test]
    fn test_finish_reason_parsing() {
        assert_eq!(FinishReason::from_string("stop"), FinishReason::Stop);
        assert_eq!(FinishReason::from_string("Stop"), FinishReason::Stop);
        assert_eq!(FinishReason::from_string("end_turn"), FinishReason::Stop);
        assert_eq!(FinishReason::from_string("length"), FinishReason::Length);
        assert_eq!(FinishReason::from_string("max_tokens"), FinishReason::Length);
        assert_eq!(FinishReason::from_string("tool_use"), FinishReason::ToolUse);
        assert_eq!(
            FinishReason::from_string("tool_calls"),
            FinishReason::ToolUse
        );
        assert_eq!(
            FinishReason::from_string("function_call"),
            FinishReason::ToolUse
        );
        assert_eq!(
            FinishReason::from_string("unknown_value"),
            FinishReason::Unknown
        );
    }

    #[test]
    fn test_tool_call_creation() {
        let call = ToolCall::new(
            "call_123",
            "bash",
            serde_json::json!({"command": "ls -la"}),
        );
        assert_eq!(call.id, "call_123");
        assert_eq!(call.name, "bash");
        assert_eq!(call.args["command"], "ls -la");
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("call_123", "file1.txt\nfile2.txt");
        assert_eq!(result.tool_call_id, "call_123");
        assert!(result.success);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("call_123", "Permission denied");
        assert_eq!(result.tool_call_id, "call_123");
        assert!(!result.success);
        assert!(result.content.contains("Permission denied"));
    }
}
