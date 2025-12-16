//! LLM model provider types.
//!
//! Core types for LLM backends, requests, and responses that are shared
//! across the apxm ecosystem.

mod response;
pub use response::LLMResponse;

use serde::{Deserialize, Serialize};

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
            FinishReason::Unknown => write!(f, "unknown"),
        }
    }
}

impl FinishReason {
    /// Parse finish reason from string.
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "timeout" => FinishReason::Timeout,
            "error" => FinishReason::Error,
            "content_filter" => FinishReason::ContentFilter,
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
        assert_eq!(FinishReason::from_string("length"), FinishReason::Length);
        assert_eq!(
            FinishReason::from_string("unknown_value"),
            FinishReason::Unknown
        );
    }
}
