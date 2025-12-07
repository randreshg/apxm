//! LLM backend trait and request/response types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use crate::types::Value;

/// Request parameters for an LLM generation call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LLMRequest {
    /// Main user prompt.
    pub prompt: String,
    /// Optional system prompt for steering behaviour.
    pub system_prompt: Option<String>,
    /// Temperature controls randomness (0-2).
    pub temperature: f64,
    /// Optional maximum tokens to generate.
    pub max_tokens: Option<usize>,
    /// Stop sequences for halting generation.
    pub stop_sequences: Vec<String>,
    /// Arbitrary metadata for the backend.
    pub metadata: HashMap<String, Value>,
}

impl LLMRequest {
    /// Creates a new request with sane defaults.
    pub fn new(prompt: impl Into<String>) -> Self {
        LLMRequest {
            prompt: prompt.into(),
            system_prompt: None,
            temperature: 0.7,
            max_tokens: None,
            stop_sequences: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Adds a system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets max tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

/// Token usage metrics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl TokenUsage {
    /// Creates a new token usage summary.
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        let total_tokens = prompt_tokens + completion_tokens;
        TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        }
    }
}

/// Reasons a generation finished.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    Other(String),
}

/// Response from an LLM backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LLMResponse {
    pub text: String,
    pub model: String,
    pub usage: TokenUsage,
    pub finish_reason: FinishReason,
    pub metadata: HashMap<String, Value>,
}

impl LLMResponse {
    /// Convenience constructor for plain text responses.
    pub fn new(text: impl Into<String>, model: impl Into<String>, usage: TokenUsage) -> Self {
        LLMResponse {
            text: text.into(),
            model: model.into(),
            usage,
            finish_reason: FinishReason::Stop,
            metadata: HashMap::new(),
        }
    }
}

/// Stream of LLM responses (e.g., for streaming generation).
pub type LLMResponseStream =
    Box<dyn Iterator<Item = Result<LLMResponse, LLMError>> + Send + Sync + 'static>;

/// Errors returned by LLM backends.
#[derive(Debug, Error, Clone, PartialEq, Serialize, Deserialize)]
pub enum LLMError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("backend unavailable: {0}")]
    Unavailable(String),
    #[error("generation failed: {0}")]
    GenerationFailed(String),
}

/// Contract for all LLM backends.
pub trait LLMBackend: Send + Sync {
    /// Performs a single-shot generation.
    fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError>;

    /// Performs streaming generation.
    fn generate_stream(&self, request: LLMRequest) -> Result<LLMResponseStream, LLMError>;

    /// Name of the underlying model/backend (for observability).
    fn model_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct MockLLM {
        model: String,
    }

    impl LLMBackend for MockLLM {
        fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError> {
            Ok(LLMResponse::new(
                format!("echo: {}", request.prompt),
                self.model.clone(),
                TokenUsage::new(1, 2),
            ))
        }

        fn generate_stream(&self, request: LLMRequest) -> Result<LLMResponseStream, LLMError> {
            let first = self.generate(request)?;
            Ok(Box::new(vec![Ok(first)].into_iter()))
        }

        fn model_name(&self) -> &str {
            &self.model
        }
    }

    #[test]
    fn test_request_builder() {
        let req = LLMRequest::new("hello")
            .with_system_prompt("sys")
            .with_temperature(0.2)
            .with_max_tokens(50);
        assert_eq!(req.prompt, "hello");
        assert_eq!(req.system_prompt.as_deref(), Some("sys"));
        assert_eq!(req.temperature, 0.2);
        assert_eq!(req.max_tokens, Some(50));
    }

    #[test]
    fn test_generate_and_stream() {
        let backend = MockLLM {
            model: "mock".into(),
        };
        let req = LLMRequest::new("world");
        let resp = backend.generate(req.clone()).expect("generate");
        assert_eq!(resp.text, "echo: world");
        assert_eq!(resp.model, "mock");
        let mut stream = backend.generate_stream(req).expect("stream");
        let streamed = stream.next().unwrap().expect("item");
        assert_eq!(streamed.text, resp.text);
    }

    #[test]
    fn test_serialization_round_trip() {
        let response = LLMResponse::new(
            "hi",
            "mock",
            TokenUsage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            },
        );
        let json = serde_json::to_string(&response).expect("serialize");
        let restored: LLMResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.text, "hi");
        assert_eq!(restored.usage.total_tokens, 3);
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn trait_is_send_sync() {
        assert_send_sync::<Box<dyn LLMBackend>>();
    }
}
