//! Mock LLM backend for deterministic testing.
//!
//! `MockLLMBackend` lets you configure pre-programmed responses for testing
//! without making real network calls. It supports:
//!
//! - Static responses (same content every call)
//! - Pattern-matched responses (different answers per prompt keyword)
//! - Call counting and inspection for assertion
//! - Tool call simulation
//! - Configurable failure injection
//!
//! # Usage
//!
//! ```rust,ignore
//! use apxm_backends::llm::backends::mock::{MockLLMBackend, MockResponse};
//!
//! // Static response mock
//! let mock = MockLLMBackend::static_response("The answer is 42.");
//!
//! // Pattern-based mock
//! let mock = MockLLMBackend::new()
//!     .when_prompt_contains("summarize", "Summary: short text")
//!     .when_prompt_contains("translate", "Translation: bonjour")
//!     .default_response("I don't know");
//!
//! // Register in runtime
//! runtime.llm_registry().register("mock", mock).unwrap();
//! runtime.llm_registry().set_default("mock").unwrap();
//! ```

use super::{LLMRequest, LLMResponse};
use super::traits::LLMBackend;
use apxm_core::types::{FinishReason, ModelCapabilities, ModelInfo, TokenUsage};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

/// A recorded LLM call for inspection in tests.
#[derive(Debug, Clone)]
pub struct RecordedCall {
    /// The prompt that was sent
    pub prompt: String,
    /// The system prompt (if any)
    pub system: Option<String>,
    /// Model name requested
    pub model: String,
    /// Temperature used
    pub temperature: f32,
}

/// Configurable response for pattern-based mocking.
#[derive(Debug, Clone)]
pub struct MockResponse {
    /// Content to return
    pub content: String,
    /// Simulated input token count (default: 10)
    pub input_tokens: usize,
    /// Simulated output token count (default: 20)
    pub output_tokens: usize,
    /// Finish reason (default: Stop)
    pub finish_reason: FinishReason,
}

impl MockResponse {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            input_tokens: 10,
            output_tokens: 20,
            finish_reason: FinishReason::Stop,
        }
    }

    pub fn with_tokens(mut self, input: usize, output: usize) -> Self {
        self.input_tokens = input;
        self.output_tokens = output;
        self
    }
}

/// A match rule: if the prompt contains `keyword`, return `response`.
#[derive(Clone)]
struct PatternRule {
    keyword: String,
    response: MockResponse,
}

/// LLM backend that returns configurable pre-programmed responses.
///
/// Thread-safe and suitable for concurrent test scenarios.
#[derive(Clone)]
pub struct MockLLMBackend {
    name: String,
    model: String,
    default_response: MockResponse,
    patterns: Vec<PatternRule>,
    /// Shared call log for test assertions
    calls: Arc<Mutex<Vec<RecordedCall>>>,
    /// If Some, return this error on every call (failure injection)
    fail_with: Option<String>,
}

impl MockLLMBackend {
    /// Create a new mock with a configurable default response.
    pub fn new() -> Self {
        Self {
            name: "mock".to_string(),
            model: "mock-model".to_string(),
            default_response: MockResponse::new("Mock response."),
            patterns: Vec::new(),
            calls: Arc::new(Mutex::new(Vec::new())),
            fail_with: None,
        }
    }

    /// Create a mock that always returns `content`.
    pub fn static_response(content: impl Into<String>) -> Self {
        Self::new().default(MockResponse::new(content))
    }

    /// Set the mock's display name.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the model name returned in responses.
    pub fn model_name(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the default response (used when no pattern matches).
    pub fn default(mut self, response: MockResponse) -> Self {
        self.default_response = response;
        self
    }

    /// Add a pattern rule: if the prompt contains `keyword`, return `content`.
    pub fn when_prompt_contains(
        mut self,
        keyword: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        self.patterns.push(PatternRule {
            keyword: keyword.into(),
            response: MockResponse::new(content),
        });
        self
    }

    /// Add a pattern rule with a fully configured response.
    pub fn when_prompt_contains_response(
        mut self,
        keyword: impl Into<String>,
        response: MockResponse,
    ) -> Self {
        self.patterns.push(PatternRule {
            keyword: keyword.into(),
            response,
        });
        self
    }

    /// Configure this mock to always fail with the given error message.
    pub fn always_fail(mut self, error: impl Into<String>) -> Self {
        self.fail_with = Some(error.into());
        self
    }

    /// Return all recorded calls for test assertions.
    pub fn recorded_calls(&self) -> Vec<RecordedCall> {
        self.calls.lock().unwrap().clone()
    }

    /// Return the number of times `generate` was called.
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    /// Clear the recorded call history.
    pub fn reset(&self) {
        self.calls.lock().unwrap().clear();
    }

    /// Select the response for a given prompt.
    fn select_response(&self, prompt: &str) -> &MockResponse {
        for rule in &self.patterns {
            if prompt.contains(&rule.keyword) {
                return &rule.response;
            }
        }
        &self.default_response
    }
}

impl Default for MockLLMBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMBackend for MockLLMBackend {
    async fn generate(&self, request: LLMRequest) -> anyhow::Result<LLMResponse> {
        // Record the call
        self.calls.lock().unwrap().push(RecordedCall {
            prompt: request.prompt.clone(),
            system: request.system_prompt.clone(),
            model: self.model.clone(),
            temperature: request.temperature as f32,
        });

        // Failure injection
        if let Some(ref err) = self.fail_with {
            return Err(anyhow::anyhow!("{}", err));
        }

        let resp = self.select_response(&request.prompt);
        Ok(LLMResponse::new(
            resp.content.clone(),
            self.model.clone(),
            TokenUsage::new(resp.input_tokens, resp.output_tokens),
            resp.finish_reason.clone(),
        ))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn health_check(&self) -> anyhow::Result<()> {
        if let Some(ref err) = self.fail_with {
            return Err(anyhow::anyhow!("Mock backend configured to fail: {}", err));
        }
        Ok(())
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        Ok(vec![ModelInfo {
            id: self.model.clone(),
            name: format!("Mock: {}", self.model),
            context_window: 128_000,
            supports_functions: true,
            supports_vision: false,
        }])
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            streaming: false,
            vision: false,
            functions: true,
            batch: false,
            fine_tuning: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_static_response() {
        let mock = MockLLMBackend::static_response("Hello, world!");
        let req = LLMRequest::new("test");
        let resp = mock.generate(req).await.unwrap();
        assert_eq!(resp.content, "Hello, world!");
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn test_pattern_matching() {
        let mock = MockLLMBackend::new()
            .when_prompt_contains("summarize", "Summary here.")
            .when_prompt_contains("translate", "Translation here.");

        let req_sum = LLMRequest::new("Please summarize this text.");
        let resp_sum = mock.generate(req_sum).await.unwrap();
        assert_eq!(resp_sum.content, "Summary here.");

        let req_trans = LLMRequest::new("Please translate to French.");
        let resp_trans = mock.generate(req_trans).await.unwrap();
        assert_eq!(resp_trans.content, "Translation here.");

        let req_other = LLMRequest::new("What is the weather?");
        let resp_other = mock.generate(req_other).await.unwrap();
        assert_eq!(resp_other.content, "Mock response.");
    }

    #[tokio::test]
    async fn test_failure_injection() {
        let mock = MockLLMBackend::new().always_fail("Backend unavailable");
        let req = LLMRequest::new("test");
        let result = mock.generate(req).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Backend unavailable"));
    }

    #[tokio::test]
    async fn test_call_recording() {
        let mock = MockLLMBackend::static_response("ok");
        let req1 = LLMRequest::new("first call");
        let req2 = LLMRequest::new("second call");
        mock.generate(req1).await.unwrap();
        mock.generate(req2).await.unwrap();

        let calls = mock.recorded_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].prompt, "first call");
        assert_eq!(calls[1].prompt, "second call");
    }

    #[tokio::test]
    async fn test_health_check_ok() {
        let mock = MockLLMBackend::new();
        assert!(mock.health_check().await.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_fails_when_configured() {
        let mock = MockLLMBackend::new().always_fail("down");
        assert!(mock.health_check().await.is_err());
    }

    #[tokio::test]
    async fn test_concurrent_calls_are_safe() {
        let mock = Arc::new(MockLLMBackend::static_response("concurrent"));
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let m = mock.clone();
                tokio::spawn(async move {
                    let req = LLMRequest::new(format!("call {}", i));
                    m.generate(req).await.unwrap();
                })
            })
            .collect();
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(mock.call_count(), 10);
    }
}
