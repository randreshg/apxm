//! OpenAI backend implementation.
//!
//! Implements the LLMBackend trait for OpenAI's API, supporting modern OpenAI
//! model identifiers (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, etc.).
//!
//! This file updates the provider default model and the list of known models
//! surfaced by `list_models()` to reflect more recent model names.

use crate::backends::{LLMBackend, LLMRequest, LLMResponse};
use anyhow::{Context, Result};
use apxm_core::types::{FinishReason, ModelCapabilities, ModelInfo, TokenUsage};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-4o-mini";

/// OpenAI LLM backend.
pub struct OpenAIBackend {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl OpenAIBackend {
    /// Create a new OpenAI backend.
    pub async fn new(api_key: &str, config: Option<serde_json::Value>) -> Result<Self> {
        let model = config
            .as_ref()
            .and_then(|c| c.get("model"))
            .and_then(|m| m.as_str())
            .unwrap_or(DEFAULT_MODEL)
            .to_string();

        let base_url = config
            .as_ref()
            .and_then(|c| c.get("base_url"))
            .and_then(|u| u.as_str())
            .unwrap_or(DEFAULT_BASE_URL)
            .to_string();

        Ok(OpenAIBackend {
            api_key: api_key.to_string(),
            model,
            base_url,
            client: reqwest::Client::new(),
        })
    }

    /// Build request body for OpenAI API.
    fn build_request_body(&self, request: &LLMRequest) -> serde_json::Value {
        let mut body = json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            "temperature": request.temperature,
        });

        // Add system prompt if provided
        if let Some(system) = &request.system_prompt {
            body["messages"] = json!([
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]);
        }

        // Add optional parameters
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }

        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }

        if let Some(freq_penalty) = request.frequency_penalty {
            body["frequency_penalty"] = json!(freq_penalty);
        }

        if let Some(pres_penalty) = request.presence_penalty {
            body["presence_penalty"] = json!(pres_penalty);
        }

        if !request.stop_sequences.is_empty() {
            body["stop"] = json!(request.stop_sequences);
        }

        body
    }

    /// Parse OpenAI API response.
    fn parse_response(&self, response: OpenAIResponse) -> Result<LLMResponse> {
        let choice = response.choices.first().context("No choices in response")?;

        let content = choice.message.content.clone();
        let finish_reason = FinishReason::from_string(&choice.finish_reason);

        let usage = TokenUsage::new(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        );

        Ok(LLMResponse::new(content, &self.model, usage, finish_reason))
    }
}

#[async_trait]
impl LLMBackend for OpenAIBackend {
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse> {
        request.validate()?;

        let body = self.build_request_body(&request);
        let url = format!("{}/chat/completions", self.base_url);

        tracing::debug!(
            model = %self.model,
            url = %url,
            "Sending request to OpenAI"
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to OpenAI")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("OpenAI API error (status {}): {}", status, error_text);
        }

        let api_response: OpenAIResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        self.parse_response(api_response)
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn health_check(&self) -> Result<()> {
        let url = format!("{}/models", self.base_url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to connect to OpenAI")?;

        if !response.status().is_success() {
            anyhow::bail!("OpenAI health check failed: {}", response.status());
        }

        Ok(())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        // Derive model metadata from the centralized enum helpers in models.rs
        Ok(crate::backends::OpenAIModel::all_models()
            .iter()
            .map(|m| m.to_model_info())
            .collect())
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            streaming: true,
            vision: self.model.contains("vision")
                || self.model.contains("turbo")
                || self.model.contains("gpt-4o"),
            functions: true,
            batch: false,
            fine_tuning: self.model.starts_with("gpt-3.5"),
        }
    }
}

// OpenAI API response types
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct Message {
    content: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::LLMRequest;

    #[test]
    fn test_build_request_body_basic() {
        let backend = OpenAIBackend {
            api_key: "test".to_string(),
            model: "gpt-4".to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
        };

        let request = LLMRequest::new("Hello").with_temperature(0.9);

        let body = backend.build_request_body(&request);

        assert_eq!(body["model"], "gpt-4");
        assert_eq!(body["temperature"], 0.9);
        assert_eq!(body["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_build_request_body_with_system() {
        let backend = OpenAIBackend {
            api_key: "test".to_string(),
            model: "gpt-4-turbo".to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
        };

        let request = LLMRequest::new("Hello")
            .with_system_prompt("You are helpful")
            .with_temperature(0.5);

        let body = backend.build_request_body(&request);

        assert_eq!(body["model"], "gpt-4-turbo");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "Hello");
    }
}
