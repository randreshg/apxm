//! Google AI backend implementation.
//!
//! Implements the LLMBackend trait for Google's Gemini API.

use crate::llm::backends::{LLMBackend, LLMRequest, LLMResponse};
use anyhow::{Context, Result};
use apxm_core::types::{FinishReason, ModelCapabilities, ModelInfo, TokenUsage};
use apxm_core::{log_debug, log_error};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MODEL: &str = "gemini-flash-latest";

/// Google AI LLM backend.
pub struct GoogleBackend {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl GoogleBackend {
    /// Create a new Google backend.
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

        Ok(GoogleBackend {
            api_key: api_key.to_string(),
            model,
            base_url,
            client: reqwest::Client::new(),
        })
    }

    /// Build request body for Google AI API.
    fn build_request_body(&self, request: &LLMRequest) -> serde_json::Value {
        let contents = vec![json!({
            "role": "user",
            "parts": [{ "text": request.prompt }]
        })];

        let mut body = json!({
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens.unwrap_or(2048),
                "topP": request.top_p.unwrap_or(0.95),
                "stopSequences": request.stop_sequences,
            }
        });

        if let Some(system) = &request.system_prompt {
            body.as_object_mut()
                .expect("generation payload must be object")
                .insert(
                    "systemInstruction".to_string(),
                    json!({
                        "parts": [{ "text": system }]
                    }),
                );
        }

        body
    }

    /// Parse Google API response.
    fn parse_response(&self, response: GoogleResponse) -> Result<LLMResponse> {
        let candidate = response
            .candidates
            .first()
            .context("No candidates in response")?;

        let part = candidate
            .content
            .parts
            .first()
            .context("No parts in content")?;

        let text = part.text.clone();
        let finish_reason = FinishReason::from_string(&candidate.finish_reason);

        // Google doesn't provide detailed token usage in all responses
        let usage = TokenUsage::new(0, text.split_whitespace().count());

        Ok(LLMResponse::new(text, &self.model, usage, finish_reason))
    }
}

#[async_trait]
impl LLMBackend for GoogleBackend {
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse> {
        request.validate()?;

        let body = self.build_request_body(&request);
        let url = format!("{}/models/{}:generateContent", self.base_url, self.model);

        log_debug!(
            "models::google",
            model = %self.model,
            "Sending request to Google AI"
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Google")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            log_error!(
                "models::google",
                status = %status,
                body = %error_text,
                model = %self.model,
                "Google API request failed"
            );
            anyhow::bail!("Google API error (status {}): {}", status, error_text);
        }

        let api_response: GoogleResponse = response
            .json()
            .await
            .context("Failed to parse Google response")?;

        self.parse_response(api_response)
    }

    fn name(&self) -> &str {
        "google"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn health_check(&self) -> Result<()> {
        let url = format!("{}/models/{}", self.base_url, self.model);

        let response = self
            .client
            .get(&url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await
            .context("Failed to connect to Google")?;

        if !response.status().is_success() {
            anyhow::bail!("Google health check failed: {}", response.status());
        }

        Ok(())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        // Derive Google model metadata from the centralized enum helpers in models.rs.
        Ok(crate::llm::backends::GoogleModel::all_models()
            .iter()
            .map(|m| m.to_model_info())
            .collect())
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            streaming: true,
            vision: true,
            functions: true,
            batch: false,
            fine_tuning: false,
        }
    }
}

// Google API response types
#[derive(Debug, Deserialize)]
struct GoogleResponse {
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: Content,
    #[serde(rename = "finishReason")]
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Deserialize)]
struct Part {
    text: String,
}
