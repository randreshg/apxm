//! Ollama backend implementation (local models).
//!
//! Implements the LLMBackend trait for Ollama's local model API.

use crate::llm::backends::{LLMBackend, LLMRequest, LLMResponse};
use anyhow::{Context, Result};
use apxm_core::types::{FinishReason, ModelCapabilities, ModelInfo, TokenUsage};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "gpt-oss:120b-cloud";

/// Known Ollama options that should be passed as integers
const INT_OPTIONS: &[&str] = &[
    "num_ctx",
    "num_gpu",
    "num_thread",
    "num_keep",
    "num_predict",
    "num_batch",
    "main_gpu",
    "seed",
    "mirostat",
];

/// Known Ollama options that should be passed as floats
const FLOAT_OPTIONS: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "tfs_z",
    "typical_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "mirostat_tau",
    "mirostat_eta",
];

/// Ollama LLM backend (local models).
pub struct OllamaBackend {
    model: String,
    base_url: String,
    client: reqwest::Client,
    /// Ollama runtime options (num_ctx, num_gpu, etc.)
    ollama_options: serde_json::Map<String, serde_json::Value>,
}

impl OllamaBackend {
    /// Create a new Ollama backend.
    pub async fn new(_api_key: &str, config: Option<serde_json::Value>) -> Result<Self> {
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

        // Parse all Ollama options from config
        // These will be passed through to the Ollama API
        let mut ollama_options = serde_json::Map::new();
        
        if let Some(config_obj) = config.as_ref().and_then(|c| c.as_object()) {
            for (key, value) in config_obj {
                // Skip non-option fields
                if key == "model" || key == "base_url" {
                    continue;
                }
                
                // Convert string values to appropriate types for Ollama
                let converted_value = if let Some(s) = value.as_str() {
                    if INT_OPTIONS.contains(&key.as_str()) {
                        // Parse as integer
                        s.parse::<i64>()
                            .map(serde_json::Value::from)
                            .unwrap_or_else(|_| value.clone())
                    } else if FLOAT_OPTIONS.contains(&key.as_str()) {
                        // Parse as float
                        s.parse::<f64>()
                            .map(serde_json::Value::from)
                            .unwrap_or_else(|_| value.clone())
                    } else if s == "true" || s == "false" {
                        // Parse as bool
                        serde_json::Value::Bool(s == "true")
                    } else {
                        value.clone()
                    }
                } else {
                    value.clone()
                };
                
                ollama_options.insert(key.clone(), converted_value);
            }
        }

        tracing::debug!(
            model = %model,
            base_url = %base_url,
            ollama_options = ?ollama_options,
            "Creating Ollama backend"
        );

        Ok(OllamaBackend {
            model,
            base_url,
            client: reqwest::Client::new(),
            ollama_options,
        })
    }

    /// Build request body for Ollama API.
    fn build_request_body(&self, request: &LLMRequest) -> serde_json::Value {
        let prompt = if let Some(system) = &request.system_prompt {
            format!("System: {}\n\nUser: {}", system, request.prompt)
        } else {
            request.prompt.clone()
        };

        // Start with configured Ollama options
        let mut options = serde_json::Value::Object(self.ollama_options.clone());

        // Override with request-specific options
        options["temperature"] = json!(request.temperature);

        if let Some(max_tokens) = request.max_tokens {
            options["num_predict"] = json!(max_tokens);
        }

        if let Some(top_p) = request.top_p {
            options["top_p"] = json!(top_p);
        }

        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": options
        });

        tracing::debug!(
            options = %options,
            "Building Ollama request"
        );

        body
    }

    /// Parse Ollama API response.
    fn parse_response(&self, response: OllamaResponse) -> Result<LLMResponse> {
        // Ollama uses approximate token counts
        let input_tokens = response.prompt_eval_count.unwrap_or(0);
        let output_tokens = response.eval_count.unwrap_or(0);

        let usage = TokenUsage::new(input_tokens, output_tokens);
        let finish_reason = if response.done {
            FinishReason::Stop
        } else {
            FinishReason::Unknown
        };

        Ok(LLMResponse::new(
            response.response,
            &self.model,
            usage,
            finish_reason,
        ))
    }
}

#[async_trait]
impl LLMBackend for OllamaBackend {
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse> {
        request.validate()?;

        let body = self.build_request_body(&request);
        let url = format!("{}/api/generate", self.base_url);

        tracing::debug!(
            model = %self.model,
            url = %url,
            "Sending request to Ollama"
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Ollama")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("Ollama API error (status {}): {}", status, error_text);
        }

        let api_response: OllamaResponse = response
            .json()
            .await
            .context("Failed to parse Ollama response")?;

        self.parse_response(api_response)
    }

    fn name(&self) -> &str {
        "ollama"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn health_check(&self) -> Result<()> {
        let url = format!("{}/api/tags", self.base_url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to connect to Ollama")?;

        if !response.status().is_success() {
            anyhow::bail!("Ollama health check failed: {}", response.status());
        }

        Ok(())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/api/tags", self.base_url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to list Ollama models")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list models: {}", response.status());
        }

        let tags: OllamaTags = response.json().await?;

        let models = tags
            .models
            .into_iter()
            .map(|m| ModelInfo {
                id: m.name.clone(),
                name: m.name,
                context_window: 4096,
                supports_vision: false,
                supports_functions: false,
            })
            .collect();

        Ok(models)
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            streaming: true,
            vision: false,
            functions: false,
            batch: false,
            fine_tuning: false,
        }
    }
}

// Ollama API response types
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
    done: bool,
    prompt_eval_count: Option<usize>,
    eval_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct OllamaTags {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
}
